import os.path as osp
import pickle

import numpy as np
import pandas as pd
import torch
from rdkit import RDLogger
from torch_geometric.data import InMemoryDataset

from feature_extraction import smiles_to_pyg_graph


SMILES_CHAR = [
    '[PAD]', 'C', '(', '=', 'O', ')', 'N', '[', '@', 'H', ']', '1', 'c',
    'n', '/', '2', '#', 'S', 's', '+', '-', '\\', '3', '4', 'l', 'F',
    'o', 'I', 'B', 'r', 'P', '5', '6', 'i', '7', '8', '9', '%', '0', 'p',
]
RDLogger.DisableLog('rdApp.*')

columns = [
    'Smiles',
    'Molecular Weight',
    'Exact Mass',
    'XLogP3',
    'Heavy Atom Count',
    'Ring Count',
    'Hydrogen Bond Acceptor Count',
    'Hydrogen Bond Donor Count',
    'Rotatable Bond Count',
    'Topological Polar Surface Area',
]

DESCRIPTOR_COLUMNS = [
    'Molecular Weight',
    'XLogP3',
    'Heavy Atom Count',
    'Ring Count',
    'Hydrogen Bond Acceptor Count',
    'Hydrogen Bond Donor Count',
    'Rotatable Bond Count',
    'Topological Polar Surface Area',
]

COLUMN_ALIASES = {
    'Smiles': ['Smiles', 'smiles', 'SMILES'],
    'Uniprot': ['Uniprot', 'POI_Uniprot', 'POI Uniprot', 'poi_uniprot'],
    'E3 ligase Uniprot': [
        'E3 ligase Uniprot',
        'E3_Uniprot',
        'E3 Uniprot',
        'e3_uniprot',
    ],
    'label': ['label', 'Label'],
}

FRAGMENT_COLUMN_ALIASES = {
    'Warhead Smiles': ['Warhead Smiles', 'warhead smiles', 'warhead_smiles'],
    'Linker Smiles': ['Linker Smiles', 'linker smiles', 'linker_smiles'],
    'E3 Ligand Smiles': [
        'E3 Ligand Smiles',
        'e3 ligand smiles',
        'e3_ligand_smiles',
        'e3_ligase_smiles',
    ],
}


def _first_existing_column(df, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _copy_first_available_column(df, target, aliases):
    if target in df.columns:
        return

    alias = _first_existing_column(df, aliases)
    if alias is not None:
        df[target] = df[alias]
        return

    raise KeyError(
        f"Missing required column '{target}'. "
        f"Accepted aliases: {aliases}"
    )


def standardize_protac_dataframe(df):
    df = df.copy()

    for target, aliases in COLUMN_ALIASES.items():
        if target == 'label':
            continue
        _copy_first_available_column(df, target, aliases)

    for target, aliases in FRAGMENT_COLUMN_ALIASES.items():
        alias = _first_existing_column(df, aliases)
        if alias is not None and target not in df.columns:
            df[target] = df[alias]

    label_col = _first_existing_column(df, COLUMN_ALIASES['label'])
    if label_col is not None:
        df['label'] = pd.to_numeric(df[label_col], errors='coerce').astype(int)

    df['Smiles'] = df['Smiles'].astype(str)
    df['Uniprot'] = df['Uniprot'].map(normalize_protein_id)
    df['E3 ligase Uniprot'] = df['E3 ligase Uniprot'].map(normalize_protein_id)
    return df


def trans_smiles(x, target_size=128, pad_value=0):
    temp = list(x)
    temp = [SMILES_CHAR.index(i) if i in SMILES_CHAR else len(SMILES_CHAR) for i in temp]
    if len(temp) < target_size:
        temp = temp + [pad_value] * (target_size - len(temp))
    else:
        temp = temp[:target_size]
    return temp


def normalize_protein_id(value):
    if pd.isna(value):
        return None

    protein_id = str(value).strip()
    if protein_id == '' or protein_id.lower() in {'nan', 'none', 'null'}:
        return None
    return protein_id


def _load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_first_existing_pickle(root, file_names):
    for file_name in file_names:
        path = osp.join(root, file_name)
        if osp.exists(path):
            return _load_pickle(path), file_name

    raise FileNotFoundError(
        f"None of these feature map files exist in {root}: {file_names}"
    )


def _as_float_tensor(feature, key, source_name):
    if isinstance(feature, dict):
        if 'feature' not in feature:
            raise KeyError(f"Missing 'feature' in {source_name} record for {key}")
        feature = feature['feature']

    if torch.is_tensor(feature):
        tensor = feature.detach().clone()
    elif isinstance(feature, np.ndarray):
        tensor = torch.from_numpy(feature)
    else:
        tensor = torch.tensor(feature)

    tensor = tensor.float().view(-1)
    if tensor.numel() == 0:
        raise ValueError(f"Empty feature in {source_name} for {key}")
    return tensor


def _check_feature_dims(features, source_name):
    dims = {int(feature.numel()) for feature in features}
    if len(dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions in {source_name}: {sorted(dims)}")


def load_map_features(feature_map, keys, source_name, normalize_key=None):
    features = []
    missing = []

    for raw_key in keys:
        key = normalize_key(raw_key) if normalize_key is not None else str(raw_key).strip()
        if key is None or key not in feature_map:
            missing.append(raw_key)
            continue
        features.append(_as_float_tensor(feature_map[key], key, source_name))

    if missing:
        sample = ', '.join(str(x) for x in missing[:10])
        raise KeyError(
            f"{source_name} is missing {len(missing)} records used by the dataset. "
            f"Examples: {sample}"
        )

    _check_feature_dims(features, source_name)
    return features


def load_saprot_features(root, file_name, protein_ids):
    path = osp.join(root, file_name)
    if not osp.exists(path):
        raise FileNotFoundError(
            f"SaProt feature map not found: {path}. "
            "Run extract_protac_saprot_features.py on the server first and copy the pkl here."
        )

    feature_map = _load_pickle(path)
    return load_map_features(
        feature_map=feature_map,
        keys=protein_ids,
        source_name=file_name,
        normalize_key=normalize_protein_id,
    )


class PROTACData(InMemoryDataset):
    """
    PyTorch Geometric dataset for PROTAC data.

    The small molecule branch is still stored as a PyG graph. Protein graph files
    are no longer loaded; POI/E3 structural information comes from precomputed
    SaProt feature maps.
    """

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.root = root
        self.name = name
        self.raw_files = [osp.join(root, f'{name}.csv')]
        super(PROTACData, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        # These legacy ESM tensors are kept on disk for compatibility but are no
        # longer loaded into memory because the current model uses SaProt-only proteins.
        self.e3_sequences = None
        self.poi_sequences = None
        self.labels = torch.load(self.processed_paths[1])
        self.poi_saprot = torch.load(self.processed_paths[2])
        self.e3_saprot = torch.load(self.processed_paths[3])
        self.protac_smiles = torch.load(self.processed_paths[4])
        self.descriptors = torch.load(self.processed_paths[5]).float()

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', self.name)

    @property
    def raw_file_names(self):
        return self.raw_files

    @property
    def processed_file_names(self):
        return [
            'protac.pt',
            'label.pt',
            'poi_saprot.pt',
            'e3_saprot.pt',
            'protac_smile.pt',
            'descriptor.pt',
        ]

    def process(self):
        print(f'Processing data to {self.processed_dir}')
        protac_df = standardize_protac_dataframe(pd.read_csv(self.raw_file_names[0]))

        print('Generating PROTAC graphs with rich features...')
        data_list = []
        for _, row in protac_df.iterrows():
            graph_data = smiles_to_pyg_graph(
                row['Smiles'],
                pos_enc_dim=8,
                warhead_smiles=row.get('Warhead Smiles'),
                linker_smiles=row.get('Linker Smiles'),
                e3_ligand_smiles=row.get('E3 Ligand Smiles'),
            )
            data_list.append(graph_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        smile_map, smile_map_name = _load_first_existing_pickle(
            self.root,
            [
                'molformer_map_pool.pkl',
                'molformer_map.pkl',
                'chemberta_map_pool.pkl',
                'chemberta_map.pkl',
            ],
        )

        e3_ids = protac_df['E3 ligase Uniprot'].to_list()
        poi_ids = protac_df['Uniprot'].to_list()

        if 'label' in protac_df.columns:
            label = torch.tensor(protac_df['label'].astype(int).to_list(), dtype=torch.long)
        else:
            label = None
        torch.save(label, self.processed_paths[1])

        print('Loading POI SaProt features...')
        poi_saprot = load_saprot_features(self.root, 'poi_saprot_map.pkl', poi_ids)
        torch.save(poi_saprot, self.processed_paths[2])

        print('Loading E3 SaProt features...')
        e3_saprot = load_saprot_features(self.root, 'e3_saprot_map.pkl', e3_ids)
        torch.save(e3_saprot, self.processed_paths[3])

        protac_smiles = protac_df['Smiles'].to_list()
        smile_data_list = load_map_features(smile_map, protac_smiles, smile_map_name)
        torch.save(smile_data_list, self.processed_paths[4])

        if all(col in protac_df.columns for col in DESCRIPTOR_COLUMNS):
            descriptor_df = protac_df[DESCRIPTOR_COLUMNS].copy()
            descriptor_df = descriptor_df.apply(pd.to_numeric, errors='coerce')
            descriptor_df = descriptor_df.fillna(descriptor_df.median(numeric_only=True))
            descriptor_tensor = torch.tensor(descriptor_df.values, dtype=torch.float32)
        else:
            descriptor_tensor = torch.zeros(len(protac_df), len(DESCRIPTOR_COLUMNS), dtype=torch.float32)
        torch.save(descriptor_tensor, self.processed_paths[5])

    def __repr__(self):
        return '{}()'.format(self.name)


if __name__ == '__main__':
    root = 'data/PROTAC-fine'
    dataset = PROTACData(root=root, name='protac-fine')
    print('Dataset info:')
    print('Number of samples:', len(dataset))
