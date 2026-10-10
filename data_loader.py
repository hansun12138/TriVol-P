import json
import os
import random
import warnings

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch

from data import PROTACData


def collate_fn(data_list):
    batch = {}
    batch['protac'] = Batch.from_data_list([item['protac'] for item in data_list])
    batch['e3_saprot'] = torch.stack([item['e3_saprot'] for item in data_list])
    batch['poi_saprot'] = torch.stack([item['poi_saprot'] for item in data_list])
    batch['protac_smile'] = torch.stack([item['protac_smile'] for item in data_list])
    batch['descriptor'] = torch.stack([item['descriptor'] for item in data_list])
    batch['index'] = torch.tensor([item['index'] for item in data_list], dtype=torch.long)

    labels = [item['label'] for item in data_list]
    batch['label'] = torch.stack(labels) if labels[0] is not None else None
    return batch


class PROTACDataset(Dataset):
    def __init__(self, protac, label, poi_saprot, e3_saprot, protac_smile, descriptors):
        self.protac = protac
        if label is not None:
            self.label = label
        self.poi_saprot = poi_saprot
        self.e3_saprot = e3_saprot
        self.protac_smile = protac_smile
        self.descriptors = descriptors.float()
        self.descriptor_mean = None
        self.descriptor_std = None

    def __len__(self):
        return len(self.protac)

    def fit_descriptor_normalizer(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long)
        descriptor_subset = self.descriptors[indices].float()
        mean = descriptor_subset.mean(dim=0)
        std = descriptor_subset.std(dim=0, unbiased=False).clamp_min(1e-6)
        self.descriptor_mean = mean
        self.descriptor_std = std
        return mean, std

    def clear_descriptor_normalizer(self):
        self.descriptor_mean = None
        self.descriptor_std = None

    def set_descriptor_normalizer(self, mean, std):
        self.descriptor_mean = mean.float()
        self.descriptor_std = std.float().clamp_min(1e-6)

    def get_descriptor(self, index):
        descriptor = self.descriptors[index].float()
        if self.descriptor_mean is not None and self.descriptor_std is not None:
            descriptor = (descriptor - self.descriptor_mean) / self.descriptor_std
        return descriptor

    def __getitem__(self, index):
        return {
            'index': index,
            'protac': self.protac[index],
            'poi_saprot': self.poi_saprot[index],
            'e3_saprot': self.e3_saprot[index],
            'protac_smile': self.protac_smile[index],
            'descriptor': self.get_descriptor(index),
            'label': self.label[index] if hasattr(self, 'label') else None,
        }


def _make_generator(seed):
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(num_workers=0, seed=None):
    kwargs = {
        'num_workers': num_workers,
    }
    if num_workers > 0:
        kwargs['worker_init_fn'] = _seed_worker
    return kwargs


def _drop_last_singleton(dataset_len, batch_size):
    return batch_size > 1 and dataset_len > batch_size and dataset_len % batch_size == 1


def _label_to_int(label):
    return int(label.item()) if torch.is_tensor(label) else int(label)


def get_protac_dataset(root='data/PROTAC-fine', name='protac-fine'):
    dataset_raw = PROTACData(root, name=name)
    return PROTACDataset(
        protac=dataset_raw,
        label=dataset_raw.labels,
        poi_saprot=dataset_raw.poi_saprot,
        e3_saprot=dataset_raw.e3_saprot,
        protac_smile=dataset_raw.protac_smiles,
        descriptors=dataset_raw.descriptors,
    )


def get_labels_and_groups(dataset):
    labels = []
    groups = []

    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(_label_to_int(item['label']))
        smiles = getattr(item['protac'], 'smiles', f'idx_{i}')
        groups.append(smiles)

    return np.array(labels), np.array(groups)


def _degrademaster_split_from_records(records, train_ratio=0.8, split_seed=111):
    num_label = sum(1 for record in records if int(record['label']) != -1)
    test_size = int(num_label * (1 - train_ratio))
    shuffled_indices = list(np.arange(len(records)))
    random.Random(split_seed).shuffle(shuffled_indices)

    train_indices = []
    test_indices = []
    test_cnt = 0

    for idx in shuffled_indices:
        record = records[int(idx)]
        if int(record['label']) != -1 and test_cnt < test_size:
            test_indices.append(int(idx))
            test_cnt += 1
        else:
            train_indices.append(int(idx))

    return train_indices, test_indices


def _dataset_records(dataset):
    records = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        records.append({
            'smiles': getattr(item['protac'], 'smiles', None),
            'label': _label_to_int(item['label']),
        })
    return records


def _reference_json_records(reference_json_path):
    with open(reference_json_path, 'r', encoding='utf-8') as f:
        name_dict = json.load(f)

    return [
        {
            'smiles': value.get('smiles'),
            'label': int(value.get('label')),
        }
        for value in name_dict.values()
    ]


def _record_key(record):
    return (record['smiles'], int(record['label']))


def _match_reference_records_to_dataset(dataset_records, reference_records):
    labeled_reference = [
        (global_idx, record)
        for global_idx, record in enumerate(reference_records)
        if int(record['label']) != -1
    ]

    if len(labeled_reference) < len(dataset_records):
        raise ValueError(
            'The labeled records in reference_json_path do not match the loaded dataset: '
            f'{len(labeled_reference)} labeled reference records vs {len(dataset_records)} dataset records. '
            'Regenerate reference_json_path or unset it for a dataset-order split.'
        )

    global_to_dataset = {}
    ref_pos = 0

    for dataset_idx, data_record in enumerate(dataset_records):
        data_key = _record_key(data_record)

        while ref_pos < len(labeled_reference):
            global_idx, ref_record = labeled_reference[ref_pos]
            ref_pos += 1

            if _record_key(ref_record) == data_key:
                global_to_dataset[global_idx] = dataset_idx
                break
        else:
            raise ValueError(
                'reference_json_path labeled records are not an ordered superset of the loaded dataset. '
                f'Could not match dataset row {dataset_idx}: {data_key}. '
                'If rows were edited or reordered, regenerate reference_json_path or unset it.'
            )

    skipped_count = len(labeled_reference) - len(dataset_records)
    if skipped_count:
        warnings.warn(
            'reference_json_path contains '
            f'{skipped_count} labeled records that are not in the loaded dataset; '
            f'keeping the reference split for the remaining {len(dataset_records)} records.',
            RuntimeWarning,
        )

    return global_to_dataset


def _map_reference_split_to_dataset(dataset, reference_records, train_ratio=0.8, split_seed=111):
    dataset_records = _dataset_records(dataset)
    global_to_dataset = _match_reference_records_to_dataset(
        dataset_records=dataset_records,
        reference_records=reference_records,
    )

    reference_train, reference_test = _degrademaster_split_from_records(
        reference_records,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )

    train_idx = [
        global_to_dataset[idx]
        for idx in reference_train
        if idx in global_to_dataset
    ]
    test_idx = [
        global_to_dataset[idx]
        for idx in reference_test
        if idx in global_to_dataset
    ]
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def build_degrademaster_split_indices(
    dataset,
    train_ratio=0.8,
    split_seed=111,
    reference_json_path=None,
):
    if reference_json_path:
        reference_json_path = os.path.abspath(reference_json_path)
        if not os.path.exists(reference_json_path):
            raise FileNotFoundError(f'reference_json_path not found: {reference_json_path}')

        reference_records = _reference_json_records(reference_json_path)
        return _map_reference_split_to_dataset(
            dataset=dataset,
            reference_records=reference_records,
            train_ratio=train_ratio,
            split_seed=split_seed,
        )

    records = _dataset_records(dataset)
    train_idx, test_idx = _degrademaster_split_from_records(
        records,
        train_ratio=train_ratio,
        split_seed=split_seed,
    )
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def build_group_stratified_folds(dataset, n_splits=5, seed=42):
    labels, groups = get_labels_and_groups(dataset)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    dummy_x = np.zeros(len(dataset))
    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(dummy_x, labels, groups)):
        folds.append((fold_id, train_idx, val_idx))

    return folds


def build_fold_loaders(
    dataset,
    train_idx,
    val_idx,
    batch_size=4,
    collate_fn=collate_fn,
    seed=None,
    normalize_descriptors=True,
    num_workers=0,
):
    if normalize_descriptors:
        dataset.fit_descriptor_normalizer(train_idx)
    else:
        dataset.clear_descriptor_normalizer()

    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=_drop_last_singleton(len(train_subset), batch_size),
        generator=_make_generator(seed),
        **_loader_kwargs(num_workers=num_workers, seed=seed),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        generator=_make_generator(None if seed is None else seed + 1),
        **_loader_kwargs(num_workers=num_workers, seed=None if seed is None else seed + 1),
    )
    return train_loader, val_loader


def PROTACLoader(
    root='data/PROTAC-fine',
    name='protac-fine',
    batch_size=2,
    collate_fn=collate_fn,
    train_ratio=0.8,
    seed=None,
    normalize_descriptors=True,
    split_mode='random',
    split_seed=111,
    reference_json_path=None,
    shuffle_train=True,
    num_workers=0,
):
    dataset = get_protac_dataset(root=root, name=name)

    if split_mode == 'degrademaster':
        train_idx, test_idx = build_degrademaster_split_indices(
            dataset=dataset,
            train_ratio=train_ratio,
            split_seed=split_seed,
            reference_json_path=reference_json_path,
        )
        train_size = len(train_idx)
        test_size = len(test_idx)
    else:
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size

    if train_ratio > 0.0:
        print('Cleaned Dataset: ')
        print('Total size: ', len(dataset))
        print('Train size: ', train_size)
        print('Test size: ', test_size)
    else:
        print('Test Dataset: ')
        print('Total size: ', len(dataset))

    if train_size == 0:
        dataset.clear_descriptor_normalizer()
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            generator=_make_generator(seed),
            **_loader_kwargs(num_workers=num_workers, seed=seed),
        )
        return None, test_loader

    if split_mode == 'degrademaster':
        print(f'Split mode: DegradeMaster | split_seed: {split_seed}')
        if reference_json_path:
            print(f'Reference JSON: {reference_json_path}')

        if normalize_descriptors:
            dataset.fit_descriptor_normalizer(train_idx)
        else:
            dataset.clear_descriptor_normalizer()

        train_dataset = Subset(dataset, train_idx.tolist())
        test_dataset = Subset(dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            collate_fn=collate_fn,
            drop_last=_drop_last_singleton(len(train_dataset), batch_size),
            generator=_make_generator(seed if shuffle_train else None),
            **_loader_kwargs(
                num_workers=num_workers,
                seed=seed if shuffle_train else None,
            ),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            generator=_make_generator(None if seed is None else seed + 1),
            **_loader_kwargs(num_workers=num_workers, seed=None if seed is None else seed + 1),
        )
        return train_loader, test_loader

    split_generator = _make_generator(seed)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=split_generator,
    )

    if normalize_descriptors:
        dataset.fit_descriptor_normalizer(train_dataset.indices)
    else:
        dataset.clear_descriptor_normalizer()

    train_smiles = {data['protac'].smiles for data in train_dataset}
    test_dataset = [data for data in test_dataset if data['protac'].smiles not in train_smiles]

    print('Dropped overlapping:')
    print('Train size: ', len(train_dataset))
    print('Test size: ', len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=_drop_last_singleton(len(train_dataset), batch_size),
        generator=_make_generator(None if seed is None else seed + 1),
        **_loader_kwargs(num_workers=num_workers, seed=None if seed is None else seed + 1),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        generator=_make_generator(None if seed is None else seed + 2),
        **_loader_kwargs(num_workers=num_workers, seed=None if seed is None else seed + 2),
    )

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = PROTACLoader()

    for item in train_loader:
        print(item)
        break

    for item in test_loader:
        print(item)
        break
