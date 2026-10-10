# feature_extraction.py
import numpy as np
import torch
from rdkit import Chem
from scipy import sparse as sp
from torch_geometric.data import Data


FRAGMENT_IDS = {
    'warhead': 0,
    'linker': 1,
    'e3_ligand': 2,
    'unknown': 3,
}

FRAGMENT_ID_NAMES = {value: key for key, value in FRAGMENT_IDS.items()}


def _is_missing_smiles(value):
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True

    text = str(value).strip()
    if text == '':
        return True

    return text.lower() in {'nan', 'none', 'null', 'na', 'n/a', 'nd', 'n.d.', 'false'}


def _clean_smiles(value):
    return None if _is_missing_smiles(value) else str(value).strip()


def _make_fragment_report(provided=False):
    return {
        'provided': bool(provided),
        'parsed': False,
        'matched': False,
        'num_matches': 0,
        'chosen_match_atoms': 0,
        'assigned_atoms': 0,
        'overlap_atoms': 0,
        'inferred': False,
    }


def _substructure_matches(mol, fragment_smiles):
    smiles = _clean_smiles(fragment_smiles)
    report = _make_fragment_report(provided=smiles is not None)
    if smiles is None:
        return report, []

    fragment_mol = Chem.MolFromSmiles(smiles)
    if fragment_mol is None:
        fragment_mol = Chem.MolFromSmarts(smiles)
    if fragment_mol is None or fragment_mol.GetNumAtoms() == 0:
        return report, []

    report['parsed'] = True
    matches = mol.GetSubstructMatches(fragment_mol, uniquify=True, useChirality=True)
    if not matches:
        matches = mol.GetSubstructMatches(fragment_mol, uniquify=True, useChirality=False)

    report['num_matches'] = len(matches)
    report['matched'] = len(matches) > 0
    return report, matches


def _choose_match(matches, fragment_ids):
    unknown_id = FRAGMENT_IDS['unknown']

    def score(match):
        new_atoms = sum(1 for atom_idx in match if fragment_ids[atom_idx] == unknown_id)
        return new_atoms, len(match)

    return max(matches, key=score)


def _assign_fragment_atoms(fragment_ids, match, fragment_id):
    unknown_id = FRAGMENT_IDS['unknown']
    assigned_atoms = 0
    overlap_atoms = 0

    for atom_idx in match:
        current_id = fragment_ids[atom_idx]
        if current_id == unknown_id:
            fragment_ids[atom_idx] = fragment_id
            assigned_atoms += 1
        elif current_id != fragment_id:
            overlap_atoms += 1

    return assigned_atoms, overlap_atoms


def assign_fragment_ids(
    mol,
    warhead_smiles=None,
    linker_smiles=None,
    e3_ligand_smiles=None,
    infer_missing_linker=True,
    return_report=False,
):
    num_atoms = mol.GetNumAtoms()
    unknown_id = FRAGMENT_IDS['unknown']
    fragment_ids = [unknown_id] * num_atoms

    report = {
        'num_atoms': num_atoms,
        'unknown_atoms': num_atoms,
        'all_fragments_missing': False,
        'fragments': {},
    }

    fragment_inputs = {
        'warhead': warhead_smiles,
        'linker': linker_smiles,
        'e3_ligand': e3_ligand_smiles,
    }
    report['all_fragments_missing'] = all(
        _clean_smiles(value) is None for value in fragment_inputs.values()
    )

    # Match terminal ligands before the linker so attachment atoms do not let
    # linker patterns overwrite warhead/E3 assignments.
    match_order = ('warhead', 'e3_ligand', 'linker')
    for fragment_name in match_order:
        fragment_id = FRAGMENT_IDS[fragment_name]
        fragment_report, matches = _substructure_matches(
            mol,
            fragment_inputs[fragment_name],
        )

        if matches:
            chosen_match = _choose_match(matches, fragment_ids)
            assigned_atoms, overlap_atoms = _assign_fragment_atoms(
                fragment_ids,
                chosen_match,
                fragment_id,
            )
            fragment_report['chosen_match_atoms'] = len(chosen_match)
            fragment_report['assigned_atoms'] = assigned_atoms
            fragment_report['overlap_atoms'] = overlap_atoms

        report['fragments'][fragment_name] = fragment_report

    linker_report = report['fragments']['linker']
    endpoints_matched = (
        report['fragments']['warhead']['matched']
        and report['fragments']['e3_ligand']['matched']
    )
    if infer_missing_linker and not linker_report['matched'] and endpoints_matched:
        inferred_atoms = 0
        linker_id = FRAGMENT_IDS['linker']
        for atom_idx, fragment_id in enumerate(fragment_ids):
            if fragment_id == unknown_id:
                fragment_ids[atom_idx] = linker_id
                inferred_atoms += 1

        if inferred_atoms > 0:
            linker_report['inferred'] = True
            linker_report['assigned_atoms'] += inferred_atoms

    report['unknown_atoms'] = sum(1 for fragment_id in fragment_ids if fragment_id == unknown_id)

    fragment_tensor = torch.tensor(fragment_ids, dtype=torch.long)
    if return_report:
        return fragment_tensor, report
    return fragment_tensor


# 保持原有的 One-hot 编码辅助函数
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


# --- 1. 原子特征 (与 compound_process.py 保持一致) ---
def atom_features(atom, explicit_H=False, use_chirality=True):
    """
    Feature size:
    Symbol(17) + Degree(7) + Charge(1) + Radical(1) + Hybrid(6) + Aromatic(1) + Hs(5) + Chirality(3)
    Total = 41 dimensions
    """
    symbol = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    degree = [0, 1, 2, 3, 4, 5, 6]
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2,
                         'other']

    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
              [atom.GetIsAromatic()]

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])


    return results


# --- 2. 键特征 (与 compound_process.py 保持一致) ---
def bond_features(bond, use_chirality=True):
    """
    Feature size:
    Type(4) + Conjugated(1) + InRing(1) + Stereo(4)
    Total = 10 dimensions
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


# --- 3. 拉普拉斯位置编码 (适配 PyG) ---
def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    基于 PyG 格式的拉普拉斯位置编码
    """
    # 构建邻接矩阵
    rows, cols = edge_index
    data = np.ones(len(rows))
    adj = sp.csr_matrix((data, (rows.numpy(), cols.numpy())), shape=(num_nodes, num_nodes))

    # 计算入度 (对于无向图，入度=出度=度)
    in_degree = np.array(adj.sum(axis=1)).flatten()

    # 归一化拉普拉斯矩阵: L = I - D^{-1/2} A D^{-1/2}
    # D_inv_sqrt
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(in_degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    L = sp.eye(num_nodes) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    # 特征分解
    # 注意：分子图通常很小，使用 dense linalg 可能比 sparse eigs 更稳定
    try:
        EigVal, EigVec = np.linalg.eig(L.toarray())
    except:
        # 极少数情况特征分解失败，返回全0
        return torch.zeros((num_nodes, k))

    idx = EigVal.argsort()  # 排序
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # 提取前 k 个非平凡特征向量 (通常跳过第0个，因为它对应特征值0)
    # 但 compound_process.py 里似乎取了 1:pos_enc_dim+1
    embedding = EigVec[:, 1:k + 1]

    # Padding 如果节点数不够 k 个
    if embedding.shape[1] < k:
        pad = np.zeros((embedding.shape[0], k - embedding.shape[1]))
        embedding = np.concatenate((embedding, pad), axis=1)

    return torch.from_numpy(embedding).float()


# --- 4. 主转换函数: SMILES -> PyG Data ---
def smiles_to_pyg_graph(
    smiles,
    explicit_H=False,
    use_chirality=True,
    pos_enc_dim=8,
    warhead_smiles=None,
    linker_smiles=None,
    e3_ligand_smiles=None,
):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None

    if mol is None:
        # 返回空的 Data 对象或全0特征，防止 DataLoader 崩溃
        # 这里建议返回一个只有 1 个节点的 Dummy Graph
        return Data(x=torch.zeros(1, 41), edge_index=torch.zeros(2, 0).long(), edge_attr=torch.zeros(0, 10),
                    lap_pos_enc=torch.zeros(1, pos_enc_dim),
                    fragment_id=torch.full((1,), FRAGMENT_IDS['unknown'], dtype=torch.long))

    # --- 节点特征 ---
    atom_feats_list = []
    for atom in mol.GetAtoms():
        atom_feats_list.append(atom_features(atom, explicit_H, use_chirality))

    # 处理手性 (Assign 到最后 3 位)
    # 之前在 atom_features 里预留了 3 位 [0,0,0]
    atom_feats = np.array(atom_feats_list)
    if use_chirality:
        # force=True 确保我们一定能拿到能拿到的手性
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([mol.GetNumAtoms(), 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1

        # 拼接: 38 + 3 = 41 维
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)
    else:
        # 如果不使用手性，为了维度统一，补3个0
        atom_feats = np.concatenate([atom_feats, np.zeros((mol.GetNumAtoms(), 3))], axis=1)

    x = torch.tensor(atom_feats, dtype=torch.float)

    # --- 边特征 ---
    src_list = []
    dst_list = []
    bond_feats_list = []

    if len(mol.GetBonds()) > 0:
        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bf = bond_features(bond, use_chirality)

            # 添加双向边
            src_list.extend([u, v])
            dst_list.extend([v, u])
            bond_feats_list.extend([bf, bf])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(bond_feats_list), dtype=torch.float)
    else:
        # 无边的情况 (比如单个离子)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)

    # --- 位置编码 ---
    num_nodes = x.size(0)
    lap_pos_enc = compute_laplacian_pe(edge_index, num_nodes, k=pos_enc_dim)
    fragment_id = assign_fragment_ids(
        mol,
        warhead_smiles=warhead_smiles,
        linker_smiles=linker_smiles,
        e3_ligand_smiles=e3_ligand_smiles,
    )

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        lap_pos_enc=lap_pos_enc,
        fragment_id=fragment_id,
        smiles=smiles,
    )
