import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import toml

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

import wandb
from data import DESCRIPTOR_COLUMNS
from data_loader import (
    PROTACLoader,
    build_fold_loaders,
    build_group_stratified_folds,
    collate_fn,
    get_protac_dataset,
)
from losses import TriVolumeLoss
from model import PROTAC_STAN


CHECKPOINT_IGNORED_TRAIN_KEYS = {
    'checkpoint_enabled',
    'resume_checkpoint',
    'checkpoint_dir',
}
TORCH_THREADS_CONFIGURED = False


def non_negative_float(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            'value must be finite and greater than or equal to 0'
        )
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Train PROTAC-STAN.')
    parser.add_argument(
        '--config',
        default='config.toml',
        help='Path to the TOML configuration file (default: config.toml).',
    )
    parser.add_argument(
        '--tri-weight',
        type=non_negative_float,
        help='Override train.tri_weight from the configuration file.',
    )
    parser.add_argument(
        '--model-dir',
        help='Directory for the trained model and run metrics.',
    )
    parser.add_argument(
        '--checkpoint-dir',
        help='Override train.checkpoint_dir. Use a distinct directory per experiment.',
    )
    parser.add_argument(
        '--metrics-file',
        help='Metrics JSON path (default: <model-dir>/metrics.json).',
    )
    return parser.parse_args(argv)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def atomic_json_dump(payload, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
            allow_nan=False,
        )
        f.write('\n')
    os.replace(tmp_path, path)
    return path


def atomic_toml_dump(payload, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        toml.dump(payload, f)
    os.replace(tmp_path, path)
    return path


def setup_seed(seed, deterministic=True, deterministic_warn_only=False, num_threads=None):
    global TORCH_THREADS_CONFIGURED

    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if num_threads is not None and not TORCH_THREADS_CONFIGURED:
        torch.set_num_threads(int(num_threads))
        torch.set_num_interop_threads(int(num_threads))
        TORCH_THREADS_CONFIGURED = True

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.allow_tf32 = False

    if hasattr(torch, 'use_deterministic_algorithms'):
        torch.use_deterministic_algorithms(
            bool(deterministic),
            warn_only=bool(deterministic_warn_only),
        )

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')


def resolve_device(device_cfg='auto'):
    if device_cfg in (None, 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device_cfg).startswith('cuda') and not torch.cuda.is_available():
        print(f'Requested device={device_cfg}, but CUDA is not available. Falling back to CPU.')
        return torch.device('cpu')
    return torch.device(device_cfg)


def state_dict_to_cpu(state_dict):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state_dict.items()
    }


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def get_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])

    torch_rng_state = state['torch']
    if torch.is_tensor(torch_rng_state):
        torch_rng_state = torch_rng_state.detach().cpu().to(dtype=torch.uint8)
    else:
        torch_rng_state = torch.ByteTensor(torch_rng_state)
    torch.set_rng_state(torch_rng_state)

    if torch.cuda.is_available() and 'cuda' in state:
        cuda_rng_states = [
            cuda_state.detach().cpu().to(dtype=torch.uint8)
            if torch.is_tensor(cuda_state)
            else torch.ByteTensor(cuda_state)
            for cuda_state in state['cuda']
        ]
        torch.cuda.set_rng_state_all(cuda_rng_states)


def get_loader_generator_state(loader):
    if loader is None:
        return None

    candidates = [
        getattr(loader, 'generator', None),
        getattr(getattr(loader, 'sampler', None), 'generator', None),
        getattr(getattr(getattr(loader, 'batch_sampler', None), 'sampler', None), 'generator', None),
    ]
    for generator in candidates:
        if generator is not None:
            return generator.get_state()
    return None


def set_loader_generator_state(loader, state):
    if loader is None or state is None:
        return

    candidates = [
        getattr(loader, 'generator', None),
        getattr(getattr(loader, 'sampler', None), 'generator', None),
        getattr(getattr(getattr(loader, 'batch_sampler', None), 'sampler', None), 'generator', None),
    ]
    seen = set()
    for generator in candidates:
        if generator is None or id(generator) in seen:
            continue
        generator.set_state(state)
        seen.add(id(generator))


def make_config_signature(cfg):
    train_cfg = {
        key: value
        for key, value in cfg['train'].items()
        if key not in CHECKPOINT_IGNORED_TRAIN_KEYS
    }
    payload = {
        'model': cfg['model'],
        'train': train_cfg,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def make_checkpoint_config(train_cfg, name='single'):
    checkpoint_dir = os.path.abspath(
        train_cfg.get('checkpoint_dir', 'saved_models/resume_checkpoint')
    )
    prefix = '' if name == 'single' else f'{name}_'
    return {
        'enabled': bool(train_cfg.get('checkpoint_enabled', False)),
        'resume': bool(train_cfg.get('resume_checkpoint', False)),
        'last_path': os.path.join(checkpoint_dir, f'{prefix}last.pt'),
        'best_path': os.path.join(checkpoint_dir, f'{prefix}best.pt'),
    }


def atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f'{path}.tmp'
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_model_wts,
    best_metrics,
    counter,
    config_signature,
    train_loader,
    finished=False,
    extra_state=None,
):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict_to_cpu(model.state_dict()),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_model_wts': state_dict_to_cpu(best_model_wts),
        'best_metrics': best_metrics,
        'early_stop_counter': counter,
        'rng_state': get_rng_state(),
        'train_loader_generator_state': get_loader_generator_state(train_loader),
        'config_signature': config_signature,
        'finished': finished,
        'extra_state': extra_state or {},
    }
    atomic_torch_save(checkpoint, path)


def save_best_checkpoint(path, epoch, best_model_wts, best_metrics, config_signature):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': state_dict_to_cpu(best_model_wts),
        'best_metrics': best_metrics,
        'config_signature': config_signature,
    }
    atomic_torch_save(checkpoint, path)


def move_batch_to_device(data, device):
    return {
        'protac': data['protac'].to(device),
        'label': data['label'].to(device),
        'index': data['index'].to(device),
        'poi_saprot': data['poi_saprot'].to(device),
        'e3_saprot': data['e3_saprot'].to(device),
        'protac_smile': data['protac_smile'].to(device),
        'descriptor': data['descriptor'].to(device),
    }


def model_forward(model, batch, return_aux=False):
    return model(
        batch['protac'],
        batch['poi_saprot'],
        batch['e3_saprot'],
        batch['protac_smile'],
        batch['descriptor'],
        return_aux=return_aux,
    )


def is_relation_ib_active(relation_ib_cfg):
    if not relation_ib_cfg:
        return False
    return (
        bool(relation_ib_cfg.get('enabled', False))
        and float(relation_ib_cfg.get('beta', 0.0)) > 0.0
    )


def compute_relation_ib_loss(aux, relation_ib_cfg, device):
    if not is_relation_ib_active(relation_ib_cfg):
        return torch.zeros((), device=device)
    if aux is None or aux.get('relation_ib') is None:
        return torch.zeros((), device=device)

    relation_ib_aux = aux['relation_ib']
    loss_type = str(relation_ib_cfg.get('loss_type', 'sparsity')).lower()
    if loss_type == 'lmi':
        return relation_ib_aux['lmi_loss']
    return relation_ib_aux['sparsity_loss']


def is_molecule_contrastive_active(molecule_cfg):
    if not molecule_cfg:
        return False
    return (
        bool(molecule_cfg.get('contrastive_enabled', False))
        and float(molecule_cfg.get('contrastive_weight', 0.0)) > 0.0
    )


def compute_molecule_contrastive_loss(aux, molecule_cfg, device):
    if not is_molecule_contrastive_active(molecule_cfg):
        return torch.zeros((), device=device)
    if aux is None or aux.get('protac_pretrained') is None:
        return torch.zeros((), device=device)

    loss = aux['protac_pretrained'].get('contrastive_loss')
    if loss is None:
        return torch.zeros((), device=device)
    return loss


def is_npp_active(npp_cfg):
    return bool((npp_cfg or {}).get('enabled', False))


def collect_loader_indices(loader):
    indices = []
    dataset = loader.dataset
    for i in range(len(dataset)):
        item = dataset[i]
        indices.append(int(item['index']))
    return torch.tensor(indices, dtype=torch.long)


def init_npp_state(train_loader, npp_cfg):
    if not is_npp_active(npp_cfg):
        return None

    train_indices = collect_loader_indices(train_loader)
    if train_indices.numel() == 0:
        return None

    sample_count = int(train_indices.max().item()) + 1
    return {
        'train_indices': train_indices,
        'weights': torch.ones(sample_count, dtype=torch.float32),
        'loss_ema': torch.zeros(sample_count, dtype=torch.float32),
        'seen': torch.zeros(sample_count, dtype=torch.bool),
        'stats': {
            'mean_weight': 1.0,
            'min_weight': 1.0,
            'max_weight': 1.0,
            'clean_fraction': 1.0,
        },
    }


def restore_npp_state(npp_state, extra_state):
    if npp_state is None:
        return
    saved = (extra_state or {}).get('npp')
    if not saved:
        return

    for key in ('weights', 'loss_ema', 'seen'):
        if key not in saved:
            continue
        saved_value = saved[key]
        if torch.is_tensor(saved_value) and saved_value.shape == npp_state[key].shape:
            npp_state[key] = saved_value.detach().cpu().clone()

    if isinstance(saved.get('stats'), dict):
        npp_state['stats'] = dict(saved['stats'])


def npp_state_for_checkpoint(npp_state):
    if npp_state is None:
        return None
    return {
        'weights': npp_state['weights'].detach().cpu().clone(),
        'loss_ema': npp_state['loss_ema'].detach().cpu().clone(),
        'seen': npp_state['seen'].detach().cpu().clone(),
        'stats': dict(npp_state.get('stats', {})),
    }


def get_npp_batch_weights(npp_state, indices, device):
    if npp_state is None:
        return None
    indices = indices.detach().cpu().long()
    return npp_state['weights'][indices].to(device)


def update_npp_loss_ema(npp_state, indices, losses, ema_decay):
    if npp_state is None:
        return

    indices = indices.detach().cpu().long()
    losses = losses.detach().cpu().float()
    old_losses = npp_state['loss_ema'][indices]
    old_seen = npp_state['seen'][indices]
    updated = torch.where(
        old_seen,
        float(ema_decay) * old_losses + (1.0 - float(ema_decay)) * losses,
        losses,
    )
    npp_state['loss_ema'][indices] = updated
    npp_state['seen'][indices] = True


def fit_gmm_clean_confidence_1d(values, max_iter=50, eps=1e-6):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 4 or float(values.std()) < eps:
        return np.ones(values.shape[0], dtype=np.float64)

    means = np.percentile(values, [30, 70]).astype(np.float64)
    variances = np.full(2, float(values.var()) + eps, dtype=np.float64)
    priors = np.full(2, 0.5, dtype=np.float64)

    for _ in range(int(max_iter)):
        log_probs = []
        for component in range(2):
            var = max(float(variances[component]), eps)
            log_prob = (
                np.log(max(float(priors[component]), eps))
                - 0.5 * np.log(2.0 * np.pi * var)
                - 0.5 * (values - means[component]) ** 2 / var
            )
            log_probs.append(log_prob)

        log_probs = np.stack(log_probs, axis=1)
        log_probs = log_probs - log_probs.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_probs)
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True).clip(min=eps)

        component_mass = responsibilities.sum(axis=0).clip(min=eps)
        priors = component_mass / values.size
        means = (responsibilities * values[:, None]).sum(axis=0) / component_mass
        variances = (
            responsibilities * (values[:, None] - means[None, :]) ** 2
        ).sum(axis=0) / component_mass
        variances = np.clip(variances, eps, None)

    clean_component = int(np.argmin(means))
    return responsibilities[:, clean_component].clip(0.0, 1.0)


def refresh_npp_weights(npp_state, npp_cfg):
    if npp_state is None:
        return None

    train_indices = npp_state['train_indices']
    seen_mask = npp_state['seen'][train_indices]
    valid_indices = train_indices[seen_mask]
    if valid_indices.numel() < 4:
        return npp_state.get('stats', {})

    losses = npp_state['loss_ema'][valid_indices].numpy()
    confidence = fit_gmm_clean_confidence_1d(
        losses,
        max_iter=npp_cfg.get('gmm_iters', 50),
        eps=npp_cfg.get('eps', 1e-6),
    )

    confidence_power = float(npp_cfg.get('confidence_power', 1.0))
    if confidence_power != 1.0:
        confidence = np.power(confidence.clip(0.0, 1.0), confidence_power)

    min_weight = float(npp_cfg.get('min_weight', 0.35))
    max_weight = float(npp_cfg.get('max_weight', 1.0))
    weights = min_weight + (max_weight - min_weight) * confidence

    npp_state['weights'][valid_indices] = torch.from_numpy(weights).float()
    stats = {
        'mean_weight': float(np.mean(weights)),
        'min_weight': float(np.min(weights)),
        'max_weight': float(np.max(weights)),
        'clean_fraction': float(np.mean(confidence >= 0.5)),
    }
    npp_state['stats'] = stats
    return stats


def safe_roc_auc_score(labels, probs):
    try:
        return roc_auc_score(labels, probs)
    except ValueError:
        return 0.5


def safe_pr_auc_score(labels, probs):
    try:
        return average_precision_score(labels, probs)
    except ValueError:
        return 0.0


def canonical_metric_name(name):
    aliases = {
        'acc': 'accuracy',
        'auc': 'roc_auc',
        'f1': 'f1_score',
        'pr': 'pr_auc',
        'ap': 'pr_auc',
    }
    return aliases.get(name, name)


def metric_value(metrics, name):
    return metrics.get(canonical_metric_name(name), -float('inf'))


def is_better_metrics(
    current,
    best,
    selection_metric='roc_auc',
    selection_tolerance=0.0,
    tie_breakers=None,
):
    if best is None:
        return True

    eps = 1e-12
    selection_tolerance = max(float(selection_tolerance), 0.0)
    current_primary = metric_value(current, selection_metric)
    best_primary = metric_value(best, selection_metric)

    if current_primary > best_primary + selection_tolerance + eps:
        return True
    if current_primary < best_primary - selection_tolerance - eps:
        return False

    for name in list(tie_breakers or []):
        current_value = metric_value(current, name)
        best_value = metric_value(best, name)
        if current_value > best_value + eps:
            return True
        if current_value < best_value - eps:
            return False
    return False


def test(model, test_loader, device, tri_criterion=None, tri_weight=0.0, relation_ib_cfg=None):
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    losses = []
    cls_losses = []
    tri_losses = []
    relation_ib_losses = []
    labels = []
    predictions = []
    probs = []

    with torch.no_grad():
        for data in test_loader:
            batch = move_batch_to_device(data, device)
            need_tri = tri_criterion is not None and tri_weight > 0
            need_relation_ib = is_relation_ib_active(relation_ib_cfg)
            need_aux = need_tri or need_relation_ib

            if need_aux:
                outputs, aux = model_forward(model, batch, return_aux=True)
            else:
                outputs = model_forward(model, batch, return_aux=False)
                aux = None

            if need_tri:
                tri_loss, _ = tri_criterion(
                    aux['tri_feats']['poi'],
                    aux['tri_feats']['protac'],
                    aux['tri_feats']['e3'],
                )
            else:
                tri_loss = torch.zeros((), device=device)

            relation_ib_loss = compute_relation_ib_loss(aux, relation_ib_cfg, device)
            relation_ib_weight = float((relation_ib_cfg or {}).get('beta', 0.0))

            cls_loss = criterion(outputs, batch['label'])
            total_loss = cls_loss + tri_weight * tri_loss + relation_ib_weight * relation_ib_loss

            losses.append(total_loss.item())
            cls_losses.append(cls_loss.item())
            tri_losses.append(tri_loss.item())
            relation_ib_losses.append(relation_ib_loss.item())

            pred = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)[:, 1]

            labels.extend(batch['label'].cpu().numpy())
            predictions.extend(pred.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    loss = sum(losses) / len(losses)
    cls_loss = sum(cls_losses) / len(cls_losses)
    tri_loss = sum(tri_losses) / len(tri_losses)
    relation_ib_loss = sum(relation_ib_losses) / len(relation_ib_losses)
    accuracy = accuracy_score(labels, predictions)
    roc_auc = safe_roc_auc_score(labels, probs)
    pr_auc = safe_pr_auc_score(labels, probs)
    f1 = f1_score(labels, predictions, average='macro')
    mcc = matthews_corrcoef(labels, predictions) if len(set(labels)) > 1 else 0.0

    return {
        'accuracy': accuracy,
        'loss': loss,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'f1_score': f1,
        'mcc': mcc,
        'cls_loss': cls_loss,
        'tri_loss': tri_loss,
        'relation_ib_loss': relation_ib_loss,
    }


def train(
    model,
    train_loader,
    test_loader,
    device,
    lr=0.001,
    num_epochs=10,
    accumulation_steps=1,
    tri_weight=0.0,
    tri_temp=0.07,
    tri_anchor='protac',
    relation_ib_cfg=None,
    molecule_cfg=None,
    npp_cfg=None,
    patience=30,
    selection_metric='roc_auc',
    selection_tolerance=0.0,
    selection_tie_breakers=None,
    weight_decay=1e-4,
    checkpoint_cfg=None,
    config_signature=None,
):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(reduction='none')
    molecule_cfg = molecule_cfg or {}
    npp_cfg = npp_cfg or {}
    tri_criterion = None
    if tri_weight > 0:
        tri_criterion = TriVolumeLoss(
            temp=tri_temp,
            anchor_mode=tri_anchor,
            label_smoothing=0.0,
        ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    npp_state = init_npp_state(train_loader, npp_cfg)

    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_metrics = None
    start_epoch = 0

    checkpoint_cfg = checkpoint_cfg or {}
    checkpoint_enabled = checkpoint_cfg.get('enabled', False)
    checkpoint_last_path = checkpoint_cfg.get('last_path')
    checkpoint_best_path = checkpoint_cfg.get('best_path')

    if checkpoint_enabled and checkpoint_cfg.get('resume', False) and checkpoint_last_path:
        if os.path.exists(checkpoint_last_path):
            checkpoint = torch_load(checkpoint_last_path, map_location='cpu')
            if checkpoint.get('config_signature') != config_signature:
                print(
                    'Checkpoint config does not match the current config; '
                    f'skipping resume: {checkpoint_last_path}'
                )
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to_device(optimizer, device)
                restore_npp_state(npp_state, checkpoint.get('extra_state', {}))
                best_model_wts = copy.deepcopy(checkpoint['best_model_wts'])
                best_metrics = checkpoint.get('best_metrics')
                counter = checkpoint.get('early_stop_counter', 0)
                start_epoch = checkpoint['epoch'] + 1
                if checkpoint.get('finished', False):
                    start_epoch = num_epochs
                set_rng_state(checkpoint.get('rng_state'))
                set_loader_generator_state(
                    train_loader,
                    checkpoint.get('train_loader_generator_state'),
                )
                print(
                    f'Resumed checkpoint from {checkpoint_last_path} '
                    f'at epoch {start_epoch}/{num_epochs}'
                )
        else:
            print(f'No checkpoint found at {checkpoint_last_path}; starting from scratch.')

    molecule_contrastive_weight = float(molecule_cfg.get('contrastive_weight', 0.0))
    npp_warmup_epochs = int(npp_cfg.get('warmup_epochs', 10))
    npp_update_interval = max(1, int(npp_cfg.get('update_interval', 1)))
    npp_ema_decay = float(npp_cfg.get('ema_decay', 0.8))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_total_loss = 0.0
        running_cls_loss = 0.0
        running_tri_loss = 0.0
        running_relation_ib_loss = 0.0
        running_molecule_contrastive_loss = 0.0
        running_npp_weight = 0.0
        optimizer.zero_grad()

        for i, data in enumerate(train_loader):
            batch = move_batch_to_device(data, device)
            need_tri = tri_criterion is not None
            need_relation_ib = is_relation_ib_active(relation_ib_cfg)
            need_molecule_contrastive = is_molecule_contrastive_active(molecule_cfg)
            need_aux = need_tri or need_relation_ib or need_molecule_contrastive

            if need_aux:
                outputs, aux = model_forward(model, batch, return_aux=True)
            else:
                outputs = model_forward(model, batch, return_aux=False)
                aux = None

            if need_tri:
                tri_loss, _ = tri_criterion(
                    aux['tri_feats']['poi'],
                    aux['tri_feats']['protac'],
                    aux['tri_feats']['e3'],
                )
            else:
                tri_loss = torch.zeros((), device=device)

            relation_ib_loss = compute_relation_ib_loss(aux, relation_ib_cfg, device)
            relation_ib_weight = float((relation_ib_cfg or {}).get('beta', 0.0))
            molecule_contrastive_loss = compute_molecule_contrastive_loss(aux, molecule_cfg, device)

            cls_losses = criterion(outputs, batch['label'])
            update_npp_loss_ema(
                npp_state,
                batch['index'],
                cls_losses,
                ema_decay=npp_ema_decay,
            )
            batch_weights = get_npp_batch_weights(npp_state, batch['index'], device)
            if batch_weights is None:
                cls_loss = cls_losses.mean()
                mean_batch_weight = 1.0
            else:
                cls_loss = (cls_losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-6)
                mean_batch_weight = float(batch_weights.detach().mean().item())

            total_loss = (
                cls_loss
                + tri_weight * tri_loss
                + relation_ib_weight * relation_ib_loss
                + molecule_contrastive_weight * molecule_contrastive_loss
            )

            loss = total_loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            running_total_loss += total_loss.item()
            running_cls_loss += cls_loss.item()
            running_tri_loss += tri_loss.item()
            running_relation_ib_loss += relation_ib_loss.item()
            running_molecule_contrastive_loss += molecule_contrastive_loss.item()
            running_npp_weight += mean_batch_weight

        train_total_loss = running_total_loss / len(train_loader)
        train_cls_loss = running_cls_loss / len(train_loader)
        train_tri_loss = running_tri_loss / len(train_loader)
        train_relation_ib_loss = running_relation_ib_loss / len(train_loader)
        train_molecule_contrastive_loss = running_molecule_contrastive_loss / len(train_loader)
        train_npp_weight = running_npp_weight / len(train_loader)

        if (
            npp_state is not None
            and epoch + 1 >= npp_warmup_epochs
            and (epoch + 1 - npp_warmup_epochs) % npp_update_interval == 0
        ):
            refresh_npp_weights(npp_state, npp_cfg)
        npp_stats = npp_state.get('stats', {}) if npp_state is not None else {}

        val_metrics = test(
            model,
            test_loader,
            device,
            tri_criterion=tri_criterion,
            tri_weight=tri_weight,
            relation_ib_cfg=relation_ib_cfg,
        )
        test_acc = val_metrics['accuracy']
        test_loss = val_metrics['loss']
        roc_auc = val_metrics['roc_auc']
        pr_auc = val_metrics['pr_auc']
        f1 = val_metrics['f1_score']
        mcc = val_metrics['mcc']
        val_relation_ib_loss = val_metrics.get('relation_ib_loss', 0.0)

        print(
            f'Epoch: {epoch + 1}/{num_epochs} | '
            f'train total: {train_total_loss:.4f} cls: {train_cls_loss:.4f} '
            f'tri: {train_tri_loss:.4f} rib: {train_relation_ib_loss:.4f} '
            f'mol: {train_molecule_contrastive_loss:.4f} '
            f'npp_w: {train_npp_weight:.3f} | '
            f'test acc: {test_acc:.4f} auc: {roc_auc:.4f} pr_auc: {pr_auc:.4f} '
            f'f1: {f1:.4f} mcc: {mcc:.4f} rib: {val_relation_ib_loss:.4f} '
            f'loss: {test_loss:.4f}'
        )

        wandb.log({
            'train/epoch': epoch + 1,
            'train/total_loss': train_total_loss,
            'train/cls_loss': train_cls_loss,
            'train/tri_loss': train_tri_loss,
            'train/relation_ib_loss': train_relation_ib_loss,
            'train/molecule_contrastive_loss': train_molecule_contrastive_loss,
            'train/molecule_contrastive_weight': molecule_contrastive_weight,
            'train/npp_batch_weight': train_npp_weight,
            'train/npp_mean_weight': npp_stats.get('mean_weight', 1.0),
            'train/npp_min_weight': npp_stats.get('min_weight', 1.0),
            'train/npp_max_weight': npp_stats.get('max_weight', 1.0),
            'train/npp_clean_fraction': npp_stats.get('clean_fraction', 1.0),
            'test/accuracy': test_acc,
            'test/loss': test_loss,
            'test/cls_loss': val_metrics['cls_loss'],
            'test/tri_loss': val_metrics['tri_loss'],
            'test/relation_ib_loss': val_relation_ib_loss,
            'test/roc_auc': roc_auc,
            'test/pr_auc': pr_auc,
            'test/f1_score': f1,
            'test/mcc': mcc,
        })

        is_better = is_better_metrics(
            val_metrics,
            best_metrics,
            selection_metric=selection_metric,
            selection_tolerance=selection_tolerance,
            tie_breakers=selection_tie_breakers,
        )
        if is_better:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
            selected = metric_value(best_metrics, selection_metric)
            print(
                f'Best model updated with {selection_metric}={selected:.4f}, '
                f'auc={roc_auc:.4f}, pr_auc={pr_auc:.4f}, acc={test_acc:.4f}'
            )
            wandb.run.summary['best_results'] = best_metrics
            counter = 0
            if checkpoint_enabled and checkpoint_best_path:
                save_best_checkpoint(
                    checkpoint_best_path,
                    epoch=epoch,
                    best_model_wts=best_model_wts,
                    best_metrics=best_metrics,
                    config_signature=config_signature,
                )
        else:
            counter += 1

        finished = counter >= patience or (epoch + 1) >= num_epochs
        if checkpoint_enabled and checkpoint_last_path:
            save_checkpoint(
                checkpoint_last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_model_wts=best_model_wts,
                best_metrics=best_metrics,
                counter=counter,
                config_signature=config_signature,
                train_loader=train_loader,
                finished=finished,
                extra_state={'npp': npp_state_for_checkpoint(npp_state)},
            )
            #print(f'Checkpoint saved: {checkpoint_last_path}')

        if finished and counter >= patience:
            print('Early stopped!')
            break

    model.load_state_dict(best_model_wts)
    return model, best_metrics


def summarize_cv_metrics(fold_metrics):
    summary = {}
    metric_names = sorted({key for metrics in fold_metrics for key in metrics.keys() if key != 'fold'})

    for name in metric_names:
        values = [metrics[name] for metrics in fold_metrics if name in metrics]
        if not values:
            continue
        values = np.array(values, dtype=float)
        summary[f'{name}_mean'] = float(values.mean())
        summary[f'{name}_std'] = float(values.std(ddof=0))

    return summary


def save_cv_results_csv(fold_metrics, cv_summary, out_path):
    metric_names = sorted({key for metrics in fold_metrics for key in metrics.keys() if key != 'fold'})
    fieldnames = ['fold'] + metric_names

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metrics in fold_metrics:
            row = {name: metrics.get(name, '') for name in fieldnames}
            writer.writerow(row)

        mean_row = {'fold': 'mean'}
        std_row = {'fold': 'std'}
        for name in metric_names:
            mean_row[name] = cv_summary.get(f'{name}_mean', '')
            std_row[name] = cv_summary.get(f'{name}_std', '')
        writer.writerow(mean_row)
        writer.writerow(std_row)

    return out_path


def log_descriptor_gates(model):
    if getattr(model, 'descriptor_encoder', None) is None:
        return None

    descriptor_gates = torch.sigmoid(model.descriptor_encoder.feature_gate).detach().cpu().tolist()
    descriptor_gate_map = {
        name: round(value, 4) for name, value in zip(DESCRIPTOR_COLUMNS, descriptor_gates)
    }
    print('Descriptor gates:', descriptor_gate_map)
    wandb.run.summary['descriptor_gates'] = descriptor_gate_map
    return descriptor_gate_map


def get_batch_smiles(data, batch_size, start_idx):
    smiles = getattr(data['protac'], 'smiles', None)
    if smiles is None:
        return [f'idx_{start_idx + i}' for i in range(batch_size)]
    if isinstance(smiles, str):
        return [smiles]
    if isinstance(smiles, (list, tuple)):
        return [str(item) for item in smiles]
    try:
        values = list(smiles)
    except TypeError:
        return [str(smiles)]
    return [str(item) for item in values]


def print_relation_attention_scores(model, data_loader, device, split_name='validation', max_samples=10):
    model = model.to(device)
    model.eval()

    all_attention = []
    all_relation_ib_gates = []
    labels = []
    predictions = []
    probabilities = []
    smiles_values = []
    relation_token_names = None
    seen = 0

    with torch.no_grad():
        for data in data_loader:
            batch = move_batch_to_device(data, device)
            outputs, aux = model_forward(model, batch, return_aux=True)
            relation_attention = aux.get('relation_attention')
            relation_ib_aux = aux.get('relation_ib')

            if relation_attention is not None:
                all_attention.append(relation_attention.detach().cpu().numpy())
            if relation_ib_aux is not None and relation_ib_aux.get('gate') is not None:
                all_relation_ib_gates.append(relation_ib_aux['gate'].detach().cpu().numpy())

            prob = torch.softmax(outputs, dim=1)[:, 1]
            pred = torch.argmax(outputs, dim=1)
            batch_size = outputs.size(0)

            labels.extend(batch['label'].detach().cpu().numpy())
            predictions.extend(pred.detach().cpu().numpy())
            probabilities.extend(prob.detach().cpu().numpy())
            smiles_values.extend(get_batch_smiles(data, batch_size, seen))
            relation_token_names = list(aux.get('relation_token_names', []))
            seen += batch_size

    if relation_token_names is None:
        print('No relation token outputs were collected.')
        return None

    relation_token_names = np.asarray(relation_token_names, dtype=object)
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)
    smiles_values = np.asarray(smiles_values, dtype=object)
    summary = {}

    print(f'\n========== Relation Scores ({split_name}) ==========')

    attention = None
    if all_attention:
        attention = np.concatenate(all_attention, axis=0)
        mean_attention = attention.mean(axis=(0, 1))
        attention_summary = {
            name: round(float(value), 4)
            for name, value in zip(relation_token_names.tolist(), mean_attention.tolist())
        }
        print('Mean relation attention:', attention_summary)
        summary['attention'] = attention_summary
    else:
        print('Relation attention is disabled.')

    relation_ib_gates = None
    if all_relation_ib_gates:
        relation_ib_gates = np.concatenate(all_relation_ib_gates, axis=0)
        mean_relation_ib_gates = relation_ib_gates.mean(axis=(0, 2))
        gate_summary = {
            name: round(float(value), 4)
            for name, value in zip(relation_token_names.tolist(), mean_relation_ib_gates.tolist())
        }
        print('Mean relation IB gates:', gate_summary)
        summary['relation_ib_gates'] = gate_summary
    else:
        print('Relation IB gates are disabled.')

    if attention is None and relation_ib_gates is None:
        return summary

    sample_count = min(max_samples, len(labels))
    if sample_count > 0:
        print(f'First {sample_count} sample relation scores:')
        for idx in range(sample_count):
            attention_scores = None
            if attention is not None:
                attention_scores = {
                    name: round(float(value), 4)
                    for name, value in zip(relation_token_names.tolist(), attention[idx, 0].tolist())
                }
            gate_scores = None
            if relation_ib_gates is not None:
                gate_scores = {
                    name: round(float(value), 4)
                    for name, value in zip(relation_token_names.tolist(), relation_ib_gates[idx, :, 0].tolist())
                }
            print(
                f'  #{idx + 1} label={int(labels[idx])} pred={int(predictions[idx])} '
                f'prob={float(probabilities[idx]):.4f} attention={attention_scores} '
                f'ib_gates={gate_scores} smiles={smiles_values[idx]}'
            )

    return summary


def maybe_print_relation_attention(model, data_loader, device, train_cfg, split_name):
    if not train_cfg.get('print_relation_attention', True):
        return None

    return print_relation_attention_scores(
        model,
        data_loader,
        device,
        split_name=split_name,
        max_samples=train_cfg.get('relation_attention_print_samples', 10),
    )


def run_single_split(
    model_cfg,
    train_cfg,
    model_dir,
    device,
    base_seed,
    dataset_root,
    dataset_name,
    normalize_descriptors,
    accumulation_steps,
    tri_weight,
    tri_temp,
    tri_anchor,
    split_mode,
    split_seed,
    reference_json_path,
    shuffle_train,
    num_workers,
    checkpoint_cfg,
    config_signature,
):
    train_loader, test_loader = PROTACLoader(
        root=dataset_root,
        name=dataset_name,
        batch_size=train_cfg['batch_size'],
        collate_fn=collate_fn,
        train_ratio=train_cfg['train_ratio'],
        seed=base_seed,
        normalize_descriptors=normalize_descriptors,
        split_mode=split_mode,
        split_seed=split_seed,
        reference_json_path=reference_json_path,
        shuffle_train=shuffle_train,
        num_workers=num_workers,
    )

    model = PROTAC_STAN(model_cfg)
    print(model)

    model, best_metrics = train(
        model,
        train_loader,
        test_loader,
        device,
        lr=train_cfg['learning_rate'],
        num_epochs=train_cfg['num_epochs'],
        accumulation_steps=accumulation_steps,
        tri_weight=tri_weight,
        tri_temp=tri_temp,
        tri_anchor=tri_anchor,
        relation_ib_cfg=model_cfg.get('relation_ib', {}),
        molecule_cfg=model_cfg.get('protac', {}).get('pretrained_molecule', {}),
        npp_cfg=train_cfg.get('npp', {}),
        patience=train_cfg.get('patience', 30),
        selection_metric=train_cfg.get('selection_metric', 'roc_auc'),
        selection_tolerance=train_cfg.get('selection_tolerance', 0.0),
        selection_tie_breakers=train_cfg.get('selection_tie_breakers', []),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
        checkpoint_cfg=checkpoint_cfg,
        config_signature=config_signature,
    )

    log_descriptor_gates(model)
    if best_metrics is not None:
        print('Best metrics:', best_metrics)

    relation_summary = maybe_print_relation_attention(
        model,
        test_loader,
        device,
        train_cfg,
        split_name='test',
    )
    if relation_summary is not None:
        wandb.run.summary['relation_scores_test'] = relation_summary

    torch.save(model, f'{model_dir}/model.pt')
    torch.save(model.state_dict(), f'{model_dir}/model_state_dict.pt')
    return model, best_metrics


def run_cross_validation(
    model_cfg,
    train_cfg,
    model_dir,
    device,
    base_seed,
    dataset_root,
    dataset_name,
    normalize_descriptors,
    accumulation_steps,
    tri_weight,
    tri_temp,
    tri_anchor,
    config_signature,
):
    dataset = get_protac_dataset(root=dataset_root, name=dataset_name)
    n_splits = train_cfg.get('cv_folds', 5)
    folds = build_group_stratified_folds(dataset, n_splits=n_splits, seed=base_seed)

    print(f'Running {n_splits}-fold StratifiedGroupKFold CV')
    print('Total size:', len(dataset))

    fold_metrics = []
    best_fold_idx = None
    best_fold_metrics = None
    best_model = None
    selection_metric = train_cfg.get('selection_metric', 'roc_auc')
    selection_tolerance = train_cfg.get('selection_tolerance', 0.0)
    selection_tie_breakers = train_cfg.get('selection_tie_breakers', [])
    cv_csv_path = os.path.abspath(os.path.join(model_dir, 'cv_results.csv'))

    for fold_id, train_idx, val_idx in folds:
        fold_no = fold_id + 1
        print(f'\n========== Fold {fold_no}/{n_splits} ==========')
        print(f'Train size: {len(train_idx)} | Val size: {len(val_idx)}')
        setup_seed(
            base_seed + fold_id,
            deterministic=train_cfg.get('deterministic', True),
            deterministic_warn_only=train_cfg.get('deterministic_warn_only', False),
            num_threads=None,
        )

        train_loader, val_loader = build_fold_loaders(
            dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            batch_size=train_cfg['batch_size'],
            collate_fn=collate_fn,
            seed=base_seed + fold_id,
            normalize_descriptors=normalize_descriptors,
            num_workers=train_cfg.get('num_workers', 0),
        )

        model = PROTAC_STAN(model_cfg)
        model, best_metrics = train(
            model,
            train_loader,
            val_loader,
            device,
            lr=train_cfg['learning_rate'],
            num_epochs=train_cfg['num_epochs'],
            accumulation_steps=accumulation_steps,
            tri_weight=tri_weight,
            tri_temp=tri_temp,
            tri_anchor=tri_anchor,
            relation_ib_cfg=model_cfg.get('relation_ib', {}),
            molecule_cfg=model_cfg.get('protac', {}).get('pretrained_molecule', {}),
            npp_cfg=train_cfg.get('npp', {}),
            patience=train_cfg.get('patience', 30),
            selection_metric=selection_metric,
            selection_tolerance=selection_tolerance,
            selection_tie_breakers=selection_tie_breakers,
            weight_decay=train_cfg.get('weight_decay', 1e-4),
            checkpoint_cfg=make_checkpoint_config(train_cfg, name=f'fold_{fold_no}'),
            config_signature=config_signature,
        )

        if best_metrics is None:
            best_metrics = {}
        best_metrics = {'fold': fold_no, **best_metrics}
        fold_metrics.append(best_metrics)
        print(f'Fold {fold_no} best metrics:', best_metrics)
        save_cv_results_csv(fold_metrics, summarize_cv_metrics(fold_metrics), cv_csv_path)
        print(f'CV results updated: {cv_csv_path}')

        fold_dir = os.path.join(model_dir, f'fold_{fold_no}')
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, 'model_state_dict.pt'))

        relation_summary = maybe_print_relation_attention(
            model,
            val_loader,
            device,
            train_cfg,
            split_name=f'fold {fold_no} validation',
        )
        if relation_summary is not None:
            wandb.run.summary[f'fold_{fold_no}_relation_scores'] = relation_summary

        wandb.log({
            'cv/fold': fold_no,
            'cv/fold_accuracy': best_metrics.get('accuracy', 0.0),
            'cv/fold_roc_auc': best_metrics.get('roc_auc', 0.0),
            'cv/fold_pr_auc': best_metrics.get('pr_auc', 0.0),
            'cv/fold_f1': best_metrics.get('f1_score', 0.0),
            'cv/fold_mcc': best_metrics.get('mcc', 0.0),
            'cv/fold_loss': best_metrics.get('loss', 0.0),
            'cv/fold_relation_ib_loss': best_metrics.get('relation_ib_loss', 0.0),
        })

        if is_better_metrics(
            best_metrics,
            best_fold_metrics,
            selection_metric=selection_metric,
            selection_tolerance=selection_tolerance,
            tie_breakers=selection_tie_breakers,
        ):
            best_fold_metrics = dict(best_metrics)
            best_fold_idx = fold_no
            best_model = copy.deepcopy(model).cpu()

    cv_summary = summarize_cv_metrics(fold_metrics)
    print('\n========== CV Summary ==========')
    for key, value in cv_summary.items():
        print(f'{key}: {value:.4f}')
    print('Fold metrics:', fold_metrics)

    wandb.run.summary['cv_fold_metrics'] = fold_metrics
    wandb.run.summary['cv_summary'] = cv_summary
    wandb.run.summary['best_fold'] = best_fold_idx

    cv_csv_path = save_cv_results_csv(fold_metrics, cv_summary, cv_csv_path)
    print(f'CV results saved to: {cv_csv_path}')
    wandb.run.summary['cv_results_csv'] = cv_csv_path

    if best_model is not None:
        torch.save(best_model, f'{model_dir}/model.pt')
        torch.save(best_model.state_dict(), f'{model_dir}/model_state_dict.pt')
        log_descriptor_gates(best_model)

    return best_model, cv_summary


def main(argv=None):
    args = parse_args(argv)
    started_at = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    started_time = time.time()

    cfg_path = os.path.abspath(args.config)
    cfg = toml.load(cfg_path)
    model_cfg = cfg['model']
    train_cfg = cfg['train']

    if args.tri_weight is not None:
        train_cfg['tri_weight'] = args.tri_weight
    if args.checkpoint_dir:
        train_cfg['checkpoint_dir'] = os.path.abspath(args.checkpoint_dir)

    if args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = os.path.abspath(
            f'saved_models/{time.strftime("%Y%m%d")}/{time.strftime("%H%M%S")}'
        )
    os.makedirs(model_dir, exist_ok=True)
    effective_config_path = atomic_toml_dump(
        cfg,
        os.path.join(model_dir, 'effective_config.toml'),
    )

    base_seed = model_cfg['seed']
    setup_seed(
        base_seed,
        deterministic=train_cfg.get('deterministic', True),
        deterministic_warn_only=train_cfg.get('deterministic_warn_only', False),
        num_threads=train_cfg.get('num_threads', None),
    )

    accumulation_steps = train_cfg.get('accumulation_steps', 4)
    tri_weight = train_cfg.get('tri_weight', 0.1)
    tri_temp = train_cfg.get('tri_temp', 0.07)
    tri_anchor = train_cfg.get('tri_anchor', 'protac')
    normalize_descriptors = train_cfg.get('normalize_descriptors', True)
    dataset_root = train_cfg.get('dataset_root', 'data/PROTAC-fine')
    dataset_name = train_cfg.get('dataset_name', 'protac-fine')
    split_mode = train_cfg.get('split_mode', 'random')
    split_seed = train_cfg.get('split_seed', base_seed)
    reference_json_path = train_cfg.get('reference_json_path', None)
    if reference_json_path and not os.path.exists(reference_json_path):
        print(f'reference_json_path not found, falling back to dataset order split: {reference_json_path}')
        reference_json_path = None
    shuffle_train = train_cfg.get('shuffle_train', True)
    num_workers = train_cfg.get('num_workers', 0)
    config_signature = make_config_signature(cfg)
    checkpoint_cfg = make_checkpoint_config(train_cfg, name='single')

    wandb_initialized = False
    try:
        wandb.init(
            mode='disabled',
            project='protac-stan',
            config=cfg,
            group=f'run_bz{train_cfg["batch_size"]}_lr{train_cfg["learning_rate"]}',
        )
        wandb_initialized = True

        wandb.run.summary['model_dir'] = model_dir
        wandb.run.summary['descriptor_columns'] = DESCRIPTOR_COLUMNS
        wandb.run.summary['protein_feature_mode'] = 'saprot_only'
        wandb.run.summary['protac_feature_mode'] = model_cfg.get('protac', {}).get('feature_mode', 'graph_seq')
        wandb.run.summary['dataset_root'] = dataset_root
        wandb.run.summary['dataset_name'] = dataset_name
        wandb.run.summary['split_mode'] = split_mode
        wandb.run.summary['split_seed'] = split_seed
        wandb.run.summary['tri_weight'] = tri_weight

        print(cfg)
        device = resolve_device(train_cfg.get('device', 'auto'))
        print(device)

        use_cv = bool(train_cfg.get('use_cv', False))
        if use_cv:
            _, metrics = run_cross_validation(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                model_dir=model_dir,
                device=device,
                base_seed=base_seed,
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                normalize_descriptors=normalize_descriptors,
                accumulation_steps=accumulation_steps,
                tri_weight=tri_weight,
                tri_temp=tri_temp,
                tri_anchor=tri_anchor,
                config_signature=config_signature,
            )
        else:
            _, metrics = run_single_split(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                model_dir=model_dir,
                device=device,
                base_seed=base_seed,
                dataset_root=dataset_root,
                dataset_name=dataset_name,
                normalize_descriptors=normalize_descriptors,
                accumulation_steps=accumulation_steps,
                tri_weight=tri_weight,
                tri_temp=tri_temp,
                tri_anchor=tri_anchor,
                split_mode=split_mode,
                split_seed=split_seed,
                reference_json_path=reference_json_path,
                shuffle_train=shuffle_train,
                num_workers=num_workers,
                checkpoint_cfg=checkpoint_cfg,
                config_signature=config_signature,
            )
    finally:
        if wandb_initialized:
            try:
                wandb.finish()
            except Exception as exc:
                print(f'Warning: wandb.finish() failed: {exc}')

    metrics = metrics or {}
    selection_metric = train_cfg.get('selection_metric', 'roc_auc')
    selection_metric_key = canonical_metric_name(selection_metric)
    selection_key = f'{selection_metric_key}_mean' if use_cv else selection_metric_key
    selection_value = metrics.get(selection_key)
    try:
        selection_value_is_finite = math.isfinite(float(selection_value))
    except (TypeError, ValueError):
        selection_value_is_finite = False
    if not selection_value_is_finite:
        raise RuntimeError(
            'Training finished without a finite value for the requested '
            f'selection metric: {selection_key}'
        )
    run_result = {
        'status': 'completed',
        'tri_weight': tri_weight,
        'mode': 'cross_validation' if use_cv else 'single_split',
        'selection_metric': selection_metric,
        'selection_key': selection_key,
        'selection_value': selection_value,
        'metrics': metrics,
        'model_dir': model_dir,
        'checkpoint_dir': os.path.abspath(train_cfg.get('checkpoint_dir', 'saved_models/resume_checkpoint')),
        'config_path': cfg_path,
        'effective_config_path': effective_config_path,
        'config_signature': config_signature,
        'seed': base_seed,
        'split_seed': split_seed,
        'started_at': started_at,
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'duration_seconds': time.time() - started_time,
    }
    metrics_path = args.metrics_file or os.path.join(model_dir, 'metrics.json')
    metrics_path = atomic_json_dump(run_result, metrics_path)
    print(f'Run metrics saved to: {metrics_path}')
    return run_result


if __name__ == '__main__':
    main()
