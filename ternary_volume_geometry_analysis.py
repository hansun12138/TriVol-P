#!/usr/bin/env python3
"""Journal-ready ternary-volume geometry analysis for the PROTAC–POI–E3 model.

The script compares TriVol-PROTAC (tri_weight > 0) with a no-tri checkpoint
on exactly the same held-out split. It reuses the project's TriVolumeLoss volume
implementation and produces:

1. matched (diagonal) versus mismatched (off-diagonal) Gramian volumes;
2. PROTAC -> paired POI–E3 and paired POI–E3 -> PROTAC retrieval metrics;
3. duplicate-aware retrieval metrics when duplicate metadata are available;
4. paired bootstrap confidence intervals and per-model confidence intervals;
5. paper-ready heatmaps, distribution plots, CSV files, and NumPy matrices.

Place this file in the project root (beside model.py, losses.py, data.py, and
data_loader.py), or pass --project-root explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None


@dataclass
class ExtractedFeatures:
    dataset_index: np.ndarray
    label: np.ndarray
    probability: np.ndarray
    protac: np.ndarray
    poi: np.ndarray
    e3: np.ndarray
    checkpoint_source: str


@dataclass
class GeometryResult:
    label: str
    slug: str
    features: ExtractedFeatures
    volumes: np.ndarray
    similarity: np.ndarray
    per_sample: pd.DataFrame
    summary: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ternary Gramian-volume geometry for TriVol-P and "
            "a no-tri model on the same held-out test split."
        )
    )
    parser.add_argument("--project-root", default=".", help="Project root containing model.py and data_loader.py.")
    parser.add_argument("--config", default="config.toml", help="Training TOML configuration.")
    parser.add_argument("--full-checkpoint", help="Checkpoint for TriVol-P.")
    parser.add_argument("--no-tri-checkpoint", help="Checkpoint trained with tri_weight=0.")
    parser.add_argument(
        "--output-dir",
        default="analysis/ternary_volume_geometry",
        help="Directory for figures, matrices, and tables.",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N. Defaults to config.")
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size override.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker override.")
    parser.add_argument("--chunk-size", type=int, default=256, help="Anchor rows computed per volume chunk.")
    parser.add_argument("--heatmap-size", type=int, default=64, help="Maximum samples shown in each heatmap.")
    parser.add_argument(
        "--mismatch-plot-samples",
        type=int,
        default=100000,
        help=(
            "Maximum valid mismatched values used in panel C. The default "
            "includes every mismatch for the current 300-sample evaluation set."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Paired bootstrap replicates.")
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument(
        "--retrieval-evaluation",
        choices=("duplicate-aware", "diagonal"),
        default="duplicate-aware",
        help=(
            "Retrieval scores displayed in panel D. 'duplicate-aware' accepts any "
            "identical PROTAC–POI–E3 row as a valid positive and falls back to "
            "diagonal retrieval when identifier metadata are unavailable."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="Analysis seed; defaults to model seed.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Volume stability floor, matching TriVolumeLoss.")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-path",
        action="append",
        default=[],
        help=(
            "Optional Times New Roman font file or directory. Repeat this option "
            "to register regular, bold, italic, and bold-italic font files."
        ),
    )
    parser.add_argument(
        "--allow-font-fallback",
        action="store_true",
        help=(
            "Allow a serif fallback when Times New Roman is unavailable. "
            "Do not use this option for the final manuscript figure."
        ),
    )
    parser.add_argument(
        "--allow-non-strict-load",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys (not recommended).",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a deterministic synthetic test of the volume/ranking calculation and exit.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def configure_publication_font(
    project_root: Path,
    font_inputs: Sequence[str],
    allow_fallback: bool,
) -> Dict[str, Any]:
    """Register and enforce Times New Roman for every element in the figure.

    Matplotlib otherwise silently falls back to DejaVu Sans on many Linux
    systems. The default behavior is deliberately strict so a non-compliant
    manuscript figure cannot be produced unnoticed.
    """
    registered_files: List[str] = []
    supported_suffixes = {".ttf", ".ttc", ".otf"}
    for raw_value in font_inputs:
        path = resolve_path(project_root, raw_value)
        assert path is not None
        if not path.exists():
            raise FileNotFoundError(f"Font path not found: {path}")
        candidates = (
            sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in supported_suffixes
            )
            if path.is_dir()
            else [path]
        )
        if not candidates:
            raise FileNotFoundError(f"No TTF/TTC/OTF font files found under: {path}")
        for candidate in candidates:
            font_manager.fontManager.addfont(str(candidate))
            registered_files.append(str(candidate))

    requested_family = "Times New Roman"
    fallback_used = False
    try:
        resolved_font = font_manager.findfont(
            font_manager.FontProperties(family=requested_family),
            fallback_to_default=False,
        )
        resolved_family = font_manager.FontProperties(fname=resolved_font).get_name()
        if resolved_family.casefold() != requested_family.casefold():
            raise RuntimeError(
                f"Requested {requested_family}, but Matplotlib resolved {resolved_family}."
            )
    except (ValueError, RuntimeError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                "Times New Roman is not available to Matplotlib. Install the font "
                "or pass one or more --font-path arguments, for example: "
                "--font-path /path/to/times.ttf --font-path /path/to/timesbd.ttf. "
                "The script stops instead of silently producing a figure in another font."
            ) from exc
        fallback_used = True
        for candidate_family in ("Nimbus Roman", "Liberation Serif", "DejaVu Serif"):
            try:
                resolved_font = font_manager.findfont(
                    font_manager.FontProperties(family=candidate_family),
                    fallback_to_default=False,
                )
                resolved_family = font_manager.FontProperties(fname=resolved_font).get_name()
                break
            except ValueError:
                continue
        else:
            raise RuntimeError("No usable serif fallback font is available.") from exc
        warnings.warn(
            f"Times New Roman is unavailable; using {resolved_family}. "
            "This fallback must not be used for the final manuscript figure."
        )

    matplotlib.rcParams.update(
        {
            "font.family": resolved_family,
            "font.serif": [resolved_family],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "savefig.facecolor": "white",
        }
    )
    return {
        "requested_family": requested_family,
        "resolved_family": resolved_family,
        "resolved_font_file": str(resolved_font),
        "registered_font_files": registered_files,
        "fallback_used": fallback_used,
    }


def enforce_figure_font(fig: Any, family: str) -> None:
    """Apply the selected family to titles, labels, ticks, legends, and colorbars."""
    for text_artist in fig.findobj(match=matplotlib.text.Text):
        text_artist.set_fontfamily(family)


def load_toml(path: Path) -> Dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)

    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("Python < 3.11 requires the 'toml' package.") from exc
    return toml.load(str(path))


def seed_everything(seed: int, num_threads: Optional[int] = None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))


def resolve_device(value: Optional[str]) -> torch.device:
    value = "auto" if value is None else str(value)
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(f"Requested {value}, but CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return torch.device(value)


def import_project_api(project_root: Path) -> Dict[str, Any]:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        data_loader = importlib.import_module("data_loader")
        model_module = importlib.import_module("model")
        losses_module = importlib.import_module("losses")
        data_module = importlib.import_module("data")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the project modules. Put this script beside "
            "model.py/data_loader.py/losses.py/data.py or set --project-root."
        ) from exc

    required = {
        "PROTACLoader": getattr(data_loader, "PROTACLoader"),
        "collate_fn": getattr(data_loader, "collate_fn"),
        "PROTAC_STAN": getattr(model_module, "PROTAC_STAN"),
        "TriVolumeLoss": getattr(losses_module, "TriVolumeLoss"),
        "standardize_protac_dataframe": getattr(data_module, "standardize_protac_dataframe"),
    }
    return required


def torch_load_compat(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def is_state_dict(candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and len(candidate) > 0
        and all(isinstance(key, str) for key in candidate.keys())
        and all(torch.is_tensor(value) for value in candidate.values())
    )


def strip_uniform_prefix(state_dict: Mapping[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def extract_state_dict(checkpoint: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    if isinstance(checkpoint, torch.nn.Module):
        return dict(checkpoint.state_dict()), "serialized nn.Module"

    if isinstance(checkpoint, Mapping):
        # A training 'last.pt' contains both the current state and best weights.
        # Prefer best_model_wts so the analysis uses the selected epoch.
        for key in ("best_model_wts", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if is_state_dict(candidate):
                return dict(candidate), key
        if is_state_dict(checkpoint):
            return dict(checkpoint), "raw state_dict"

    raise ValueError(
        "Unsupported checkpoint format. Expected a saved model, raw state_dict, "
        "or a dictionary containing best_model_wts/model_state_dict/state_dict."
    )


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_model(
    checkpoint_path: Path,
    model_cfg: Mapping[str, Any],
    device: torch.device,
    model_class: Any,
    allow_non_strict: bool,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    state_dict, source = extract_state_dict(checkpoint)
    for prefix in ("module.", "_orig_mod."):
        state_dict = strip_uniform_prefix(state_dict, prefix)

    model = model_class(model_cfg)
    incompatible = model.load_state_dict(state_dict, strict=not allow_non_strict)
    if allow_non_strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        warnings.warn(
            "Non-strict checkpoint load: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )

    model.to(device)
    model.eval()
    info = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "state_source": source,
    }
    print(f"Loaded {checkpoint_path.name} using checkpoint field: {source}")
    return model, info


def build_test_loader(
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    project_root: Path,
    api: Mapping[str, Any],
) -> Tuple[Any, Path, str]:
    train_cfg = cfg["train"]
    if train_cfg.get("use_cv", False):
        raise ValueError(
            "This script currently analyzes the fixed held-out split. Set use_cv=false "
            "or run it separately for each fold-specific checkpoint/loader."
        )

    dataset_root = resolve_path(project_root, train_cfg.get("dataset_root", "data/PROTAC-fine"))
    dataset_name = train_cfg.get("dataset_name", "protac-fine")
    reference_value = train_cfg.get("reference_json_path")
    reference_path = resolve_path(project_root, reference_value) if reference_value else None
    if reference_path is not None and not reference_path.exists():
        warnings.warn(
            f"reference_json_path does not exist ({reference_path}); reproducing main.py fallback to dataset order."
        )
        reference_path = None

    batch_size = args.batch_size or int(train_cfg.get("batch_size", 16))
    num_workers = (
        int(args.num_workers)
        if args.num_workers is not None
        else int(train_cfg.get("num_workers", 0))
    )
    seed = int(args.seed if args.seed is not None else cfg["model"].get("seed", 111))

    _, test_loader = api["PROTACLoader"](
        root=str(dataset_root),
        name=dataset_name,
        batch_size=batch_size,
        collate_fn=api["collate_fn"],
        train_ratio=float(train_cfg.get("train_ratio", 0.8)),
        seed=seed,
        normalize_descriptors=bool(train_cfg.get("normalize_descriptors", True)),
        split_mode=train_cfg.get("split_mode", "random"),
        split_seed=int(train_cfg.get("split_seed", seed)),
        reference_json_path=str(reference_path) if reference_path else None,
        shuffle_train=False,
        num_workers=num_workers,
    )
    return test_loader, dataset_root, dataset_name


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


@torch.inference_mode()
def collect_features(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    device: torch.device,
    checkpoint_source: str,
) -> ExtractedFeatures:
    indices: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    probabilities: List[torch.Tensor] = []
    protac_features: List[torch.Tensor] = []
    poi_features: List[torch.Tensor] = []
    e3_features: List[torch.Tensor] = []

    for raw_batch in loader:
        if raw_batch.get("label") is None:
            raise ValueError("The selected split has no labels; held-out labels are required.")
        indices.append(raw_batch["index"].detach().cpu().long())
        labels.append(raw_batch["label"].detach().cpu().long())

        batch = move_batch_to_device(raw_batch, device)
        logits, aux = model(
            batch["protac"],
            batch["poi_saprot"],
            batch["e3_saprot"],
            batch["protac_smile"],
            batch["descriptor"],
            return_aux=True,
        )

        if "tri_feats" not in aux:
            raise KeyError("model(return_aux=True) did not return aux['tri_feats'].")
        tri = aux["tri_feats"]
        for required in ("protac", "poi", "e3"):
            if required not in tri:
                raise KeyError(f"Missing aux['tri_feats']['{required}']")

        probabilities.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().float())
        protac_features.append(tri["protac"].detach().cpu().float())
        poi_features.append(tri["poi"].detach().cpu().float())
        e3_features.append(tri["e3"].detach().cpu().float())

    if not indices:
        raise ValueError("The test loader produced no samples.")

    index = torch.cat(indices).numpy()
    order = np.argsort(index, kind="stable")
    extracted = ExtractedFeatures(
        dataset_index=index[order],
        label=torch.cat(labels).numpy()[order],
        probability=torch.cat(probabilities).numpy()[order],
        protac=torch.cat(protac_features).numpy()[order],
        poi=torch.cat(poi_features).numpy()[order],
        e3=torch.cat(e3_features).numpy()[order],
        checkpoint_source=checkpoint_source,
    )

    for name in ("protac", "poi", "e3", "probability"):
        values = getattr(extracted, name)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Non-finite values detected in {name} outputs.")
    return extracted


@torch.inference_mode()
def compute_volume_matrix(
    features: ExtractedFeatures,
    device: torch.device,
    tri_loss_class: Any,
    chunk_size: int,
    eps: float,
) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    protac = F.normalize(torch.from_numpy(features.protac).to(device), dim=-1)
    poi = F.normalize(torch.from_numpy(features.poi).to(device), dim=-1)
    e3 = F.normalize(torch.from_numpy(features.e3).to(device), dim=-1)
    criterion = tri_loss_class(temp=1.0, anchor_mode="protac", eps=eps).to(device)

    rows: List[torch.Tensor] = []
    for start in range(0, protac.shape[0], chunk_size):
        stop = min(start + chunk_size, protac.shape[0])
        volume = criterion.compute_3modal_volume_batch(protac[start:stop], poi, e3)
        rows.append(volume.detach().cpu().float())
    matrix = torch.cat(rows, dim=0).numpy()

    expected = (features.dataset_index.size, features.dataset_index.size)
    if matrix.shape != expected:
        raise AssertionError(f"Expected volume matrix {expected}, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Non-finite entries detected in the volume matrix.")
    return matrix


def average_tie_ranks(matrix: np.ndarray) -> np.ndarray:
    """Rank each diagonal target within its row; smaller volume is better."""
    diagonal = np.diag(matrix)
    tolerance = np.maximum(1e-10, np.abs(diagonal) * 1e-7)
    less = (matrix < (diagonal[:, None] - tolerance[:, None])).sum(axis=1)
    equal = np.isclose(
        matrix,
        diagonal[:, None],
        rtol=1e-7,
        atol=1e-10,
    ).sum(axis=1)
    return 1.0 + less.astype(float) + 0.5 * np.maximum(equal - 1, 0)


def per_sample_geometry(volumes: np.ndarray, recall_k: Sequence[int]) -> pd.DataFrame:
    n = volumes.shape[0]
    if n < 2:
        raise ValueError("At least two held-out samples are required for mismatched comparisons.")

    positive = np.diag(volumes)
    negative_forward = (volumes.sum(axis=1) - positive) / (n - 1)
    negative_reverse = (volumes.sum(axis=0) - positive) / (n - 1)
    rank_forward = average_tie_ranks(volumes)
    rank_reverse = average_tie_ranks(volumes.T)

    data: Dict[str, np.ndarray] = {
        "matched_volume": positive,
        "mean_mismatched_volume_forward": negative_forward,
        "mean_mismatched_volume_reverse": negative_reverse,
        "volume_gap_forward": negative_forward - positive,
        "volume_gap_reverse": negative_reverse - positive,
        "volume_gap_bidirectional": 0.5 * (negative_forward + negative_reverse) - positive,
        "rank_forward": rank_forward,
        "rank_reverse": rank_reverse,
        "reciprocal_rank_forward": 1.0 / rank_forward,
        "reciprocal_rank_reverse": 1.0 / rank_reverse,
        "reciprocal_rank_bidirectional": 0.5 * (1.0 / rank_forward + 1.0 / rank_reverse),
    }
    for k in recall_k:
        data[f"recall_at_{k}_forward"] = (rank_forward <= k).astype(float)
        data[f"recall_at_{k}_reverse"] = (rank_reverse <= k).astype(float)
        data[f"recall_at_{k}_bidirectional"] = 0.5 * (
            (rank_forward <= k).astype(float) + (rank_reverse <= k).astype(float)
        )
    return pd.DataFrame(data)


def build_exact_triplet_positive_mask(
    metadata: pd.DataFrame,
    metadata_info: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Return an [N, N] mask of candidates that represent the same raw triplet.

    The training loss uses the diagonal as its positive target. For post-hoc
    retrieval, however, a repeated dataset row is also a valid answer. Treating
    those repeated rows as negatives would make R@K unnecessarily pessimistic.
    """
    if not metadata_info.get("metadata_available", False):
        return None

    key_columns = ["smiles", "poi_uniprot", "e3_uniprot"]
    if any(column not in metadata for column in key_columns):
        return None

    keys = metadata[key_columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    group_ids, _ = pd.factorize(keys, sort=False)
    group_sizes = pd.Series(group_ids).map(pd.Series(group_ids).value_counts()).to_numpy()
    positive_mask = group_ids[:, None] == group_ids[None, :]

    metadata["exact_triplet_group_id"] = group_ids
    metadata["exact_triplet_group_size"] = group_sizes
    metadata_info["exact_duplicate_triplet_groups_in_test"] = int(
        sum(size > 1 for size in pd.Series(group_ids).value_counts().to_numpy())
    )
    metadata_info["duplicate_aware_retrieval_available"] = True
    return positive_mask


def valid_positive_ranks(matrix: np.ndarray, positive_mask: np.ndarray) -> np.ndarray:
    """Rank the best valid positive in each row; smaller matrix values are better.

    Valid positives do not compete with one another. Invalid candidates tied
    with the best positive contribute half a rank, matching average-tie ranking.
    """
    if matrix.shape != positive_mask.shape:
        raise ValueError("matrix and positive_mask must have identical shapes.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    if not np.all(positive_mask.any(axis=1)):
        raise ValueError("Every query must have at least one valid positive.")

    best_positive = np.where(positive_mask, matrix, np.inf).min(axis=1)
    tolerance = np.maximum(1e-10, np.abs(best_positive) * 1e-7)
    invalid = ~positive_mask
    less = (
        invalid
        & (matrix < (best_positive[:, None] - tolerance[:, None]))
    ).sum(axis=1)
    tied = (
        invalid
        & np.isclose(matrix, best_positive[:, None], rtol=1e-7, atol=1e-10)
    ).sum(axis=1)
    return 1.0 + less.astype(float) + 0.5 * tied.astype(float)


def duplicate_aware_geometry(
    volumes: np.ndarray,
    recall_k: Sequence[int],
    positive_mask: np.ndarray,
) -> pd.DataFrame:
    """Compute retrieval and mismatch statistics without duplicate false negatives."""
    if volumes.shape != positive_mask.shape:
        raise ValueError("volumes and positive_mask must have identical shapes.")

    negative_mask = ~positive_mask
    forward_counts = negative_mask.sum(axis=1)
    reverse_counts = negative_mask.sum(axis=0)
    if np.any(forward_counts == 0) or np.any(reverse_counts == 0):
        raise ValueError("Duplicate-aware retrieval requires at least one invalid candidate per query.")

    positive = np.diag(volumes)
    negative_forward = np.where(negative_mask, volumes, 0.0).sum(axis=1) / forward_counts
    negative_reverse = np.where(negative_mask, volumes, 0.0).sum(axis=0) / reverse_counts
    rank_forward = valid_positive_ranks(volumes, positive_mask)
    rank_reverse = valid_positive_ranks(volumes.T, positive_mask.T)

    suffix = "_duplicate_aware"
    data: Dict[str, np.ndarray] = {
        f"mean_mismatched_volume_forward{suffix}": negative_forward,
        f"mean_mismatched_volume_reverse{suffix}": negative_reverse,
        f"volume_gap_forward{suffix}": negative_forward - positive,
        f"volume_gap_reverse{suffix}": negative_reverse - positive,
        f"volume_gap_bidirectional{suffix}": 0.5 * (
            negative_forward + negative_reverse
        ) - positive,
        f"rank_forward{suffix}": rank_forward,
        f"rank_reverse{suffix}": rank_reverse,
        f"reciprocal_rank_forward{suffix}": 1.0 / rank_forward,
        f"reciprocal_rank_reverse{suffix}": 1.0 / rank_reverse,
        f"reciprocal_rank_bidirectional{suffix}": 0.5 * (
            1.0 / rank_forward + 1.0 / rank_reverse
        ),
    }
    for k in recall_k:
        data[f"recall_at_{k}_forward{suffix}"] = (rank_forward <= k).astype(float)
        data[f"recall_at_{k}_reverse{suffix}"] = (rank_reverse <= k).astype(float)
        data[f"recall_at_{k}_bidirectional{suffix}"] = 0.5 * (
            (rank_forward <= k).astype(float) + (rank_reverse <= k).astype(float)
        )
    return pd.DataFrame(data)


def add_duplicate_aware_summary(
    summary: Dict[str, Any],
    per_sample: pd.DataFrame,
    recall_k: Sequence[int],
) -> None:
    suffix = "_duplicate_aware"
    summary[f"volume_gap_mean{suffix}"] = float(
        per_sample[f"volume_gap_bidirectional{suffix}"].mean()
    )
    summary[f"volume_gap_median{suffix}"] = float(
        per_sample[f"volume_gap_bidirectional{suffix}"].median()
    )
    summary[f"mrr_forward{suffix}"] = float(
        per_sample[f"reciprocal_rank_forward{suffix}"].mean()
    )
    summary[f"mrr_reverse{suffix}"] = float(
        per_sample[f"reciprocal_rank_reverse{suffix}"].mean()
    )
    summary[f"mrr_bidirectional{suffix}"] = float(
        per_sample[f"reciprocal_rank_bidirectional{suffix}"].mean()
    )
    for k in recall_k:
        for direction in ("forward", "reverse", "bidirectional"):
            column = f"recall_at_{k}_{direction}{suffix}"
            summary[column] = float(per_sample[column].mean())


def safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size > 1 else float("nan")


def summarize_geometry(
    label: str,
    checkpoint_info: Mapping[str, Any],
    features: ExtractedFeatures,
    volumes: np.ndarray,
    per_sample: pd.DataFrame,
    recall_k: Sequence[int],
) -> Dict[str, Any]:
    labels = features.label.astype(int)
    predictions = (features.probability >= 0.5).astype(int)
    mask = ~np.eye(volumes.shape[0], dtype=bool)
    summary: Dict[str, Any] = {
        "model": label,
        "checkpoint": checkpoint_info["path"],
        "checkpoint_sha256": checkpoint_info["sha256"],
        "checkpoint_state_source": checkpoint_info["state_source"],
        "n_samples": int(volumes.shape[0]),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": safe_roc_auc(labels, features.probability),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "matched_volume_mean": float(np.diag(volumes).mean()),
        "matched_volume_median": float(np.median(np.diag(volumes))),
        "mismatched_volume_mean": float(volumes[mask].mean()),
        "mismatched_volume_median": float(np.median(volumes[mask])),
        "volume_gap_mean": float(per_sample["volume_gap_bidirectional"].mean()),
        "volume_gap_median": float(per_sample["volume_gap_bidirectional"].median()),
        "matched_below_mean_mismatch_fraction": float(
            (
                per_sample["matched_volume"]
                < 0.5
                * (
                    per_sample["mean_mismatched_volume_forward"]
                    + per_sample["mean_mismatched_volume_reverse"]
                )
            ).mean()
        ),
        "mrr_forward": float(per_sample["reciprocal_rank_forward"].mean()),
        "mrr_reverse": float(per_sample["reciprocal_rank_reverse"].mean()),
        "mrr_bidirectional": float(per_sample["reciprocal_rank_bidirectional"].mean()),
    }
    for k in recall_k:
        summary[f"recall_at_{k}_forward"] = float(per_sample[f"recall_at_{k}_forward"].mean())
        summary[f"recall_at_{k}_reverse"] = float(per_sample[f"recall_at_{k}_reverse"].mean())
        summary[f"recall_at_{k}_bidirectional"] = float(
            per_sample[f"recall_at_{k}_bidirectional"].mean()
        )
    return summary


def paired_bootstrap_comparison(
    full: pd.DataFrame,
    no_tri: pd.DataFrame,
    recall_k: Sequence[int],
    replicates: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    if len(full) != len(no_tri):
        raise ValueError("Paired bootstrap requires the same samples for both models.")
    if replicates <= 0:
        return pd.DataFrame()
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap-confidence must be between 0 and 1.")

    metrics: List[Tuple[str, str]] = [
        ("matched_volume", "lower"),
        ("volume_gap_bidirectional", "higher"),
        ("reciprocal_rank_forward", "higher"),
        ("reciprocal_rank_reverse", "higher"),
        ("reciprocal_rank_bidirectional", "higher"),
    ]
    for k in recall_k:
        metrics.extend(
            [
                (f"recall_at_{k}_forward", "higher"),
                (f"recall_at_{k}_reverse", "higher"),
                (f"recall_at_{k}_bidirectional", "higher"),
            ]
        )

    duplicate_suffix = "_duplicate_aware"
    duplicate_metrics: List[Tuple[str, str]] = [
        (f"volume_gap_bidirectional{duplicate_suffix}", "higher"),
        (f"reciprocal_rank_forward{duplicate_suffix}", "higher"),
        (f"reciprocal_rank_reverse{duplicate_suffix}", "higher"),
        (f"reciprocal_rank_bidirectional{duplicate_suffix}", "higher"),
    ]
    for k in recall_k:
        duplicate_metrics.extend(
            [
                (f"recall_at_{k}_forward{duplicate_suffix}", "higher"),
                (f"recall_at_{k}_reverse{duplicate_suffix}", "higher"),
                (f"recall_at_{k}_bidirectional{duplicate_suffix}", "higher"),
            ]
        )
    metrics.extend(
        (name, direction)
        for name, direction in duplicate_metrics
        if name in full.columns and name in no_tri.columns
    )

    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(full), size=(replicates, len(full)))
    alpha = 0.5 * (1.0 - confidence)
    records: List[Dict[str, Any]] = []

    for name, direction in metrics:
        full_values = full[name].to_numpy(dtype=float)
        no_values = no_tri[name].to_numpy(dtype=float)
        raw_diff = full_values - no_values
        improvement_values = -raw_diff if direction == "lower" else raw_diff
        boot = improvement_values[sample_index].mean(axis=1)
        records.append(
            {
                "metric": name,
                "better_direction": direction,
                "complete_mean": float(full_values.mean()),
                "no_tri_mean": float(no_values.mean()),
                "raw_difference_complete_minus_no_tri": float(raw_diff.mean()),
                "improvement": float(improvement_values.mean()),
                "improvement_ci_low": float(np.quantile(boot, alpha)),
                "improvement_ci_high": float(np.quantile(boot, 1.0 - alpha)),
                "bootstrap_replicates": int(replicates),
                "confidence_level": float(confidence),
            }
        )
    return pd.DataFrame(records)


def retrieval_metric_columns(recall_k: Sequence[int], suffix: str = "") -> List[str]:
    columns = [
        f"reciprocal_rank_forward{suffix}",
        f"reciprocal_rank_reverse{suffix}",
        f"reciprocal_rank_bidirectional{suffix}",
    ]
    for k in recall_k:
        columns.extend(
            [
                f"recall_at_{k}_forward{suffix}",
                f"recall_at_{k}_reverse{suffix}",
                f"recall_at_{k}_bidirectional{suffix}",
            ]
        )
    return columns


def bootstrap_model_metric_intervals(
    results: Sequence[GeometryResult],
    recall_k: Sequence[int],
    replicates: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    """Bootstrap each model's mean retrieval score for panel-D error bars."""
    if replicates <= 0:
        return pd.DataFrame()
    if not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap-confidence must be between 0 and 1.")

    rng = np.random.default_rng(seed + 104729)
    alpha = 0.5 * (1.0 - confidence)
    records: List[Dict[str, Any]] = []
    for result in results:
        n = len(result.per_sample)
        sample_index = rng.integers(0, n, size=(replicates, n))
        candidates = retrieval_metric_columns(recall_k)
        duplicate_suffix = "_duplicate_aware"
        candidates.extend(retrieval_metric_columns(recall_k, duplicate_suffix))
        for column in candidates:
            if column not in result.per_sample:
                continue
            values = result.per_sample[column].to_numpy(dtype=float)
            boot = values[sample_index].mean(axis=1)
            records.append(
                {
                    "model": result.label,
                    "model_slug": result.slug,
                    "metric": column,
                    "mean": float(values.mean()),
                    "ci_low": float(np.quantile(boot, alpha)),
                    "ci_high": float(np.quantile(boot, 1.0 - alpha)),
                    "bootstrap_replicates": int(replicates),
                    "confidence_level": float(confidence),
                }
            )
    return pd.DataFrame(records)


def load_metadata(
    dataset_root: Path,
    dataset_name: str,
    indices: np.ndarray,
    standardize_dataframe: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    raw_csv = dataset_root / f"{dataset_name}.csv"
    fallback = pd.DataFrame(
        {
            "dataset_index": indices,
            "smiles": [""] * len(indices),
            "poi_uniprot": [""] * len(indices),
            "e3_uniprot": [""] * len(indices),
        }
    )
    info: Dict[str, Any] = {
        "raw_csv": str(raw_csv),
        "metadata_available": False,
        "exact_duplicate_triplet_rows_in_test": None,
        "exact_duplicate_triplet_groups_in_test": None,
        "duplicate_aware_retrieval_available": False,
    }
    if not raw_csv.exists():
        warnings.warn(f"Raw CSV not found; identifiers will be omitted: {raw_csv}")
        return fallback, info

    try:
        frame = standardize_dataframe(pd.read_csv(raw_csv))
        if int(indices.max()) >= len(frame):
            raise IndexError(
                f"Largest dataset index {indices.max()} exceeds raw CSV length {len(frame)}."
            )
        selected = frame.iloc[indices].reset_index(drop=True)
        metadata = pd.DataFrame(
            {
                "dataset_index": indices,
                "smiles": selected["Smiles"].astype(str),
                "poi_uniprot": selected["Uniprot"].astype(str),
                "e3_uniprot": selected["E3 ligase Uniprot"].astype(str),
            }
        )
        duplicate_rows = int(
            metadata.duplicated(
                subset=["smiles", "poi_uniprot", "e3_uniprot"], keep=False
            ).sum()
        )
        info.update(
            {
                "metadata_available": True,
                "exact_duplicate_triplet_rows_in_test": duplicate_rows,
            }
        )
        if duplicate_rows:
            warnings.warn(
                f"The test set contains {duplicate_rows} rows belonging to duplicated exact triplets. "
                "Both diagonal/loss-aligned and duplicate-aware retrieval will be reported."
            )
        return metadata, info
    except Exception as exc:
        warnings.warn(f"Could not load identifier metadata: {exc}")
        return fallback, info


def choose_heatmap_indices(n: int, maximum: int) -> np.ndarray:
    size = min(n, max(2, maximum))
    return np.unique(np.linspace(0, n - 1, num=size, dtype=int))


def sample_mismatches(
    volumes: np.ndarray,
    maximum: int,
    rng: np.random.Generator,
    positive_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    excluded = (
        positive_mask
        if positive_mask is not None
        else np.eye(volumes.shape[0], dtype=bool)
    )
    mismatch = volumes[~excluded]
    if mismatch.size <= maximum:
        return mismatch
    selected = rng.choice(mismatch.size, size=maximum, replace=False)
    return mismatch[selected]


def panel_d_metric_specs(
    recall_k: Sequence[int],
    duplicate_aware: bool,
) -> List[Tuple[str, str, str]]:
    suffix = "_duplicate_aware" if duplicate_aware else ""
    preferred_k = [k for k in (1, 5) if k in recall_k]
    if not preferred_k:
        preferred_k = list(recall_k[:2])

    specs: List[Tuple[str, str, str]] = [
        ("MRR\nm to (p,e)", f"mrr_forward{suffix}", f"reciprocal_rank_forward{suffix}"),
        ("MRR\n(p,e) to m", f"mrr_reverse{suffix}", f"reciprocal_rank_reverse{suffix}"),
    ]
    for k in preferred_k:
        specs.extend(
            [
                (
                    f"R@{k}\nm to (p,e)",
                    f"recall_at_{k}_forward{suffix}",
                    f"recall_at_{k}_forward{suffix}",
                ),
                (
                    f"R@{k}\n(p,e) to m",
                    f"recall_at_{k}_reverse{suffix}",
                    f"recall_at_{k}_reverse{suffix}",
                ),
            ]
        )
    return specs


def ci_error_for_bar(
    interval_table: pd.DataFrame,
    model_slug: str,
    metric: str,
    mean: float,
) -> Tuple[float, float]:
    if interval_table.empty:
        return 0.0, 0.0
    selected = interval_table[
        (interval_table["model_slug"] == model_slug)
        & (interval_table["metric"] == metric)
    ]
    if selected.empty:
        return 0.0, 0.0
    row = selected.iloc[0]
    return max(0.0, mean - float(row["ci_low"])), max(
        0.0, float(row["ci_high"]) - mean
    )


def plot_geometry(
    results: Sequence[GeometryResult],
    output_dir: Path,
    heatmap_size: int,
    mismatch_plot_samples: int,
    recall_k: Sequence[int],
    positive_mask: Optional[np.ndarray],
    retrieval_evaluation: str,
    retrieval_intervals: pd.DataFrame,
    bootstrap_confidence: float,
    font_family: str,
    seed: int,
    dpi: int,
) -> None:
    if not results:
        return
    by_slug = {result.slug: result for result in results}
    full = by_slug.get("complete", results[-1])
    no_tri = by_slug.get("no_tri")
    heatmap_results = [result for result in (no_tri, full) if result is not None]

    heat_idx = choose_heatmap_indices(len(full.features.dataset_index), heatmap_size)
    heat_values = [result.similarity[np.ix_(heat_idx, heat_idx)] for result in heatmap_results]
    pooled = np.concatenate([value.ravel() for value in heat_values])
    color_min, color_max = np.quantile(pooled, [0.01, 0.99])
    if np.isclose(color_min, color_max):
        color_min, color_max = float(pooled.min()), float(pooled.max() + 1e-6)

    # Give the shared colorbar its own GridSpec column.  If ``fig.colorbar``
    # steals space directly from the two heatmap axes, their fixed 1:1 aspect
    # makes panels A/B shrink against one side of the wider C/D panels.  With a
    # dedicated colorbar column, A and C (and likewise B and D) retain the same
    # column centre even though the heatmaps themselves remain square.
    fig = plt.figure(figsize=(13.5, 9.4), constrained_layout=True)
    geometry_grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.0, 1.0, 0.035),
        height_ratios=(1.0, 1.0),
    )
    axes = np.asarray(
        [
            [
                fig.add_subplot(geometry_grid[0, 0]),
                fig.add_subplot(geometry_grid[0, 1]),
            ],
            [
                fig.add_subplot(geometry_grid[1, 0]),
                fig.add_subplot(geometry_grid[1, 1]),
            ],
        ],
        dtype=object,
    )
    colorbar_axis = fig.add_subplot(geometry_grid[0, 2])
    heatmap_images: List[Any] = []
    for panel, axis, result in zip(("A", "B"), axes[0], heatmap_results):
        image = axis.imshow(
            result.similarity[np.ix_(heat_idx, heat_idx)],
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
            aspect="equal",
            interpolation="nearest",
        )
        heatmap_images.append(image)
        # Explicitly centre the square heatmap inside its GridSpec cell so that
        # its centre line coincides with the wider panel directly below it.
        axis.set_anchor("C")
        # Do not overlay a diagonal guide: every bright diagonal cell must come
        # from the learned similarity values themselves.
        axis.set_title(f"({panel}) {result.label}")
        axis.set_xlabel("Candidate POI–E3 observation")
        axis.set_ylabel("PROTAC anchor observation")

    if len(heatmap_results) == 1:
        axes[0, 1].axis("off")
        axes[0, 1].text(
            0.5,
            0.5,
            "Provide --no-tri-checkpoint\nfor a controlled comparison.",
            ha="center",
            va="center",
            fontsize=12,
        )

    cbar = fig.colorbar(
        heatmap_images[-1],
        cax=colorbar_axis,
        extend="both",
    )
    cbar.set_label("Loss-aligned similarity, -V/tau")

    rng = np.random.default_rng(seed)
    distribution_axis = axes[1, 0]
    positions: List[float] = []
    values: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    position = 1.0
    for result in results:
        short_label = "w/o volume loss" if result.slug == "no_tri" else "TriVol-P"
        positions.extend([position, position + 0.7])
        values.extend(
            [
                np.diag(result.volumes),
                sample_mismatches(
                    result.volumes,
                    mismatch_plot_samples,
                    rng,
                    positive_mask=positive_mask,
                ),
            ]
        )
        labels.extend([f"{short_label}\nMatched", f"{short_label}\nMismatched"])
        colors.extend(["#4C78A8", "#F58518"])
        position += 2.0
    boxes = distribution_axis.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.75)
    distribution_axis.set_xticks(positions)
    distribution_axis.set_xticklabels(labels, rotation=0, ha="center")
    distribution_axis.set_ylabel("Gramian volume V")
    distribution_axis.set_title("(C) Matched versus mismatched triplet volumes")
    distribution_axis.set_axisbelow(True)
    distribution_axis.grid(axis="y", alpha=0.25)

    retrieval_axis = axes[1, 1]
    duplicate_columns_available = all(
        "mrr_forward_duplicate_aware" in result.summary for result in results
    )
    duplicate_aware = retrieval_evaluation == "duplicate-aware" and duplicate_columns_available
    if retrieval_evaluation == "duplicate-aware" and not duplicate_columns_available:
        warnings.warn(
            "Duplicate-aware panel D was requested, but identifier metadata are unavailable; "
            "falling back to diagonal/loss-aligned retrieval."
        )
    metric_specs = panel_d_metric_specs(recall_k, duplicate_aware=duplicate_aware)
    x = np.arange(len(metric_specs))
    width = 0.8 / max(len(results), 1)
    palette = ["#9C755F", "#59A14F", "#4E79A7"]
    for model_index, result in enumerate(results):
        heights = [float(result.summary[summary_key]) for _, summary_key, _ in metric_specs]
        errors = [
            ci_error_for_bar(
                retrieval_intervals,
                result.slug,
                per_sample_key,
                height,
            )
            for height, (_, _, per_sample_key) in zip(heights, metric_specs)
        ]
        yerr = np.asarray(errors, dtype=float).T
        retrieval_axis.bar(
            x + (model_index - (len(results) - 1) / 2) * width,
            heights,
            width=width,
            label=result.label,
            color=palette[model_index % len(palette)],
            alpha=0.9,
            yerr=yerr,
            capsize=2.5,
            error_kw={"elinewidth": 0.9, "capthick": 0.9},
        )
    retrieval_axis.set_xticks(x)
    retrieval_axis.set_xticklabels(
        [name for name, _, _ in metric_specs], rotation=0, ha="center"
    )
    retrieval_axis.set_ylim(0.0, 1.18)
    retrieval_axis.set_ylabel("Retrieval score")
    qualifier = "Duplicate-aware " if duplicate_aware else "Diagonal "
    if retrieval_intervals.empty:
        uncertainty_text = ""
    else:
        confidence_percent = 100.0 * bootstrap_confidence
        uncertainty_text = f" ({confidence_percent:g}% CI)"
    retrieval_axis.set_title(
        f"(D) {qualifier}bidirectional retrieval{uncertainty_text}"
    )
    retrieval_axis.set_axisbelow(True)
    retrieval_axis.grid(axis="y", alpha=0.25)
    retrieval_axis.legend(loc="upper center", ncol=max(1, len(results)), frameon=False)

    fig.align_labels()
    enforce_figure_font(fig, font_family)
    fig.savefig(output_dir / "ternary_volume_geometry.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "ternary_volume_geometry.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ternary_volume_geometry.svg", bbox_inches="tight")
    plt.close(fig)


def run_self_test(api: Mapping[str, Any], device: torch.device, eps: float) -> None:
    # For each i, m_i=(p_i+e_i)/sqrt(2), making the matched three vectors
    # linearly dependent. For j != i, m_i, p_j, e_j are orthogonal.
    n = 4
    dim = 2 * n
    poi = torch.zeros(n, dim)
    e3 = torch.zeros(n, dim)
    for i in range(n):
        poi[i, 2 * i] = 1.0
        e3[i, 2 * i + 1] = 1.0
    protac = F.normalize(poi + e3, dim=-1)
    features = ExtractedFeatures(
        dataset_index=np.arange(n),
        label=np.array([0, 1, 0, 1]),
        probability=np.array([0.1, 0.9, 0.2, 0.8]),
        protac=protac.numpy(),
        poi=poi.numpy(),
        e3=e3.numpy(),
        checkpoint_source="synthetic",
    )
    volumes = compute_volume_matrix(
        features,
        device=device,
        tri_loss_class=api["TriVolumeLoss"],
        chunk_size=2,
        eps=eps,
    )
    diagonal = np.diag(volumes)
    off_diagonal = volumes[~np.eye(n, dtype=bool)]
    ranks_forward = average_tie_ranks(volumes)
    ranks_reverse = average_tie_ranks(volumes.T)
    if not (
        diagonal.max() < off_diagonal.min()
        and np.allclose(ranks_forward, 1.0)
        and np.allclose(ranks_reverse, 1.0)
    ):
        raise AssertionError(
            "Self-test failed: matched volumes should be smaller and rank first.\n"
            f"volumes={volumes}\nforward ranks={ranks_forward}\nreverse ranks={ranks_reverse}"
        )

    duplicate_example = np.array(
        [[0.1, 0.1, 0.9], [0.1, 0.1, 0.9], [0.9, 0.9, 0.1]],
        dtype=float,
    )
    duplicate_positive_mask = np.array(
        [[True, True, False], [True, True, False], [False, False, True]]
    )
    if not np.allclose(
        valid_positive_ranks(duplicate_example, duplicate_positive_mask),
        1.0,
    ):
        raise AssertionError("Self-test failed for duplicate-aware ranking.")
    print(
        "Self-test passed: volume calculation, diagonal ranking, and "
        "duplicate-aware ranking are valid."
    )


def main() -> None:
    args = parse_args()
    if any(k <= 0 for k in args.recall_k):
        raise ValueError("All --recall-k values must be positive integers.")
    args.recall_k = sorted(set(args.recall_k))
    if args.heatmap_size < 2:
        raise ValueError("--heatmap-size must be at least 2.")
    if args.mismatch_plot_samples <= 0:
        raise ValueError("--mismatch-plot-samples must be positive.")
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    os.chdir(project_root)
    api = import_project_api(project_root)

    requested_device = args.device or "auto"
    device = resolve_device(requested_device)
    if args.self_test:
        run_self_test(api, device=device, eps=args.eps)
        return

    if not args.full_checkpoint:
        raise ValueError("--full-checkpoint is required unless --self-test is used.")

    font_info = configure_publication_font(
        project_root=project_root,
        font_inputs=args.font_path,
        allow_fallback=args.allow_font_fallback,
    )
    print(
        "Figure font: "
        f"{font_info['resolved_family']} ({font_info['resolved_font_file']})"
    )

    config_path = resolve_path(project_root, args.config)
    assert config_path is not None
    cfg = load_toml(config_path)
    train_cfg = cfg["train"]
    seed = int(args.seed if args.seed is not None else cfg["model"].get("seed", 111))
    seed_everything(seed, num_threads=train_cfg.get("num_threads"))
    device = resolve_device(args.device or train_cfg.get("device", "auto"))
    print(f"Analysis device: {device}")

    output_dir = resolve_path(project_root, args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    test_loader, dataset_root, dataset_name = build_test_loader(
        cfg=cfg,
        args=args,
        project_root=project_root,
        api=api,
    )

    checkpoint_specs: List[Tuple[str, str, Path]] = []
    no_tri_path = resolve_path(project_root, args.no_tri_checkpoint)
    if no_tri_path is not None:
        checkpoint_specs.append(("w/o ternary volume loss", "no_tri", no_tri_path))
    full_path = resolve_path(project_root, args.full_checkpoint)
    assert full_path is not None
    if no_tri_path is not None and no_tri_path == full_path:
        raise ValueError("The TriVol-P and no-tri checkpoint paths are identical.")
    checkpoint_specs.append(("TriVol-P", "complete", full_path))

    temperature = float(train_cfg.get("tri_temp", 0.07))
    if temperature <= 0:
        raise ValueError("tri_temp must be positive.")
    results: List[GeometryResult] = []
    reference_indices: Optional[np.ndarray] = None
    reference_labels: Optional[np.ndarray] = None

    for label, slug, checkpoint_path in checkpoint_specs:
        model, checkpoint_info = load_model(
            checkpoint_path=checkpoint_path,
            model_cfg=cfg["model"],
            device=device,
            model_class=api["PROTAC_STAN"],
            allow_non_strict=args.allow_non_strict_load,
        )
        features = collect_features(
            model=model,
            loader=test_loader,
            device=device,
            checkpoint_source=checkpoint_info["state_source"],
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if reference_indices is None:
            reference_indices = features.dataset_index.copy()
            reference_labels = features.label.copy()
        else:
            if not np.array_equal(reference_indices, features.dataset_index):
                raise AssertionError("The two models were not evaluated on identical dataset indices.")
            if not np.array_equal(reference_labels, features.label):
                raise AssertionError("The two models produced inconsistent held-out labels.")

        volumes = compute_volume_matrix(
            features=features,
            device=device,
            tri_loss_class=api["TriVolumeLoss"],
            chunk_size=args.chunk_size,
            eps=args.eps,
        )
        similarity = -volumes / temperature
        per_sample = per_sample_geometry(volumes, args.recall_k)
        summary = summarize_geometry(
            label=label,
            checkpoint_info=checkpoint_info,
            features=features,
            volumes=volumes,
            per_sample=per_sample,
            recall_k=args.recall_k,
        )
        results.append(
            GeometryResult(
                label=label,
                slug=slug,
                features=features,
                volumes=volumes,
                similarity=similarity,
                per_sample=per_sample,
                summary=summary,
            )
        )

        np.save(output_dir / f"volume_matrix_{slug}.npy", volumes)
        np.save(output_dir / f"similarity_matrix_{slug}.npy", similarity)
        np.savez_compressed(
            output_dir / f"tri_features_{slug}.npz",
            dataset_index=features.dataset_index,
            label=features.label,
            probability=features.probability,
            protac=F.normalize(torch.from_numpy(features.protac), dim=-1).numpy(),
            poi=F.normalize(torch.from_numpy(features.poi), dim=-1).numpy(),
            e3=F.normalize(torch.from_numpy(features.e3), dim=-1).numpy(),
        )
        print(
            f"{label}: Accuracy={summary['accuracy']:.4f}, "
            f"AUROC={summary['roc_auc']:.4f}, Macro-F1={summary['macro_f1']:.4f}, "
            f"volume gap={summary['volume_gap_mean']:.6f}, "
            f"bidirectional MRR={summary['mrr_bidirectional']:.4f}"
        )

    assert reference_indices is not None
    metadata, metadata_info = load_metadata(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        indices=reference_indices,
        standardize_dataframe=api["standardize_protac_dataframe"],
    )
    positive_mask = build_exact_triplet_positive_mask(metadata, metadata_info)
    if positive_mask is not None:
        for result in results:
            duplicate_metrics = duplicate_aware_geometry(
                result.volumes,
                args.recall_k,
                positive_mask,
            )
            for column in duplicate_metrics.columns:
                result.per_sample[column] = duplicate_metrics[column].to_numpy()
            add_duplicate_aware_summary(result.summary, result.per_sample, args.recall_k)

    sample_table = metadata.copy()
    sample_table["label"] = results[0].features.label
    for result in results:
        sample_table[f"probability_{result.slug}"] = result.features.probability
        for column in result.per_sample.columns:
            sample_table[f"{column}_{result.slug}"] = result.per_sample[column].to_numpy()
    sample_table.to_csv(output_dir / "per_sample_geometry.csv", index=False)

    summary_table = pd.DataFrame([result.summary for result in results])
    summary_table.to_csv(output_dir / "geometry_summary.csv", index=False)

    bootstrap_table = pd.DataFrame()
    by_slug = {result.slug: result for result in results}
    if "complete" in by_slug and "no_tri" in by_slug:
        bootstrap_table = paired_bootstrap_comparison(
            full=by_slug["complete"].per_sample,
            no_tri=by_slug["no_tri"].per_sample,
            recall_k=args.recall_k,
            replicates=args.bootstrap,
            confidence=args.bootstrap_confidence,
            seed=seed,
        )
        bootstrap_table.to_csv(output_dir / "bootstrap_comparison.csv", index=False)

    retrieval_intervals = bootstrap_model_metric_intervals(
        results=results,
        recall_k=args.recall_k,
        replicates=args.bootstrap,
        confidence=args.bootstrap_confidence,
        seed=seed,
    )
    if not retrieval_intervals.empty:
        retrieval_intervals.to_csv(
            output_dir / "retrieval_confidence_intervals.csv",
            index=False,
        )

    plot_geometry(
        results=results,
        output_dir=output_dir,
        heatmap_size=args.heatmap_size,
        mismatch_plot_samples=args.mismatch_plot_samples,
        recall_k=args.recall_k,
        positive_mask=positive_mask,
        retrieval_evaluation=args.retrieval_evaluation,
        retrieval_intervals=retrieval_intervals,
        bootstrap_confidence=args.bootstrap_confidence,
        font_family=str(font_info["resolved_family"]),
        seed=seed,
        dpi=args.dpi,
    )

    if positive_mask is not None:
        valid_mismatch_count = int((~positive_mask).sum())
    else:
        sample_count = len(reference_indices)
        valid_mismatch_count = int(sample_count * (sample_count - 1))

    manifest = {
        "config": str(config_path),
        "project_root": str(project_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": seed,
        "tri_temperature": temperature,
        "eps": args.eps,
        "recall_k": list(args.recall_k),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_confidence": args.bootstrap_confidence,
        "figure_font": font_info,
        "figure_heatmap_observations": int(
            len(choose_heatmap_indices(len(reference_indices), args.heatmap_size))
        ),
        "figure_valid_mismatch_count_per_model": valid_mismatch_count,
        "figure_mismatch_values_used_per_model": int(
            min(valid_mismatch_count, args.mismatch_plot_samples)
        ),
        "retrieval_evaluation_requested": args.retrieval_evaluation,
        "retrieval_evaluation_used_in_figure": (
            "duplicate-aware"
            if args.retrieval_evaluation == "duplicate-aware" and positive_mask is not None
            else "diagonal"
        ),
        "split_mode": train_cfg.get("split_mode", "random"),
        "split_seed": int(train_cfg.get("split_seed", seed)),
        "train_ratio": float(train_cfg.get("train_ratio", 0.8)),
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "metadata": metadata_info,
        "models": [result.summary for result in results],
        "outputs": {
            "figure_png": "ternary_volume_geometry.png",
            "figure_pdf": "ternary_volume_geometry.pdf",
            "figure_svg": "ternary_volume_geometry.svg",
            "summary_csv": "geometry_summary.csv",
            "per_sample_csv": "per_sample_geometry.csv",
        },
        "notes": [
            "Volumes are computed from L2-normalized aux['tri_feats'] using the project's TriVolumeLoss implementation.",
            "Rows are PROTAC anchors; columns are contextualized paired POI–E3 observations.",
            "Strict diagonal retrieval remains in the CSV files because it is loss-aligned.",
            "When metadata are available, the main figure uses duplicate-aware retrieval so repeated raw triplets are valid positives rather than false negatives.",
            "Panel-D error bars are per-model nonparametric bootstrap confidence intervals over held-out observations.",
            "Heatmaps use a shared color scale clipped at the pooled 1st and 99th percentiles and contain no artificial diagonal overlay.",
            "The two heatmaps share one colorbar, and the manuscript title/sample-count statement is intentionally left to the external figure caption.",
            "All figure text is explicitly assigned to the resolved publication font; PDF output embeds TrueType fonts (fonttype 42), while SVG keeps text as editable text (svg.fonttype='none').",
            "Gramian volume is representation-space geometry, not physical ternary-complex volume.",
        ],
    }
    with (output_dir / "analysis_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\nAnalysis complete. Outputs written to: {output_dir}")
    print("Primary files:")
    print(f"  {output_dir / 'ternary_volume_geometry.png'}")
    print(f"  {output_dir / 'ternary_volume_geometry.pdf'}")
    print(f"  {output_dir / 'ternary_volume_geometry.svg'}")
    print(f"  {output_dir / 'geometry_summary.csv'}")
    print(f"  {output_dir / 'per_sample_geometry.csv'}")
    if not bootstrap_table.empty:
        print(f"  {output_dir / 'bootstrap_comparison.csv'}")
    if not retrieval_intervals.empty:
        print(f"  {output_dir / 'retrieval_confidence_intervals.csv'}")


if __name__ == "__main__":
    main()
