#!/usr/bin/env python3
"""Publication-ready relation-representation-space analysis for PROTAC-STAN.

The analysis uses the complete model only. It captures the exact tensor entering
``model.classifier`` and therefore visualizes the downstream relation
representation actually used for activity prediction. In the current active
configuration this tensor is the concatenation of the PROTAC-POI, PROTAC-E3,
POI-E3, and global ternary relation tokens.

The script:

1. reproduces the configured train/test split and descriptor normalization;
2. loads the selected complete-model checkpoint;
3. captures the pre-classifier relation representation for train and test data;
4. fits preprocessing/reduction on training representations and transforms the
   held-out test representations (UMAP is recommended);
5. plots true activity labels and predicted active-class probabilities using
   identical coordinates;
6. quantifies class organization in the original high-dimensional standardized
   relation space using the silhouette score and the between/within-class
   distance ratio;
7. assesses both organization statistics with a label-permutation null test.

Place this file in the project root beside ``model.py``, ``data.py``, and
``data_loader.py``, or pass ``--project-root`` explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    pairwise_distances,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None


@dataclass
class RelationFeatures:
    split: str
    dataset_index: np.ndarray
    label: np.ndarray
    probability: np.ndarray
    representation: np.ndarray


@dataclass
class PreprocessedRepresentations:
    train_high_dim: np.ndarray
    test_high_dim: np.ndarray
    train_pca: np.ndarray
    test_pca: np.ndarray
    pca_explained_variance: float
    pca_components: int


@dataclass
class OrganizationResult:
    silhouette: float
    distance_ratio: float
    within_class_distance: float
    between_class_distance: float
    silhouette_permutation_p: float
    distance_ratio_permutation_p: float
    silhouette_null_mean: float
    silhouette_null_std: float
    distance_ratio_null_mean: float
    distance_ratio_null_std: float
    permutation_table: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize and statistically assess the complete model's held-out "
            "relation representation space."
        )
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root containing model.py, data.py, and data_loader.py.",
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Training TOML configuration, relative to --project-root unless absolute.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Complete-model checkpoint. When omitted, the script uses "
            "<train.checkpoint_dir>/best.pt from the configuration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/relation_space",
        help="Directory for the figure, coordinates, statistics, and extracted features.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="auto, cpu, cuda, or cuda:N. Defaults to train.device in the configuration.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Active-class probability threshold used to mark errors.",
    )
    parser.add_argument(
        "--reducer",
        choices=("umap", "pca", "tsne"),
        default="umap",
        help=(
            "Two-dimensional reducer. UMAP is recommended because it is fitted "
            "on training representations and can transform held-out test samples."
        ),
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Dimensions retained before UMAP/t-SNE; capped by data rank.",
    )
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.10)
    parser.add_argument(
        "--umap-metric",
        default="euclidean",
        help="UMAP distance metric after training-set standardization and PCA.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1500)
    parser.add_argument(
        "--permutations",
        type=int,
        default=2000,
        help="Label permutations for the high-dimensional organization null test.",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Do not fit a StandardScaler on the training relation representations.",
    )
    parser.set_defaults(standardize=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Analysis seed; defaults to model.seed in the configuration.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--font-path",
        action="append",
        default=[],
        help=(
            "Optional Times New Roman font file or directory. Repeat to register "
            "regular, bold, italic, and bold-italic files."
        ),
    )
    parser.add_argument(
        "--allow-font-fallback",
        action="store_true",
        help=(
            "Allow a serif fallback if Times New Roman is unavailable. Do not use "
            "this option for the final manuscript figure."
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
        help="Run a deterministic synthetic test without importing the project or PyTorch.",
    )
    return parser.parse_args()


def resolve_path(root: Path, value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration not found: {path}")
    if tomllib is not None:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("Python < 3.11 requires the 'toml' package.") from exc
    return toml.load(str(path))


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int, torch_module: Optional[Any] = None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        if torch_module.cuda.is_available():
            torch_module.cuda.manual_seed(seed)
            torch_module.cuda.manual_seed_all(seed)


def configure_publication_font(
    project_root: Path,
    font_inputs: Sequence[str],
    allow_fallback: bool,
) -> Dict[str, Any]:
    """Register and strictly enforce Times New Roman for every figure element."""
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
                "or pass --font-path. The script stops rather than silently using "
                "a different typeface."
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
            "Do not use this fallback for the final manuscript figure."
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
    for text_artist in fig.findobj(match=matplotlib.text.Text):
        text_artist.set_fontfamily(family)


def import_project_api(project_root: Path) -> Dict[str, Any]:
    root_text = str(project_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        torch_module = importlib.import_module("torch")
        data_loader = importlib.import_module("data_loader")
        model_module = importlib.import_module("model")
    except Exception as exc:
        raise RuntimeError(
            "Could not import the project modules. Put this script beside "
            "model.py/data_loader.py/data.py or set --project-root."
        ) from exc

    return {
        "torch": torch_module,
        "PROTACLoader": getattr(data_loader, "PROTACLoader"),
        "collate_fn": getattr(data_loader, "collate_fn"),
        "PROTAC_STAN": getattr(model_module, "PROTAC_STAN"),
    }


def resolve_device(torch_module: Any, value: Optional[str]) -> Any:
    value = "auto" if value is None else str(value)
    if value == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if value.startswith("cuda") and not torch_module.cuda.is_available():
        warnings.warn(f"Requested {value}, but CUDA is unavailable; using CPU.")
        return torch_module.device("cpu")
    return torch_module.device(value)


def torch_load_compat(torch_module: Any, path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch_module.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(str(path), map_location=map_location)


def is_state_dict(torch_module: Any, candidate: Any) -> bool:
    return (
        isinstance(candidate, Mapping)
        and len(candidate) > 0
        and all(isinstance(key, str) for key in candidate.keys())
        and all(torch_module.is_tensor(value) for value in candidate.values())
    )


def strip_uniform_prefix(state_dict: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return dict(state_dict)


def extract_state_dict(torch_module: Any, checkpoint: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(checkpoint, torch_module.nn.Module):
        return dict(checkpoint.state_dict()), "serialized nn.Module"
    if isinstance(checkpoint, Mapping):
        # Training last.pt may contain both current and best states. Prefer the
        # selected best weights whenever they are present.
        for key in ("best_model_wts", "model_state_dict", "state_dict"):
            candidate = checkpoint.get(key)
            if is_state_dict(torch_module, candidate):
                return dict(candidate), key
        if is_state_dict(torch_module, checkpoint):
            return dict(checkpoint), "raw state_dict"
    raise ValueError(
        "Unsupported checkpoint format. Expected a serialized model, raw state_dict, "
        "or a dictionary containing best_model_wts/model_state_dict/state_dict."
    )


def resolve_checkpoint(
    project_root: Path,
    cfg: Mapping[str, Any],
    checkpoint_arg: Optional[str],
) -> Path:
    explicit = resolve_path(project_root, checkpoint_arg)
    if explicit is not None:
        return explicit
    checkpoint_dir = resolve_path(
        project_root,
        cfg["train"].get("checkpoint_dir", "saved_models/resume_checkpoint"),
    )
    assert checkpoint_dir is not None
    best_path = checkpoint_dir / "best.pt"
    if best_path.exists():
        return best_path
    last_path = checkpoint_dir / "last.pt"
    if last_path.exists():
        warnings.warn(
            f"{best_path} was not found; using {last_path} and preferring its best_model_wts field."
        )
        return last_path
    raise FileNotFoundError(
        f"Neither {best_path} nor {last_path} exists. Pass --checkpoint explicitly."
    )


def load_model(
    checkpoint_path: Path,
    model_cfg: Mapping[str, Any],
    device: Any,
    api: Mapping[str, Any],
    allow_non_strict: bool,
) -> Tuple[Any, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    torch_module = api["torch"]
    checkpoint = torch_load_compat(torch_module, checkpoint_path, map_location="cpu")
    state_dict, source = extract_state_dict(torch_module, checkpoint)
    for prefix in ("module.", "_orig_mod."):
        state_dict = strip_uniform_prefix(state_dict, prefix)

    model = api["PROTAC_STAN"](model_cfg)
    incompatible = model.load_state_dict(state_dict, strict=not allow_non_strict)
    if allow_non_strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        warnings.warn(
            "Non-strict checkpoint load: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    if not hasattr(model, "classifier"):
        raise AttributeError(
            "The model has no 'classifier' module. A pre-classifier hook is required "
            "to capture the exact downstream relation representation."
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


def build_loaders(
    cfg: Mapping[str, Any],
    args: argparse.Namespace,
    project_root: Path,
    api: Mapping[str, Any],
) -> Tuple[Any, Any, Path, str]:
    train_cfg = cfg["train"]
    if train_cfg.get("use_cv", False):
        raise ValueError(
            "This script analyzes a fixed held-out split. Set use_cv=false or run "
            "the script separately with a fold-specific loader and checkpoint."
        )
    dataset_root = resolve_path(
        project_root,
        train_cfg.get("dataset_root", "data/PROTAC-fine"),
    )
    assert dataset_root is not None
    dataset_name = str(train_cfg.get("dataset_name", "protac-fine"))
    reference_value = train_cfg.get("reference_json_path")
    reference_path = resolve_path(project_root, reference_value) if reference_value else None
    if reference_path is not None and not reference_path.exists():
        warnings.warn(
            f"reference_json_path does not exist ({reference_path}); reproducing "
            "main.py fallback to dataset-order splitting."
        )
        reference_path = None

    batch_size = int(args.batch_size or train_cfg.get("batch_size", 16))
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else train_cfg.get("num_workers", 0)
    )
    seed = int(args.seed if args.seed is not None else cfg["model"].get("seed", 111))
    train_loader, test_loader = api["PROTACLoader"](
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
    if train_loader is None:
        raise ValueError(
            "A training subset is required because preprocessing and UMAP are fitted "
            "without held-out labels on training representations."
        )

    # Rebuild both loaders deterministically with drop_last=False. The underlying
    # dataset already contains the training-fitted descriptor normalizer.
    torch_module = api["torch"]
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": api["collate_fn"],
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    deterministic_train_loader = torch_module.utils.data.DataLoader(
        train_loader.dataset,
        **loader_kwargs,
    )
    deterministic_test_loader = torch_module.utils.data.DataLoader(
        test_loader.dataset,
        **loader_kwargs,
    )
    return deterministic_train_loader, deterministic_test_loader, dataset_root, dataset_name


def move_batch_to_device(batch: Mapping[str, Any], device: Any) -> Dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def logits_to_active_probability(torch_module: Any, logits: Any) -> Any:
    if logits.ndim == 1:
        return torch_module.sigmoid(logits)
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with 1 or 2 dimensions, got {tuple(logits.shape)}")
    if logits.size(1) == 1:
        return torch_module.sigmoid(logits[:, 0])
    if logits.size(1) == 2:
        return torch_module.softmax(logits, dim=1)[:, 1]
    raise ValueError(
        f"Expected a binary classifier with one or two outputs, got {logits.size(1)}."
    )


def collect_relation_features(
    model: Any,
    loader: Iterable[Mapping[str, Any]],
    device: Any,
    torch_module: Any,
    split: str,
) -> RelationFeatures:
    indices: List[Any] = []
    labels: List[Any] = []
    probabilities: List[Any] = []
    representations: List[Any] = []
    captured_inputs: List[Any] = []

    def capture_classifier_input(module: Any, inputs: Tuple[Any, ...]) -> None:
        del module
        if len(inputs) != 1:
            raise RuntimeError(
                f"Expected classifier to receive one tensor, received {len(inputs)} inputs."
            )
        captured_inputs.append(inputs[0].detach().cpu().float())

    hook = model.classifier.register_forward_pre_hook(capture_classifier_input)
    try:
        with torch_module.inference_mode():
            for raw_batch in loader:
                if raw_batch.get("label") is None:
                    raise ValueError(f"The {split} split has no labels.")
                captured_inputs.clear()
                batch = move_batch_to_device(raw_batch, device)
                logits = model(
                    batch["protac"],
                    batch["poi_saprot"],
                    batch["e3_saprot"],
                    batch["protac_smile"],
                    batch["descriptor"],
                )
                if len(captured_inputs) != 1:
                    raise RuntimeError(
                        "The classifier-input hook did not capture exactly one tensor "
                        f"for a forward pass (captured={len(captured_inputs)})."
                    )
                relation_repr = captured_inputs[0]
                raw_labels = raw_batch["label"].detach().cpu().long()
                raw_indices = raw_batch["index"].detach().cpu().long()
                if relation_repr.size(0) != raw_labels.numel():
                    raise RuntimeError(
                        "Captured representation batch size does not match labels: "
                        f"{relation_repr.size(0)} vs {raw_labels.numel()}."
                    )
                indices.append(raw_indices)
                labels.append(raw_labels)
                probabilities.append(
                    logits_to_active_probability(torch_module, logits)
                    .detach()
                    .cpu()
                    .float()
                )
                representations.append(relation_repr)
    finally:
        hook.remove()

    if not indices:
        raise ValueError(f"The {split} loader produced no samples.")
    dataset_index = torch_module.cat(indices).numpy()
    order = np.argsort(dataset_index, kind="stable")
    result = RelationFeatures(
        split=split,
        dataset_index=dataset_index[order],
        label=torch_module.cat(labels).numpy()[order],
        probability=torch_module.cat(probabilities).numpy()[order],
        representation=torch_module.cat(representations).numpy()[order],
    )
    if result.representation.ndim != 2:
        raise ValueError(
            f"Expected a 2D relation representation, got {result.representation.shape}."
        )
    for name in ("probability", "representation"):
        values = getattr(result, name)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Non-finite values detected in {split} {name}.")
    unique_labels = np.unique(result.label)
    if not set(unique_labels.tolist()).issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found {unique_labels.tolist()}.")
    return result


def preprocess_representations(
    train: np.ndarray,
    test: np.ndarray,
    standardize: bool,
    pca_components: int,
    seed: int,
) -> PreprocessedRepresentations:
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError(
            f"Incompatible train/test representation shapes: {train.shape}, {test.shape}."
        )
    if standardize:
        scaler = StandardScaler()
        train_high_dim = scaler.fit_transform(train)
        test_high_dim = scaler.transform(test)
    else:
        train_high_dim = train.astype(np.float64, copy=True)
        test_high_dim = test.astype(np.float64, copy=True)

    max_components = min(train_high_dim.shape[0] - 1, train_high_dim.shape[1])
    if max_components < 2:
        raise ValueError("At least two non-degenerate PCA components are required.")
    components = min(max(2, int(pca_components)), max_components)
    pca = PCA(n_components=components, random_state=seed)
    train_pca = pca.fit_transform(train_high_dim)
    test_pca = pca.transform(test_high_dim)
    return PreprocessedRepresentations(
        train_high_dim=train_high_dim,
        test_high_dim=test_high_dim,
        train_pca=train_pca,
        test_pca=test_pca,
        pca_explained_variance=float(np.sum(pca.explained_variance_ratio_)),
        pca_components=int(components),
    )


def reduce_to_two_dimensions(
    preprocessed: PreprocessedRepresentations,
    reducer_name: str,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    reducer_name = reducer_name.lower()
    if reducer_name == "pca":
        coordinates = preprocessed.test_pca[:, :2]
        return coordinates, {
            "method": "PCA",
            "fit_scope": "training representations; held-out test transformed",
        }

    if reducer_name == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError(
                "UMAP is not installed. Run 'pip install umap-learn' in the model "
                "environment, or pass --reducer pca/--reducer tsne."
            ) from exc
        neighbors = min(
            max(2, int(args.umap_neighbors)),
            preprocessed.train_pca.shape[0] - 1,
        )
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=neighbors,
            min_dist=float(args.umap_min_dist),
            metric=str(args.umap_metric),
            random_state=seed,
            transform_seed=seed,
            n_jobs=1,
        )
        reducer.fit(preprocessed.train_pca)
        coordinates = reducer.transform(preprocessed.test_pca)
        return np.asarray(coordinates), {
            "method": "UMAP",
            "fit_scope": "training representations; held-out test transformed",
            "n_neighbors": neighbors,
            "min_dist": float(args.umap_min_dist),
            "metric": str(args.umap_metric),
        }

    if reducer_name == "tsne":
        n_test = preprocessed.test_pca.shape[0]
        max_perplexity = max(1.0, (n_test - 1) / 3.0)
        perplexity = min(float(args.tsne_perplexity), max_perplexity)
        if perplexity < 1.0:
            raise ValueError("Too few held-out samples for t-SNE.")
        kwargs: Dict[str, Any] = {
            "n_components": 2,
            "perplexity": perplexity,
            "learning_rate": "auto",
            "init": "pca",
            "random_state": seed,
        }
        parameter_names = inspect.signature(TSNE).parameters
        if "max_iter" in parameter_names:
            kwargs["max_iter"] = int(args.tsne_iterations)
        else:  # older scikit-learn
            kwargs["n_iter"] = int(args.tsne_iterations)
        coordinates = TSNE(**kwargs).fit_transform(preprocessed.test_pca)
        warnings.warn(
            "t-SNE has no out-of-sample transform in scikit-learn; it was fitted "
            "unsupervised on held-out representations. UMAP is preferred for the final analysis."
        )
        return np.asarray(coordinates), {
            "method": "t-SNE",
            "fit_scope": "held-out test representations only (unsupervised)",
            "perplexity": perplexity,
            "iterations": int(args.tsne_iterations),
        }

    raise ValueError(f"Unknown reducer: {reducer_name}")


def distance_components(
    distances: np.ndarray,
    labels: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> Tuple[float, float, float]:
    pair_distances = distances[pair_i, pair_j]
    same_class = labels[pair_i] == labels[pair_j]
    if not same_class.any() or same_class.all():
        raise ValueError("Both within-class and between-class sample pairs are required.")
    within = float(pair_distances[same_class].mean())
    between = float(pair_distances[~same_class].mean())
    ratio = float(between / max(within, np.finfo(float).eps))
    return within, between, ratio


def analyze_class_organization(
    high_dim_test: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> OrganizationResult:
    if permutations <= 0:
        raise ValueError("--permutations must be positive.")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 2 or np.any(counts < 2):
        raise ValueError(
            "Silhouette analysis requires two classes with at least two held-out samples each."
        )
    distances = pairwise_distances(high_dim_test, metric="euclidean")
    np.fill_diagonal(distances, 0.0)
    pair_i, pair_j = np.triu_indices(labels.size, k=1)
    observed_silhouette = float(
        silhouette_score(distances, labels, metric="precomputed")
    )
    within, between, observed_ratio = distance_components(
        distances,
        labels,
        pair_i,
        pair_j,
    )

    rng = np.random.default_rng(seed)
    null_silhouette = np.empty(permutations, dtype=np.float64)
    null_ratio = np.empty(permutations, dtype=np.float64)
    for permutation_index in range(permutations):
        permuted = rng.permutation(labels)
        null_silhouette[permutation_index] = silhouette_score(
            distances,
            permuted,
            metric="precomputed",
        )
        _, _, null_ratio[permutation_index] = distance_components(
            distances,
            permuted,
            pair_i,
            pair_j,
        )

    silhouette_p = float(
        (1 + np.count_nonzero(null_silhouette >= observed_silhouette))
        / (permutations + 1)
    )
    ratio_p = float(
        (1 + np.count_nonzero(null_ratio >= observed_ratio))
        / (permutations + 1)
    )
    table = pd.DataFrame(
        {
            "permutation": np.arange(1, permutations + 1, dtype=int),
            "silhouette": null_silhouette,
            "between_within_distance_ratio": null_ratio,
        }
    )
    return OrganizationResult(
        silhouette=observed_silhouette,
        distance_ratio=observed_ratio,
        within_class_distance=within,
        between_class_distance=between,
        silhouette_permutation_p=silhouette_p,
        distance_ratio_permutation_p=ratio_p,
        silhouette_null_mean=float(null_silhouette.mean()),
        silhouette_null_std=float(null_silhouette.std(ddof=1)),
        distance_ratio_null_mean=float(null_ratio.mean()),
        distance_ratio_null_std=float(null_ratio.std(ddof=1)),
        permutation_table=table,
    )


def classification_summary(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1.")
    predictions = (probabilities >= threshold).astype(int)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(unique, counts)}
    return {
        "n": int(labels.size),
        "n_inactive": int(count_map.get(0, 0)),
        "n_active": int(count_map.get(1, 0)),
        "n_misclassified": int(np.count_nonzero(predictions != labels)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }


def plot_relation_space(
    coordinates: np.ndarray,
    features: RelationFeatures,
    reducer_info: Mapping[str, Any],
    output_dir: Path,
    dpi: int,
    font_info: Mapping[str, Any],
) -> Tuple[Path, Path, Path]:
    if coordinates.shape != (features.label.size, 2):
        raise ValueError(
            f"Expected coordinates with shape {(features.label.size, 2)}, got {coordinates.shape}."
        )
    inactive_color = "#4C78A8"
    active_color = "#F58518"

    # The physical width is close to a journal's full two-column figure width.
    # This avoids shrinking 12-inch artwork (and its text) when inserted at
    # approximately 6.7-7.2 inches in the manuscript.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.905, bottom=0.17, top=0.88, wspace=0.23)
    reducer_label = str(reducer_info["method"])
    x_label = f"{reducer_label} 1"
    y_label = f"{reducer_label} 2"

    for label_value, label_text, color in (
        (0, "Inactive", inactive_color),
        (1, "Active", active_color),
    ):
        mask = features.label == label_value
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=20,
            c=color,
            alpha=0.84,
            edgecolors="white",
            linewidths=0.25,
            label=f"{label_text} (n={int(mask.sum())})",
        )

    probability_scatter = axes[1].scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=20,
        c=features.probability,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        alpha=0.88,
        edgecolors="white",
        linewidths=0.25,
    )

    axes[0].set_title("(A) Ground-truth activity labels", pad=6, fontsize=10.5)
    axes[1].set_title("(B) Predicted active-class probability", pad=6, fontsize=10.5)
    for axis in axes:
        axis.set_xlabel(x_label, fontsize=9.5)
        axis.set_ylabel(y_label, fontsize=9.5)
        axis.tick_params(labelsize=8.5)
        axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.50)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    axes[0].legend(
        loc="best",
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
        edgecolor="#BFBFBF",
    )
    colorbar = fig.colorbar(
        probability_scatter,
        ax=axes[1],
        fraction=0.050,
        pad=0.035,
    )
    colorbar.ax.tick_params(labelsize=8.5)
    colorbar.set_label("Predicted probability of active class", fontsize=9.0)
    enforce_figure_font(fig, str(font_info["resolved_family"]))

    png_path = output_dir / "relation_space_visualization.png"
    pdf_path = output_dir / "relation_space_visualization.pdf"
    svg_path = output_dir / "relation_space_visualization.svg"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    # Keep the scatter markers as vector paths in PDF and SVG. The dpi setting
    # only affects unavoidable raster artists such as the continuous colorbar.
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path, svg_path


def save_outputs(
    output_dir: Path,
    train_features: RelationFeatures,
    test_features: RelationFeatures,
    coordinates: np.ndarray,
    preprocessed: PreprocessedRepresentations,
    classification: Mapping[str, Any],
    organization: OrganizationResult,
    reducer_info: Mapping[str, Any],
    checkpoint_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    dataset_root: Path,
    dataset_name: str,
    font_info: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    predictions = (test_features.probability >= args.threshold).astype(int)
    coordinates_table = pd.DataFrame(
        {
            "dataset_index": test_features.dataset_index,
            "label": test_features.label,
            "predicted_probability": test_features.probability,
            "prediction": predictions,
            "correct": predictions == test_features.label,
            "coordinate_1": coordinates[:, 0],
            "coordinate_2": coordinates[:, 1],
        }
    )
    coordinates_table.to_csv(output_dir / "relation_space_coordinates.csv", index=False)
    organization.permutation_table.to_csv(
        output_dir / "relation_space_permutation_null.csv",
        index=False,
    )
    np.savez_compressed(
        output_dir / "relation_representations.npz",
        train_dataset_index=train_features.dataset_index,
        train_label=train_features.label,
        train_representation=train_features.representation,
        test_dataset_index=test_features.dataset_index,
        test_label=test_features.label,
        test_probability=test_features.probability,
        test_representation=test_features.representation,
        test_representation_analysis_space=preprocessed.test_high_dim,
        test_coordinates=coordinates,
    )

    summary: Dict[str, Any] = {
        "analysis": "complete-model downstream relation representation space",
        "interpretation_boundary": (
            "The analysis characterizes the integrated complete-model relation "
            "representation and does not isolate the contribution of any single component."
        ),
        "dataset": {
            "root": str(dataset_root),
            "name": dataset_name,
            "split_mode": cfg["train"].get("split_mode", "random"),
            "split_seed": int(
                cfg["train"].get(
                    "split_seed",
                    args.seed if args.seed is not None else cfg["model"].get("seed", 111),
                )
            ),
            "train_n": int(train_features.label.size),
            "test_n": int(test_features.label.size),
        },
        "checkpoint": dict(checkpoint_info),
        "representation": {
            "source": "exact input tensor captured immediately before model.classifier",
            "dimension": int(test_features.representation.shape[1]),
            "relation_pooling_enabled": bool(
                cfg["model"].get("relation_pooling", {}).get("enabled", True)
            ),
            "relation_information_bottleneck_enabled": bool(
                cfg["model"].get("relation_ib", {}).get("enabled", False)
            ),
            "standardized_with_training_statistics": bool(args.standardize),
            "pca_components": int(preprocessed.pca_components),
            "pca_explained_variance": float(preprocessed.pca_explained_variance),
        },
        "reduction": dict(reducer_info),
        "classification": dict(classification),
        "high_dimensional_organization": {
            "distance_metric": "Euclidean",
            "silhouette": float(organization.silhouette),
            "within_class_distance": float(organization.within_class_distance),
            "between_class_distance": float(organization.between_class_distance),
            "between_within_distance_ratio": float(organization.distance_ratio),
            "permutations": int(len(organization.permutation_table)),
            "silhouette_permutation_p": float(
                organization.silhouette_permutation_p
            ),
            "distance_ratio_permutation_p": float(
                organization.distance_ratio_permutation_p
            ),
            "silhouette_null_mean": float(organization.silhouette_null_mean),
            "silhouette_null_std": float(organization.silhouette_null_std),
            "distance_ratio_null_mean": float(organization.distance_ratio_null_mean),
            "distance_ratio_null_std": float(organization.distance_ratio_null_std),
        },
        "font": dict(font_info),
        "seed": int(
            args.seed if args.seed is not None else cfg["model"].get("seed", 111)
        ),
        "outputs": {
            "figure_png": "relation_space_visualization.png",
            "figure_pdf": "relation_space_visualization.pdf",
            "figure_svg": "relation_space_visualization.svg",
            "coordinates_csv": "relation_space_coordinates.csv",
            "permutation_csv": "relation_space_permutation_null.csv",
            "features_npz": "relation_representations.npz",
        },
    }
    with (output_dir / "relation_space_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def run_self_test(
    output_dir: Path,
    font_info: Mapping[str, Any],
    seed: int,
    dpi: int,
) -> None:
    rng = np.random.default_rng(seed)
    dimensions = 18
    train_labels = np.repeat([0, 1], [160, 140])
    test_labels = np.repeat([0, 1], [64, 56])
    direction = rng.normal(size=dimensions)
    direction /= np.linalg.norm(direction)
    train_repr = rng.normal(scale=0.9, size=(train_labels.size, dimensions))
    test_repr = rng.normal(scale=0.9, size=(test_labels.size, dimensions))
    train_repr += (train_labels[:, None] * 2 - 1) * direction * 1.35
    test_repr += (test_labels[:, None] * 2 - 1) * direction * 1.35
    score = test_repr @ direction + rng.normal(scale=0.35, size=test_labels.size)
    probability = 1.0 / (1.0 + np.exp(-score))

    train_features = RelationFeatures(
        split="train",
        dataset_index=np.arange(train_labels.size),
        label=train_labels,
        probability=np.zeros(train_labels.size),
        representation=train_repr,
    )
    test_features = RelationFeatures(
        split="test",
        dataset_index=np.arange(test_labels.size),
        label=test_labels,
        probability=probability,
        representation=test_repr,
    )
    preprocessed = preprocess_representations(
        train=train_repr,
        test=test_repr,
        standardize=True,
        pca_components=12,
        seed=seed,
    )
    organization = analyze_class_organization(
        high_dim_test=preprocessed.test_high_dim,
        labels=test_labels,
        permutations=250,
        seed=seed,
    )
    if organization.silhouette <= organization.silhouette_null_mean:
        raise AssertionError("Synthetic class organization was not detected.")
    coordinates = preprocessed.test_pca[:, :2]
    classification = classification_summary(test_labels, probability, threshold=0.5)
    reducer_info = {
        "method": "PCA",
        "fit_scope": "synthetic training representations; synthetic test transformed",
    }
    plot_relation_space(
        coordinates=coordinates,
        features=test_features,
        reducer_info=reducer_info,
        output_dir=output_dir,
        dpi=dpi,
        font_info=font_info,
    )
    expected_figures = (
        output_dir / "relation_space_visualization.png",
        output_dir / "relation_space_visualization.pdf",
        output_dir / "relation_space_visualization.svg",
    )
    missing_figures = [path.name for path in expected_figures if not path.exists()]
    if missing_figures:
        raise AssertionError(
            "Synthetic figure output was not written: " + ", ".join(missing_figures)
        )
    # Keep references alive to exercise the same data shapes as the real path.
    assert train_features.representation.shape[1] == test_features.representation.shape[1]
    print(
        "Self-test passed: preprocessing, high-dimensional permutation analysis, "
        "classification summary, and publication plotting are valid."
    )


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")
    output_dir = resolve_path(project_root, args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    font_info = configure_publication_font(
        project_root=project_root,
        font_inputs=args.font_path,
        allow_fallback=args.allow_font_fallback,
    )
    print(
        "Figure font: "
        f"{font_info['resolved_family']} ({font_info['resolved_font_file']})"
    )

    if args.self_test:
        seed = int(args.seed if args.seed is not None else 111)
        seed_everything(seed)
        run_self_test(output_dir=output_dir, font_info=font_info, seed=seed, dpi=args.dpi)
        return

    os.chdir(project_root)
    config_path = resolve_path(project_root, args.config)
    assert config_path is not None
    cfg = load_toml(config_path)
    seed = int(args.seed if args.seed is not None else cfg["model"].get("seed", 111))
    api = import_project_api(project_root)
    torch_module = api["torch"]
    seed_everything(seed, torch_module=torch_module)
    device = resolve_device(
        torch_module,
        args.device or cfg["train"].get("device", "auto"),
    )
    print(f"Analysis device: {device}")

    checkpoint_path = resolve_checkpoint(project_root, cfg, args.checkpoint)
    train_loader, test_loader, dataset_root, dataset_name = build_loaders(
        cfg=cfg,
        args=args,
        project_root=project_root,
        api=api,
    )
    model, checkpoint_info = load_model(
        checkpoint_path=checkpoint_path,
        model_cfg=cfg["model"],
        device=device,
        api=api,
        allow_non_strict=args.allow_non_strict_load,
    )
    train_features = collect_relation_features(
        model=model,
        loader=train_loader,
        device=device,
        torch_module=torch_module,
        split="train",
    )
    test_features = collect_relation_features(
        model=model,
        loader=test_loader,
        device=device,
        torch_module=torch_module,
        split="test",
    )
    del model
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()

    print(
        "Captured relation representations: "
        f"train={train_features.representation.shape}, "
        f"test={test_features.representation.shape}"
    )
    preprocessed = preprocess_representations(
        train=train_features.representation,
        test=test_features.representation,
        standardize=bool(args.standardize),
        pca_components=int(args.pca_components),
        seed=seed,
    )
    coordinates, reducer_info = reduce_to_two_dimensions(
        preprocessed=preprocessed,
        reducer_name=args.reducer,
        args=args,
        seed=seed,
    )
    classification = classification_summary(
        labels=test_features.label,
        probabilities=test_features.probability,
        threshold=float(args.threshold),
    )
    organization = analyze_class_organization(
        high_dim_test=preprocessed.test_high_dim,
        labels=test_features.label,
        permutations=int(args.permutations),
        seed=seed,
    )
    png_path, pdf_path, svg_path = plot_relation_space(
        coordinates=coordinates,
        features=test_features,
        reducer_info=reducer_info,
        output_dir=output_dir,
        dpi=int(args.dpi),
        font_info=font_info,
    )
    save_outputs(
        output_dir=output_dir,
        train_features=train_features,
        test_features=test_features,
        coordinates=coordinates,
        preprocessed=preprocessed,
        classification=classification,
        organization=organization,
        reducer_info=reducer_info,
        checkpoint_info=checkpoint_info,
        cfg=cfg,
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        font_info=font_info,
        args=args,
    )
    print(f"Saved figure: {png_path}")
    print(f"Saved vector figure: {pdf_path}")
    print(f"Saved vector figure: {svg_path}")
    print(f"Saved analysis files under: {output_dir}")


if __name__ == "__main__":
    main()
