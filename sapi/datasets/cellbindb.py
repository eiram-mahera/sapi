"""
CellBinDB dataset loader.

The loader performs ONLY dataset-correctness preprocessing ("Layer A").  It does
NOT normalise, scale, or augment images — every model normalises, augments and
otherwise preprocesses its own inputs (for training, validation and test).

Layer-A preprocessing is declared per dataset in configs/datasets/<name>.yaml
via the ``preprocessing`` field, so adding a new modality needs no code change
here — only a YAML file (and the data folder).  Supported modes:

  * ``none``            Return raw float32 values unchanged.
  * ``invert``          Return ``arr.max() - arr`` (cells dark on a bright
                        background become bright, matching the other modalities;
                        used by mIF).
  * ``stain_separate``  H&E RGB is unmixed with rgb2hed; the ``.image`` property
                        returns only the Hematoxylin channel (1-channel models)
                        while ``.raw_image`` returns the raw RGB (3-channel
                        models).

If a dataset YAML lacks the ``preprocessing`` field, the mode is inferred from
the dataset name (H&E → stain_separate, mIF → invert, else none) so older
configs keep working.

Images are returned as float32 in their natural intensity range (no scaling).
Fluorescence files are a mix of uint8 and uint16; H&E is uint8 RGB.  tifffile
decodes the TIFF byte order (II/MM) on read, so values are always correct.
Instance masks are returned as uint16 label maps (0 = background).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from skimage.color import rgb2hed

logger = logging.getLogger(__name__)

# Project root = two levels above this file's package (…/sapi/datasets/).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_DATASET_CONFIG_DIR: Path = _PROJECT_ROOT / "configs" / "datasets"

# Layer-A preprocessing modes (the set of valid `preprocessing:` YAML values).
_PREPROCESSING_MODES = ("none", "invert", "stain_separate")

# Fallback used only when a dataset YAML omits the `preprocessing` field, so
# pre-existing configs keep their historical behaviour.
_HE_DATASETS = ("HE", "10xGenomics_HE")

DATASET_NAMES: List[str] = [
    "DAPI",
    "HE",
    "mIF",
    "ssDNA",
    "10xGenomics_DAPI",
    "10xGenomics_HE",
]

# Channel count of the .image property after Layer-A preprocessing (always 1;
# H&E is reduced to the Hematoxylin channel).  3-channel models read .raw_image.
DATASET_CHANNELS: Dict[str, int] = {name: 1 for name in DATASET_NAMES}


def _load_dataset_config(dataset_name: str) -> dict:
    """Return the parsed configs/datasets/<name>.yaml, or {} if it is absent."""
    path = _DATASET_CONFIG_DIR / f"{dataset_name}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - config read is best-effort
        logger.warning(f"Could not read dataset config {path}: {exc}")
        return {}


def _resolve_preprocessing(dataset_name: str, config: dict | None = None) -> str:
    """Resolve the Layer-A preprocessing mode for *dataset_name*.

    Priority: explicit ``preprocessing`` field in the dataset YAML → name-based
    fallback (H&E → stain_separate, mIF → invert, else none).
    """
    cfg = config if config is not None else _load_dataset_config(dataset_name)
    mode = cfg.get("preprocessing")
    if mode in _PREPROCESSING_MODES:
        return mode
    if mode is not None:
        logger.warning(
            f"Unknown preprocessing '{mode}' for dataset '{dataset_name}'; "
            f"falling back to name-based inference."
        )
    if dataset_name in _HE_DATASETS:
        return "stain_separate"
    if dataset_name == "mIF":
        return "invert"
    return "none"


class CellBinDBSample:
    """Lazy-loaded single sample (image + instance mask), read on access."""

    def __init__(self, sample_dir: Path, dataset_name: str, preprocessing: str = "none"):
        self.sample_dir = sample_dir
        self.dataset_name = dataset_name
        self.preprocessing = preprocessing
        self.name = sample_dir.name

        files = {f.name: f for f in sample_dir.iterdir() if not f.name.startswith(".")}
        self._img_path = self._find(files, "-img.tif")
        self._inst_path = self._find(files, "-instancemask.tif")
        self._mask_path = self._find(files, "-mask.tif")

        if self._img_path is None:
            raise FileNotFoundError(f"No *-img.tif in {sample_dir}")
        if self._inst_path is None:
            raise FileNotFoundError(f"No *-instancemask.tif in {sample_dir}")

    @staticmethod
    def _find(files: dict, suffix: str) -> Optional[Path]:
        for name, path in files.items():
            if name.endswith(suffix):
                return path
        return None

    def _load_tif(self, path: Path) -> np.ndarray:
        # tifffile decodes the TIFF byte order (II/MM) on read and returns a
        # correctly-ordered native array, so no manual byteswap is needed.
        return tifffile.imread(str(path))

    @property
    def image(self) -> np.ndarray:
        """1-channel float32 image (H, W), Layer-A corrected, NOT normalised.

        stain_separate -> Hematoxylin channel; invert -> max-img; none -> raw.
        """
        if self.preprocessing == "stain_separate":
            raw = self._load_tif(self._img_path)                 # uint8 H x W x 3
            hed = rgb2hed(raw.astype(np.float64) / 255.0)        # stain separation
            return hed[..., 0].astype(np.float32)                # Hematoxylin channel

        arr = self._load_tif(self._img_path).astype(np.float32)
        if self.preprocessing == "invert":
            arr = arr.max() - arr                                # cells dark -> bright
        return arr

    @property
    def raw_image(self) -> np.ndarray:
        """Raw RGB float32 (H, W, 3) for stain-separated modalities; else == .image."""
        if self.preprocessing == "stain_separate":
            return self._load_tif(self._img_path).astype(np.float32)
        return self.image

    @property
    def instance_mask(self) -> np.ndarray:
        """Instance label map, uint16 (H, W). 0 = background."""
        return self._load_tif(self._inst_path).astype(np.uint16)

    @property
    def binary_mask(self) -> np.ndarray:
        """Binary semantic mask, uint8 (H, W). 1 = cell region."""
        return self._load_tif(self._mask_path).astype(np.uint8)

    def __repr__(self) -> str:
        return f"CellBinDBSample(dataset={self.dataset_name}, name={self.name})"


class CellBinDBDataset:
    """Dataset wrapper for one CellBinDB modality. Loads all samples."""

    def __init__(self, dataset_name: str, data_root: str | Path):
        # A dataset is valid if it is one of the built-in modalities OR has a
        # configs/datasets/<name>.yaml — so a new modality is added with a YAML
        # (+ data folder), no code change here.
        if dataset_name not in DATASET_NAMES \
                and not (_DATASET_CONFIG_DIR / f"{dataset_name}.yaml").exists():
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Provide "
                f"configs/datasets/{dataset_name}.yaml or choose from {DATASET_NAMES}."
            )

        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.preprocessing = _resolve_preprocessing(dataset_name)

        ds_dir = self.data_root / dataset_name
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}")

        self._samples = sorted(
            (
                CellBinDBSample(d, dataset_name, preprocessing=self.preprocessing)
                for d in ds_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ),
            key=lambda s: s.name,
        )
        logger.info(
            f"CellBinDB [{dataset_name}]: {len(self._samples)} samples  "
            f"(preprocessing={self.preprocessing})"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> CellBinDBSample:
        return self._samples[idx]

    def __iter__(self):
        return iter(self._samples)

    @property
    def num_channels(self) -> int:
        # .image is always 1-channel after Layer-A (H&E is reduced to the
        # Hematoxylin channel); default to 1 for datasets not in the table.
        return DATASET_CHANNELS.get(self.dataset_name, 1)

    @property
    def sample_names(self) -> List[str]:
        return [s.name for s in self._samples]

    def get_images_and_masks(self, raw_he: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (images, instance_masks) for all samples.

        raw_he=True returns raw RGB for H&E (3-channel models); otherwise the
        stain-separated Hematoxylin channel is returned.
        """
        images = [(s.raw_image if raw_he else s.image) for s in self._samples]
        masks = [s.instance_mask for s in self._samples]
        return images, masks

    def get_samples_by_indices(
        self, indices: List[int], raw_he: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return (images, instance_masks) for the given sample indices."""
        images = [(self._samples[i].raw_image if raw_he else self._samples[i].image) for i in indices]
        masks = [self._samples[i].instance_mask for i in indices]
        return images, masks


def load_dataset(dataset_name: str, data_root: str | Path) -> CellBinDBDataset:
    """Instantiate a CellBinDBDataset."""
    return CellBinDBDataset(dataset_name=dataset_name, data_root=data_root)
