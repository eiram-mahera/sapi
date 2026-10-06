"""
Shared tissue-stratified support/test split.

The fewshot_finetuning experiment selects its support set as one random image
per tissue group (seed = ``config.seed``), with every remaining labelled
image forming the test set.  That logic lives here so it — and any future
tissue-based experiment — share a single implementation instead of copying it.

Tissue labels come from ``configs/tissue_labels.json`` (derived once from
Supplementary Table S2 of the CellBinDB paper; full coverage across all six
datasets).
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Canonical location of the tissue label map, relative to the project root.
_TISSUE_LABELS_RELPATH: str = "configs/tissue_labels.json"

# Project root = two levels above this file's package (…/sapi/experiments/).
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


@dataclass
class TissueSplit:
    """Result of a tissue-stratified split for one dataset."""

    support_idx: List[int] = field(default_factory=list)
    support_names: List[str] = field(default_factory=list)
    support_tissues: List[str] = field(default_factory=list)
    test_idx: List[int] = field(default_factory=list)
    groups: Dict[str, List[int]] = field(default_factory=dict)
    total_samples: int = 0


def load_tissue_labels(dataset_name: str, project_root: Path | None = None) -> Dict[str, str]:
    """Return ``{sample_folder_name: tissue_label}`` for *dataset_name*."""
    root = project_root or _PROJECT_ROOT
    labels_path = root / _TISSUE_LABELS_RELPATH
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Tissue label file not found: {labels_path}. "
            "Run the S2 parsing script to generate it or check the path."
        )
    with open(labels_path) as fh:
        all_labels: dict = json.load(fh)
    if dataset_name not in all_labels:
        raise KeyError(
            f"Dataset '{dataset_name}' not found in {labels_path}. "
            f"Available: {list(all_labels.keys())}"
        )
    return all_labels[dataset_name]


def tissue_stratified_split(
    dataset,
    dataset_name: str,
    seed: int,
    project_root: Path | None = None,
) -> TissueSplit:
    """Select one image per tissue group as the support set; the rest are test.

    Parameters
    ----------
    dataset
        A loaded dataset exposing ``sample_names`` and ``__len__``.
    dataset_name
        Modality name, used to look up tissue labels.
    seed
        Seed for the per-group random choice (reproducible support set).

    Raises
    ------
    RuntimeError
        If any sample has no tissue label, if no groups are found, or if the
        resulting test set is empty.
    """
    total_samples = len(dataset)
    all_names = dataset.sample_names
    tissue_map = load_tissue_labels(dataset_name, project_root)

    groups: Dict[str, List[int]] = defaultdict(list)
    unlabeled: List[int] = []
    for idx, name in enumerate(all_names):
        tissue = tissue_map.get(name)
        if tissue:
            groups[tissue].append(idx)
        else:
            unlabeled.append(idx)

    if unlabeled:
        missing = [all_names[i] for i in unlabeled]
        raise RuntimeError(
            f"Data integrity error: {len(unlabeled)} sample(s) in dataset "
            f"'{dataset_name}' have no tissue label in configs/tissue_labels.json.\n"
            f"Missing: {missing[:10]}" + (" ..." if len(unlabeled) > 10 else "") + "\n"
            "Run `python scripts/utils/validate_tissue_labels.py` to diagnose "
            "and update configs/tissue_labels.json to achieve full coverage."
        )

    if not groups:
        raise RuntimeError(
            f"No tissue groups found for dataset '{dataset_name}'. "
            "Check configs/tissue_labels.json."
        )

    rng = random.Random(seed)
    support_idx: List[int] = []
    support_tissues: List[str] = []
    for tissue in sorted(groups.keys()):
        chosen = rng.choice(groups[tissue])
        support_idx.append(chosen)
        support_tissues.append(tissue)
        logger.info(
            f"    {tissue}: chose idx={chosen}  name={all_names[chosen]}  "
            f"(pool size={len(groups[tissue])})"
        )

    excluded = set(support_idx)
    test_idx = [i for i in range(total_samples) if i not in excluded]

    if len(test_idx) == 0:
        raise RuntimeError(
            f"Test set is empty for '{dataset_name}' "
            f"(total={total_samples}, support={len(support_idx)}). "
            "Dataset too small for tissue-stratified fine-tuning."
        )

    logger.info(
        f"  Dataset: {dataset_name}  total={total_samples}  "
        f"tissue_groups={len(groups)}  support={len(support_idx)}  "
        f"test={len(test_idx)}  groups={sorted(groups.keys())}"
    )

    return TissueSplit(
        support_idx=support_idx,
        support_names=[all_names[i] for i in support_idx],
        support_tissues=support_tissues,
        test_idx=test_idx,
        groups=dict(groups),
        total_samples=total_samples,
    )
