"""
Cellpose 3.x model wrapper (cyto2, cyto3).  For Cellpose 4.x see cellpose_sam_model.py.

Cellpose normalises (normalize99) and augments (random_rotate_and_resize)
internally.  finetune() runs a train_seg-style loop with patience-based early
stopping and best-weight restore (train_seg has no native best-checkpoint
mechanism) and stores the learned diameter for subsequent eval() calls.
eval() uses augment=False (no TTA).
"""

from __future__ import annotations

import copy
import logging
from typing import List

import numpy as np

from .base import BaseSegmentationModel

logger = logging.getLogger(__name__)


def _freeze_cellpose_encoder(net) -> tuple[int, int]:
    """Freeze the encoder (downsampling path) of the Cellpose U-Net.

    CPnet stores the encoder blocks in ``self.downsample``.  Falls back to
    freezing any submodule whose name starts with ``"down"`` if the attribute
    is absent; last resort: freezes the first half of all parameter tensors.

    Returns:
        (n_frozen_params, n_total_params)
    """
    total = sum(p.numel() for p in net.parameters())
    frozen = 0

    if hasattr(net, "downsample"):
        for param in net.downsample.parameters():
            param.requires_grad = False
            frozen += param.numel()
    else:
        for name, module in net.named_modules():
            if name.lower().startswith("down"):
                for param in module.parameters():
                    param.requires_grad = False
                    frozen += param.numel()
        if frozen == 0:
            params = list(net.parameters())
            half = len(params) // 2
            for param in params[:half]:
                param.requires_grad = False
                frozen += param.numel()
            logger.warning(
                "Cellpose: could not detect encoder by attribute or name. "
                f"Frozen first {half}/{len(params)} parameter tensors as proxy."
            )

    return frozen, total


class CellposeModel(BaseSegmentationModel):
    """Cellpose wrapper (cyto2, cyto3 variants)."""

    supports_finetune = True

    _VARIANT_TO_MODEL_TYPE = {
        "cellpose_cyto2": "cyto2",
        "cellpose_cyto3": "cyto3",
    }

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        # Dataset-level minimum instance size (configs/datasets/<name>.yaml).
        # Cellpose drops masks with `npix < min_size`, which is exactly the
        # benchmark's definition, so the value passes through unchanged.
        self.min_size = int(config.get("min_size", 1))
        variant = config.get("variant", "cellpose_cyto3")
        self.model_name = variant
        self.model_type = self._VARIANT_TO_MODEL_TYPE.get(variant, "cyto3")
        self.diameter = config.get("diameter", None)  # updated to learned value after training
        self.num_channels = config.get("num_channels", 1)
        self._use_gpu = device.startswith("cuda")

    def _get_channels(self, image: np.ndarray) -> List[int]:
        return [0, 0]  # always grayscale mode

    def _ensure_loaded(self):
        if self._model is None:
            from cellpose import models as cp_models
            self._model = cp_models.CellposeModel(
                model_type=self.model_type,
                gpu=self._use_gpu,
            )
            logger.info(f"Loaded Cellpose model: {self.model_type}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # NOTE — Cellpose thresholds are deliberately NOT calibrated.
    # flow_threshold (0.4) and cellprob_threshold (0.0) stay at the library
    # defaults, which is what the reference does: Cellpose ships no threshold
    # optimisation API of any kind, and docs/settings.rst presents both as knobs
    # the USER nudges by eye ("Increase this threshold if cellpose is not
    # returning as many ROIs as you'd expect").  Contrast StarDist, which ships
    # optimize_thresholds(), and micro-sam, which ships
    # run_instance_segmentation_grid_search — both DO provide one, so their
    # wrappers calibrate.  Fitting these on the support set would be a
    # benchmark-level intervention beyond the reference.

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        img = self.preprocess(image)
        masks, _, _ = self._model.eval(
            img,
            diameter=self.diameter,
            channels=self._get_channels(img),
            normalize=True,
            augment=False,
            min_size=self.min_size,
            bsize=256,
            tile_overlap=0.5,
            do_3D=False,
        )
        return masks.astype(np.uint16)

    def predict_batch(self, images: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        self._ensure_loaded()
        results = []
        for i in range(0, len(images), batch_size):
            chunk = [self.preprocess(img) for img in images[i:i + batch_size]]
            batch_results, _, _ = self._model.eval(
                chunk,
                diameter=self.diameter,
                channels=[0, 0],
                normalize=True,
                augment=False,
                min_size=self.min_size,
                bsize=256,
                tile_overlap=0.5,
                do_3D=False,
            )
            if isinstance(batch_results, np.ndarray) and batch_results.ndim == 2:
                batch_results = [batch_results]
            results.extend([r.astype(np.uint16) for r in batch_results])
        return results

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        """Fine-tune Cellpose with a custom training loop that mirrors train_seg()
        exactly but adds patience-based early stopping with true in-memory
        best-weight restoration.

        Cellpose's train_seg() has no native best-checkpoint save — it writes
        only the final-epoch model to disk and returns that path.  This loop
        instead saves the best state_dict in RAM (via _EarlyStopper) and
        restores it unconditionally at the end, whether or not early stopping
        triggered.

        LR schedule, eval cadence (epoch 5 then every 10th), batch construction,
        and loss function are identical to train_seg() so results are comparable.
        """
        import torch
        import cellpose.transforms as cp_transforms
        from cellpose import train as cp_train
        import cellpose.models as cp_models
        from .base import _EarlyStopper

        n_epochs       = int(kwargs.get("n_epochs", 100))
        learning_rate  = float(kwargs.get("learning_rate", 0.005))
        weight_decay   = float(kwargs.get("weight_decay", 1e-4))
        batch_size     = int(kwargs.get("batch_size", 8))
        freeze_encoder = bool(kwargs.get("freeze_encoder", True))
        val_images     = kwargs.get("val_images", [])
        val_masks_kw   = kwargs.get("val_masks", [])
        patience       = int(kwargs.get("patience", 20))
        bsize          = 256   # tile size — matches predict_single/batch

        self._ensure_loaded()
        net    = self._model.net
        device = net.device

        train_imgs    = [self.preprocess(img) for img in support_images]
        train_masks   = [m.astype(np.uint16) for m in support_masks]
        val_imgs_proc = [self.preprocess(img) for img in val_images] if val_images else None
        logger.info(f"Cellpose finetune: {len(train_imgs)} support images")

        if freeze_encoder:
            n_frozen, n_total = _freeze_cellpose_encoder(net)
            logger.info(f"Cellpose encoder frozen: {n_frozen:,}/{n_total:,} params frozen")

        try:
            # Cellpose normalises (normalize99) and augments (random_rotate_and_resize)
            # internally; _process_train_test converts masks to flow targets and
            # estimates per-image diameters.
            normalize_params = cp_models.normalize_default
            out = cp_train._process_train_test(
                train_data=train_imgs,
                train_labels=train_masks,
                train_files=None, train_labels_files=None, train_probs=None,
                test_data=val_imgs_proc,
                test_labels=list(val_masks_kw) if val_imgs_proc else None,
                test_files=None, test_labels_files=None, test_probs=None,
                load_files=False,
                min_train_masks=1,
                compute_flows=False,
                channels=[0, 0],
                channel_axis=None,
                rgb=False,
                normalize_params=normalize_params,
                device=device,
            )
            (train_data_cp, train_labels_cp, _, _, train_probs_cp, diam_train,
             test_data_cp,  test_labels_cp,  _, _, test_probs_cp,  diam_test,
             normed) = out

            # If preprocessing normalised the data, skip per-batch re-normalisation
            kwargs_batch = {} if normed else {
                "normalize_params": normalize_params,
                "channels": [0, 0],
                "channel_axis": None,
                "rgb": False,
            }

            net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

            nimg           = len(train_data_cp)
            nimg_per_epoch = max(nimg, batch_size)
            has_val        = test_data_cp is not None and len(test_data_cp) > 0
            nimg_test      = len(test_data_cp) if has_val else 0

            # ── Pre-compute val tensors (done once, no per-epoch randomness) ─
            # The gamma-corrected val images are evaluated as-is: a single fixed
            # resize/crop to bsize×bsize (seed=42, scale_range=0, no flip) so
            # val loss reflects only model weight changes, not augmentation noise.
            val_batch_imgi: list = []
            val_batch_lbl:  list = []
            if has_val:
                _rng_state = np.random.get_state()
                np.random.seed(42)
                for _ibv in range(0, nimg_test, batch_size):
                    _inds_v  = list(range(_ibv, min(_ibv + batch_size, nimg_test)))
                    _imgs_v, _lbls_v = cp_train._get_batch(
                        _inds_v,
                        data=test_data_cp, labels=test_labels_cp,
                        files=None, labels_files=None,
                        **kwargs_batch,
                    )
                    _diams_v = np.array([diam_test[i] for i in _inds_v])
                    _rsc_v   = _diams_v / net.diam_mean.item()
                    _imgi_v, _lbl_v = cp_transforms.random_rotate_and_resize(
                        _imgs_v, Y=_lbls_v, rescale=_rsc_v,
                        scale_range=0.0, xy=(bsize, bsize), do_flip=False,
                    )[:2]
                    val_batch_imgi.append(_imgi_v)
                    val_batch_lbl.append(_lbl_v)
                np.random.set_state(_rng_state)

            # ── LR schedule ────────────────────────────────────────────────
            # Identical to train_seg() lines 422-432:
            #   warm-up (0 → lr over 10 epochs) then constant, with a
            #   step-decay tail for n_epochs > 100.
            LR = np.linspace(0, learning_rate, 10)
            LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
            if n_epochs > 300:
                LR = LR[:-100]
                for _ in range(10):
                    LR = np.append(LR, LR[-1] / 2 * np.ones(10))
            elif n_epochs > 100:
                LR = LR[:-50]
                for _ in range(10):
                    LR = np.append(LR, LR[-1] / 2 * np.ones(5))

            optimizer = torch.optim.AdamW(
                [p for p in net.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # Parameter split for this run's mode (frozen_encoder vs full_model),
            # captured while the freeze flags are active (they are restored before
            # this method returns, so it must be counted here).
            _n_trainable_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
            _n_frozen_params    = int(sum(p.numel() for p in net.parameters() if not p.requires_grad))

            # ── Early-stopping state ───────────────────────────────────────
            # Val is checked every epoch; patience is in actual epochs directly.
            stopper      = _EarlyStopper(patience)
            best_val     = float("inf")
            best_epoch   = n_epochs   # updated each time val loss improves
            best_sd: dict | None = None  # net.state_dict() snapshot at best_val
            train_losses: list = []
            val_losses:   list = []
            actual_epochs = n_epochs

            # ── Training loop ──────────────────────────────────────────────
            for iepoch in range(n_epochs):
                # Use a local RandomState so the global np.random state is not
                # disturbed between shot levels or between train/val phases.
                # Mirrors train_seg's internal per-epoch seed without polluting
                # the process-level RNG.
                _epoch_rng = np.random.RandomState(iepoch)

                rperm = (
                    _epoch_rng.choice(nimg, size=nimg_per_epoch, p=train_probs_cp)
                    if nimg != nimg_per_epoch
                    else _epoch_rng.permutation(nimg)
                )

                for param_group in optimizer.param_groups:
                    param_group["lr"] = LR[iepoch]

                net.train()
                epoch_loss = 0.0
                for k in range(0, nimg_per_epoch, batch_size):
                    inds = rperm[k:min(k + batch_size, nimg_per_epoch)]
                    imgs, lbls = cp_train._get_batch(
                        inds,
                        data=train_data_cp, labels=train_labels_cp,
                        files=None, labels_files=None,
                        **kwargs_batch,
                    )
                    diams = np.array([diam_train[i] for i in inds])
                    rsc   = diams / net.diam_mean.item()
                    imgi, lbl = cp_transforms.random_rotate_and_resize(
                        imgs, Y=lbls, rescale=rsc, scale_range=0.5,
                        xy=(bsize, bsize),
                    )[:2]
                    X    = torch.from_numpy(imgi).to(device)
                    y    = net(X)[0]
                    loss = cp_train._loss_fn_seg(lbl, y, device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(imgi)
                train_losses.append(epoch_loss / nimg_per_epoch)

                # ── Validation (every epoch, using pre-computed tensors) ──────
                ep    = iepoch + 1
                lavgt = 0.0
                if has_val:
                    net.eval()
                    for _imgi_v, _lbl_v in zip(val_batch_imgi, val_batch_lbl):
                        with torch.no_grad():
                            X    = torch.from_numpy(_imgi_v).to(device)
                            y    = net(X)[0]
                            lavgt += cp_train._loss_fn_seg(_lbl_v, y, device).item() * _imgi_v.shape[0]
                    lavgt /= nimg_test
                    val_losses.append(lavgt)

                    if lavgt < best_val:
                        best_val   = lavgt
                        best_epoch = ep
                        best_sd    = copy.deepcopy(net.state_dict())

                    if stopper.step(lavgt):
                        actual_epochs = ep
                        logger.info(
                            f"  Cellpose early stop at epoch {ep}/{n_epochs}  "
                            f"best_val={best_val:.4f}  best_epoch={best_epoch}"
                        )
                        break

                if ep % 10 == 0 or ep == 1:
                    logger.info(
                        f"  epoch {ep:>4}/{n_epochs}  "
                        f"train={train_losses[-1]:.4f}  val={lavgt:.4f}  "
                        f"LR={LR[iepoch]:.6f}"
                    )

            else:
                actual_epochs = n_epochs

            # Restore best val-loss weights (tracked by best_sd, updated every epoch).
            if has_val and best_sd is not None:
                net.load_state_dict(best_sd)
                logger.info(f"  Restored best weights from epoch {best_epoch}")

        finally:
            # Restore requires_grad state for encoder parameters (only if they were frozen)
            if freeze_encoder:
                if hasattr(net, "downsample"):
                    for param in net.downsample.parameters():
                        param.requires_grad = True
                else:
                    for name, module in net.named_modules():
                        if name.lower().startswith("down"):
                            for param in module.parameters():
                                param.requires_grad = True
                logger.info("Cellpose encoder parameters unfrozen")

        self.diameter = float(net.diam_labels.item())
        logger.info(
            f"Cellpose fine-tuning complete.  "
            f"Best epoch: {best_epoch}/{n_epochs}  "
            f"({len(val_losses)} val evaluations, {actual_epochs} epochs run).  "
            f"Learned diameter: {self.diameter:.1f} px"
        )

        return {
            "n_epochs_run":        actual_epochs,
            "n_epochs_max":        n_epochs,
            "best_epoch":          best_epoch,
            "early_stopping_mode": "patience_restore_best_weights",
            "n_train_images":      len(support_images),
            "n_train_patches":     nimg_per_epoch,   # crops processed per epoch (max(n_images, batch_size))
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
            "train_losses":        train_losses,
            "val_losses":          val_losses,
            "learned_diameter":    self.diameter,
        }

    def get_param_count(self) -> int:
        self._ensure_loaded()
        try:
            return sum(p.numel() for p in self._model.net.parameters())
        except Exception as exc:
            logger.warning(f"Failed to get parameter count for {self.model_name}: {exc}")
            return -1
