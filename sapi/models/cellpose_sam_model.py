"""
CellposeSAM model wrapper (Cellpose 4.0+, ViT-L/SAM 'cpsam' backbone).

Cellpose normalises (normalize99) and augments internally.  eval() uses
resample=True (ViT-L quality) and augment=False (no TTA).  finetune() runs a
train_seg-style loop with patience-based early stopping and best-weight restore.

Precision — bfloat16 inference, float32 fine-tuning, matching Cellpose 4.x:
  - Inference: parameters are loaded as bfloat16 (use_bfloat16=True) and eval()
    casts the input to net.dtype, so inference is bfloat16 throughout.  This is
    Cellpose's default and is 5x faster than float32 (~195 ms/image vs
    ~915 ms/image on V100).
  - Fine-tuning: the net is converted to float32 via _set_net_dtype() and the
    training loop uses NO autocast.  This reproduces train_seg: it wraps its
    forward in autocast(dtype=net.dtype), but by then net.dtype is float32, which
    is not a supported CUDA autocast dtype, so PyTorch disables the context and
    training runs in plain float32.  Do not add autocast here to mirror the
    bfloat16 inference path — the asymmetry is what Cellpose does.
  - Every dtype switch goes through _set_net_dtype(), which sets net.dtype AND
    casts the weights explicitly.  A bare `net.dtype = torch.float32` is NOT
    enough: an installed 4.x was seen to leave the weights in bfloat16 while
    _dtype reported float32, which crashes the first conv with "Input type
    (float) and bias type (c10::BFloat16) should be the same".
  - After training, the net is restored so inference returns to bfloat16.

Requires cellpose >= 4.0.
"""

from __future__ import annotations

import copy
import logging
from typing import List

import numpy as np

from .base import BaseSegmentationModel
from .cellpose_model import _freeze_cellpose_encoder

logger = logging.getLogger(__name__)


def _set_net_dtype(net, dtype) -> None:
    """Set ``net.dtype`` AND guarantee the parameters really are *dtype*.

    Both halves are needed and they do different jobs:
      * the attribute keeps ``net.dtype`` consistent, because Cellpose casts the
        input to it (``core.py``: ``_to_device(x, dtype=net.dtype)``);
      * the explicit ``.to()`` is what actually moves the weights.

    Do NOT collapse this back to a bare ``net.dtype = dtype``. Cellpose's dtype
    setter is *supposed* to cast the parameters itself
    (``vit_sam.py``: ``if self._dtype != value: self.to(value)``), but an
    installed 4.x was observed not to, leaving a net whose ``_dtype`` says
    float32 while its weights are still bfloat16. That combination fails at the
    first conv with "Input type (float) and bias type (c10::BFloat16) should be
    the same" — and, worse, was silent while the training forward still ran under
    autocast, because autocast cast the *input* down to bfloat16 to match the
    weights instead. This helper is idempotent: where the setter does cast, the
    ``.to()`` is a no-op.
    """
    net.dtype = dtype
    current = next((p.dtype for p in net.parameters() if p.is_floating_point()), None)
    if current is not None and current != dtype:
        net.to(dtype)
        logger.info(f"CellposeSAM: recast parameters {current} → {dtype}")


def _freeze_cellpose_sam_encoder(net) -> tuple[int, int]:
    """Freeze the ViT image encoder of the CellposeSAM network.

    The Cellpose 4.0 ``Transformer`` class stores the SAM ViT encoder as
    ``net.encoder`` (set in ``vit_sam.py:Transformer.__init__``).  A secondary
    check for ``net.image_encoder`` is kept for forward-compatibility.

    Returns:
        (n_frozen_params, n_total_params)
    """
    total = sum(p.numel() for p in net.parameters())
    frozen = 0

    if hasattr(net, "encoder"):
        for param in net.encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
    elif hasattr(net, "image_encoder"):
        for param in net.image_encoder.parameters():
            param.requires_grad = False
            frozen += param.numel()
    else:
        frozen, total = _freeze_cellpose_encoder(net)

    return frozen, total


class CellposeSAMModel(BaseSegmentationModel):
    """Cellpose 4.x / CellposeSAM wrapper (ViT-L backbone, pretrained cpsam)."""

    supports_finetune = True

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        # Dataset-level minimum instance size (configs/datasets/<name>.yaml).
        # Cellpose drops masks with `npix < min_size`, which is exactly the
        # benchmark's definition, so the value passes through unchanged.
        self.min_size = int(config.get("min_size", 1))
        self.model_name = config.get("variant", "cellpose_sam")
        self.diameter = config.get("diameter", None)  # updated to learned value after training
        self._use_gpu = device.startswith("cuda")

    def _ensure_loaded(self):
        if self._model is None:
            from cellpose import models as cp_models
            self._model = cp_models.CellposeModel(
                pretrained_model="cpsam",
                gpu=self._use_gpu,
                use_bfloat16=True,
            )
            logger.info("Loaded CellposeSAM model: cpsam")

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
        masks, _, _ = self._model.eval(
            image,
            diameter=self.diameter,
            normalize=True,
            augment=False,
            resample=True,
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
            chunk = images[i:i + batch_size]
            batch_results, _, _ = self._model.eval(
                chunk,
                diameter=self.diameter,
                normalize=True,
                augment=False,
                resample=True,
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
        """Fine-tune CellposeSAM with a custom training loop that mirrors
        Cellpose 4.x train_seg() exactly but adds patience-based early
        stopping with true in-memory best-weight restoration.

        Cellpose 4.x API differences from 3.x (all handled here):
          - _process_train_test: no channels= or rgb= params (removed in 4.x)
          - kwargs_batch: channel_axis= instead of channels=
          - net.dtype may be bfloat16 → converted to float32 for training
            (float32 training, no autocast — see the module docstring)
          - labels moved to device before _loss_fn_seg (4.x expects tensor)
          - rescale=True: rsc = diam/diam_mean zooms crops to ~30 px cells
          - LR schedule tail: n_epochs > 99 (vs > 100 in 3.x)
        """
        import torch
        import cellpose.transforms as cp_transforms
        from cellpose import train as cp_train
        import cellpose.models as cp_models
        from .base import _EarlyStopper

        n_epochs       = int(kwargs.get("n_epochs", 100))
        learning_rate  = float(kwargs.get("learning_rate", 1e-5))
        weight_decay   = float(kwargs.get("weight_decay", 0.1))
        batch_size     = int(kwargs.get("batch_size", 8))
        freeze_encoder = bool(kwargs.get("freeze_encoder", True))
        val_images     = kwargs.get("val_images", [])
        val_masks_kw   = kwargs.get("val_masks", [])
        patience       = int(kwargs.get("patience", 20))
        # Gradient accumulation: split each logical batch into micro-batches of
        # this size to cap peak activation memory.  0 (or ≥ batch_size) disables
        # it — the loop then does one forward/backward per batch, byte-identical
        # to the non-accumulated path.  Fine-tuning is fp32 (see module docstring),
        # so full-model ViT-L at batch 8 OOMs on a 32 GB GPU; micro-batching lowers
        # the peak without changing the update (see the training loop below).
        grad_accum_micro_batch = int(kwargs.get("grad_accum_micro_batch", 0))
        bsize          = 256

        self._ensure_loaded()
        net    = self._model.net
        device = net.device

        train_imgs    = list(support_images)
        train_masks   = [m.astype(np.uint16) for m in support_masks]
        val_imgs_proc = list(val_images) if val_images else None
        logger.info(f"CellposeSAM finetune: {len(train_imgs)} support images")
        if 0 < grad_accum_micro_batch < batch_size:
            logger.info(
                f"CellposeSAM: gradient accumulation active — "
                f"micro-batch={grad_accum_micro_batch}, effective batch={batch_size}"
            )

        if freeze_encoder:
            n_frozen, n_total = _freeze_cellpose_sam_encoder(net)
            logger.info(f"CellposeSAM encoder frozen: {n_frozen:,}/{n_total:,} params frozen")

        # Cellpose 4.x trains in float32 even when the loaded net is bfloat16.
        # Store the original dtype and convert the net to float32 (same as
        # train_seg).  _set_net_dtype casts the weights for real rather than
        # trusting the dtype setter to do it — see that helper.  The loop below
        # adds no autocast, which is what train_seg effectively does: it wraps its
        # forward in autocast(dtype=net.dtype), and float32 is not a supported CUDA
        # autocast dtype, so PyTorch disables that context.  Fine-tuning is
        # therefore fp32.  Inference stays bfloat16 (restored at the end).
        original_dtype = getattr(net, "dtype", None)
        if original_dtype == torch.bfloat16:
            _set_net_dtype(net, torch.float32)
            logger.info("CellposeSAM: converted bfloat16 net to float32 for training")

        try:
            # Cellpose 4.x normalises and augments internally; _process_train_test
            # has no channels= or rgb= params.
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
                channel_axis=None,           # 4.x: no channels= or rgb=
                normalize_params=normalize_params,
                device=device,
            )
            (train_data_cp, train_labels_cp, _, _, train_probs_cp, diam_train,
             test_data_cp,  test_labels_cp,  _, _, test_probs_cp,  diam_test,
             normed) = out

            # Cellpose 4.x kwargs_batch uses channel_axis= (no channels= or rgb=)
            kwargs_batch = {} if normed else {
                "normalize_params": normalize_params,
                "channel_axis": None,
            }

            net.diam_labels.data = torch.Tensor([diam_train.mean()]).to(device)

            nimg           = len(train_data_cp)
            nimg_per_epoch = max(nimg, batch_size)
            has_val        = test_data_cp is not None and len(test_data_cp) > 0
            nimg_test      = len(test_data_cp) if has_val else 0

            # ── Pre-compute val tensors (done once, no per-epoch randomness) ─
            # Same design as cellpose_model.py: fixed seed, scale_range=0,
            # no flip → val loss is a pure function of model weights.
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
            # Identical to Cellpose 4.x train_seg (note: > 99, not > 100).
            LR = np.linspace(0, learning_rate, 10)
            LR = np.append(LR, learning_rate * np.ones(max(0, n_epochs - 10)))
            if n_epochs > 300:
                LR = LR[:-100]
                for _ in range(10):
                    LR = np.append(LR, LR[-1] / 2 * np.ones(10))
            elif n_epochs > 99:
                LR = LR[:-50]
                for _ in range(10):
                    LR = np.append(LR, LR[-1] / 2 * np.ones(5))

            optimizer = torch.optim.AdamW(
                [p for p in net.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=weight_decay,
            )

            # Parameter split for this run's mode (frozen_encoder vs full_model),
            # captured while the freeze flags are active (restored before return).
            _n_trainable_params = int(sum(p.numel() for p in net.parameters() if p.requires_grad))
            _n_frozen_params    = int(sum(p.numel() for p in net.parameters() if not p.requires_grad))

            # ── Early-stopping state ───────────────────────────────────────
            # Val is checked every epoch; patience is in actual epochs directly.
            stopper      = _EarlyStopper(patience)
            best_val     = float("inf")
            best_epoch   = n_epochs
            best_sd: dict | None = None  # weights snapshot at best_val
            train_losses: list = []
            val_losses:   list = []
            actual_epochs = n_epochs

            # ── Training loop ──────────────────────────────────────────────
            for iepoch in range(n_epochs):
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
                    # Augmentation above is done once for the whole batch, so the
                    # crops are identical to the non-accumulated path; only the
                    # forward/backward is chunked into micro-batches.  Each
                    # micro-loss is scaled by (micro_n / n) so the accumulated
                    # gradients sum to exactly the batch-mean gradient
                    # (_loss_fn_seg returns the batch mean — confirmed by the
                    # epoch_loss weighting below), and a single optimizer.step()
                    # per logical batch is mathematically identical to batch=n.
                    n_in  = imgi.shape[0]
                    micro = (grad_accum_micro_batch
                             if 0 < grad_accum_micro_batch < n_in else n_in)
                    optimizer.zero_grad()
                    for j in range(0, n_in, micro):
                        xi = imgi[j:j + micro]
                        li = lbl[j:j + micro]
                        X     = torch.from_numpy(xi).to(device)
                        lbl_t = torch.from_numpy(li).to(device)   # 4.x: tensor to device
                        y     = net(X)[0]
                        mloss = cp_train._loss_fn_seg(lbl_t, y, device)
                        (mloss * (xi.shape[0] / n_in)).backward()
                        epoch_loss += mloss.item() * xi.shape[0]
                    optimizer.step()
                train_losses.append(epoch_loss / nimg_per_epoch)

                # ── Validation (every epoch, using pre-computed tensors) ──────
                ep    = iepoch + 1
                lavgt = 0.0
                if has_val:
                    net.eval()
                    for _imgi_v, _lbl_v in zip(val_batch_imgi, val_batch_lbl):
                        with torch.no_grad():
                            X     = torch.from_numpy(_imgi_v).to(device)
                            lbl_t = torch.from_numpy(_lbl_v).to(device)
                            y     = net(X)[0]
                            lavgt += cp_train._loss_fn_seg(lbl_t, y, device).item() * _imgi_v.shape[0]
                    lavgt /= nimg_test
                    val_losses.append(lavgt)

                    if lavgt < best_val:
                        best_val   = lavgt
                        best_epoch = ep
                        best_sd    = copy.deepcopy(net.state_dict())

                    if stopper.step(lavgt):
                        actual_epochs = ep
                        logger.info(
                            f"  CellposeSAM early stop at epoch {ep}/{n_epochs}  "
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

            # Release AdamW first/second-moment buffers and the best-weight snapshot
            # from GPU/CPU memory before the inference phase begins.  Without this,
            # the optimizer states (~2× parameter size for a 304 M-param ViT-L) remain
            # allocated, increasing VRAM pressure and slowing inference by ~65 %.
            del optimizer
            best_sd = None
            torch.cuda.empty_cache()

        finally:
            if freeze_encoder:
                for p in net.parameters():
                    p.requires_grad = True
                logger.info("CellposeSAM encoder unfrozen")
            # Restore net.dtype so eval() runs in bfloat16 again: Cellpose casts the
            # input to net.dtype (core.py: `_to_device(x, dtype=net.dtype)`), so
            # leaving it at float32 would run inference in float32 — a 5× slowdown
            # (~195 ms/image with bfloat16 vs ~915 ms/image with float32 on V100)
            # and a departure from Cellpose's own bfloat16 inference default.
            if original_dtype is not None:
                _set_net_dtype(net, original_dtype)

        self.diameter = float(net.diam_labels.item())
        logger.info(
            f"CellposeSAM fine-tuning complete.  "
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

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_param_count(self) -> int:
        self._ensure_loaded()
        try:
            return sum(p.numel() for p in self._model.net.parameters())
        except Exception as exc:
            logger.warning(f"Failed to get parameter count for {self.model_name}: {exc}")
            return -1
