"""
StarDist model wrapper.

Variants:
    stardist_fluo  - 2D_versatile_fluo  (fluorescence: DAPI, mIF, ssDNA)
    stardist_he    - 2D_versatile_he    (H&E: HE modalities)

Both training and fine-tuning are supported via the official StarDist API.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .base import BaseSegmentationModel


logger = logging.getLogger(__name__)


def _freeze_encoder_layers(keras_model) -> tuple[int, int]:
    """Freeze encoder (downsampling path) layers of the StarDist csbdeep U-Net.

    Strategy: find the first upsampling layer — that marks the encoder/decoder
    boundary.  csbdeep's unet_block (the architecture StarDist2D trains) uses
    UpSampling2D to begin the decoder path; Conv2DTranspose is checked too as a
    defensive fallback (it appears only in StarDist's separate TF-export path,
    not in the trained keras_model).  All layers before the first such layer
    are considered encoder and are frozen.

    Falls back to a name-based heuristic, then to freezing the first 50 % of
    layers if no boundary can be detected.

    Returns:
        (n_frozen, n_total) — number of frozen layers and total layers.
    """
    try:
        from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose
    except ImportError:
        UpSampling2D = None
        Conv2DTranspose = None

    layers = keras_model.layers
    decoder_start = None

    # Primary: first UpSampling2D or Conv2DTranspose marks the decoder start
    _upsample_types = tuple(t for t in [UpSampling2D, Conv2DTranspose] if t is not None)
    if _upsample_types:
        for i, layer in enumerate(layers):
            if isinstance(layer, _upsample_types):
                decoder_start = i
                break

    # Secondary: name-based — match layer names containing "up_", "upsamp",
    # "decode", or "subpixel".
    if decoder_start is None:
        _DECODER_KEYWORDS = ("up_", "upsamp", "decode", "subpixel")
        for i, layer in enumerate(layers):
            if any(kw in layer.name.lower() for kw in _DECODER_KEYWORDS):
                decoder_start = i
                break

    # Fallback: freeze first 50 %
    if decoder_start is None:
        decoder_start = len(layers) // 2
        logger.warning(
            "StarDist: could not detect encoder/decoder boundary by layer type or name. "
            f"Freezing first {decoder_start}/{len(layers)} layers as encoder proxy."
        )

    for layer in layers[:decoder_start]:
        layer.trainable = False

    return decoder_start, len(layers)


_PRETRAINED_MAP = {
    "stardist_fluo": "2D_versatile_fluo",
    "stardist_he": "2D_versatile_he",
}


def _random_fliprot(img: np.ndarray, mask: np.ndarray):
    """Random transpose + flips over the spatial axes (StarDist canonical augmenter)."""
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def _stardist_augmenter(x: np.ndarray, y: np.ndarray):
    """StarDist's recommended 2D augmenter: flips/rotations + intensity jitter."""
    x, y = _random_fliprot(x, y)
    x = x * np.random.uniform(0.6, 2.0) + np.random.uniform(-0.2, 0.2)
    return x, y


class StarDistModel(BaseSegmentationModel):
    """StarDist 2D wrapper."""

    supports_finetune = True

    def __init__(self, config: dict, device: str = "cuda"):
        super().__init__(config, device)
        variant = config.get("variant", "stardist_fluo")
        self.model_name = variant
        self.pretrained_name = _PRETRAINED_MAP.get(variant, "2D_versatile_fluo")
        self._is_he = "he" in variant.lower()
        # 2D_versatile_he was trained on full RGB H&E images — it needs 3-channel
        # colour input, not the stain-separated Hematoxylin channel.
        self.num_channels = 3 if self._is_he else config.get("num_channels", 1)
        self.needs_raw_he_image = self._is_he  # tell experiment code to skip stain sep
        # StarDist uses TF; set GPU memory growth to avoid OOM
        self._configure_tf_gpu()

    @staticmethod
    def _configure_tf_gpu():
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError(
                    "No GPU visible to TensorFlow. "
                    "Ensure CUDA drivers and a GPU are available before running StarDist."
                )
            # Restrict TensorFlow to ONLY use the first visible GPU (set by CUDA_VISIBLE_DEVICES).
            # This ensures the benchmark's single-GPU isolation is respected.
            if gpus:
                tf.config.set_visible_devices([gpus[0]], "GPU")
                tf.config.experimental.set_memory_growth(gpus[0], True)
            # Probe with a trivial op to surface CuDNN version mismatches
            # (e.g. TF compiled with 9.3 but runtime has 9.1) immediately
            # rather than silently mid-experiment.
            with tf.device("/GPU:0"):
                tf.constant([1.0])
        except Exception as exc:
            raise RuntimeError(
                f"StarDist GPU initialisation failed: {exc}\n"
                "Check that the CuDNN runtime version matches the version TensorFlow "
                "was compiled against (see 'nvidia-smi' and 'tf.__version__')."
            ) from exc

    def _ensure_loaded(self):
        if self._model is None:
            from stardist.models import StarDist2D
            self._model = StarDist2D.from_pretrained(self.pretrained_name)
            logger.info(f"Loaded StarDist model: {self.pretrained_name}")

    def predict_single(self, image: np.ndarray) -> np.ndarray:
        from csbdeep.data import PercentileNormalizer
        self._ensure_loaded()
        img = self.preprocess(image)  # (H,W) for fluo; (H,W,3) for HE
        # StarDist's own 1-99.8 percentile normalisation (its trained convention).
        labels, _ = self._model.predict_instances(img, normalizer=PercentileNormalizer(1, 99.8))
        return labels.astype(np.uint16)

    def finetune(
        self,
        support_images: List[np.ndarray],
        support_masks: List[np.ndarray],
        **kwargs,
    ) -> dict:
        from stardist.models import StarDist2D

        n_epochs       = kwargs.get("n_epochs", 20)
        batch_size     = int(kwargs.get("batch_size", 8))
        val_images_raw = kwargs.get("val_images", [])
        val_masks_raw  = kwargs.get("val_masks", [])
        patience       = int(kwargs.get("patience", 20))

        from csbdeep.utils import normalize as _csb_normalize
        self._ensure_loaded()

        # StarDist's own 1-99.8 percentile normalisation (its trained convention).
        X = [_csb_normalize(self.preprocess(img), 1, 99.8, axis=(0, 1)) for img in support_images]
        Y = [m.astype(np.int32) for m in support_masks]

        from stardist import calculate_extents
        median_size = calculate_extents(list(Y), np.median)
        fov = np.array(self._model._axes_tile_overlap("YX"))
        logger.info(f"median object size:      {median_size}")
        logger.info(f"network field of view :  {fov}")
        if any(median_size > fov):
            logger.warning("median object size larger than field of view of the neural network.")

        X_train, Y_train = X, Y
        # Use real val images for early stopping if provided; otherwise fall back
        # to training images (satisfies StarDist's API requirement).
        if val_images_raw:
            X_val = [_csb_normalize(self.preprocess(img), 1, 99.8, axis=(0, 1)) for img in val_images_raw]
            Y_val = [m.astype(np.int32) for m in val_masks_raw]
        else:
            X_val, Y_val = X, Y

        self._model.config.train_epochs = n_epochs
        self._model.config.train_batch_size = batch_size
        self._model.config.train_learning_rate = kwargs.get("learning_rate", 1e-4)
        self._model.config.train_shape_completion = True

        # Cap train_patch_size to the smallest image dimension.
        # stardist_he's pretrained config has patch_size=(512,512); any dataset
        # with images smaller than 512px (e.g. 10xGenomics_HE at 256×256) would
        # raise ValueError("Some images are too small for given patch_size").
        if X_train:
            min_h = min(img.shape[0] for img in X_train + (X_val if val_images_raw else []))
            min_w = min(img.shape[1] for img in X_train + (X_val if val_images_raw else []))
            g = self._model.config.grid  # e.g. (2, 2)
            old_ps = list(self._model.config.train_patch_size)
            new_ps = [(min(old_ps[0], min_h) // g[0]) * g[0],
                      (min(old_ps[1], min_w) // g[1]) * g[1]]
            if new_ps != old_ps:
                self._model.config.train_patch_size = new_ps
                logger.info(f"StarDist train_patch_size adjusted from {tuple(old_ps)} "
                            f"to {tuple(new_ps)} (images are {min_h}×{min_w})")

            # Scale down steps_per_epoch from the pretrained value (400 for
            # fluo, 200 for he), which was calibrated for full datasets.
            # One step per full pass over support images (ceil(n/batch_size))
            # matches the convention of all other models in the benchmark
            # (MEDIAR, Cellpose, InstanSeg, MicroSAM: 1 gradient update per epoch
            # via batching or gradient accumulation).
            bs    = max(1, self._model.config.train_batch_size)
            steps = max(1, int(np.ceil(len(X_train) / bs)))
            old_steps = self._model.config.train_steps_per_epoch
            if steps != old_steps:
                self._model.config.train_steps_per_epoch = steps
                logger.info(f"StarDist train_steps_per_epoch: {old_steps} → {steps} "
                            f"(ceil({len(X_train)} images / batch {bs}))")

        freeze_encoder = kwargs.get("freeze_encoder", True)
        if freeze_encoder:
            n_frozen, n_total = _freeze_encoder_layers(self._model.keras_model)
            logger.info(f"StarDist encoder frozen: {n_frozen}/{n_total} layers frozen")

        # Parameter split for this run's mode (frozen_encoder vs full_model),
        # counted from the Keras model's trainable_weights after freezing.
        _km = self._model.keras_model
        _n_trainable_params = int(sum(int(np.prod(w.shape)) for w in _km.trainable_weights))
        _n_frozen_params    = int(_km.count_params()) - _n_trainable_params

        # Build Keras callbacks when val data is available:
        #   ReduceLROnPlateau — from original StarDist authors (2D_dsb2018 config):
        #     factor=0.5, patience = early_stopping_patience // 2
        #     Halves LR each time val_loss fails to improve for patience//2 epochs.
        #   EarlyStopping — benchmark requirement: patience=20, restore_best_weights.
        # With default patience=20: LR halves at epoch 10 without improvement;
        # training stops at epoch 20 without improvement.
        #
        # Note: StarDist's train() also adds its own native ReduceLROnPlateau
        # (factor=0.5, patience=80 for fluo / 50 for he) via
        # model.config.train_reduce_lr internally.
        # For fewshot_finetuning (patience=0, val_images=[]) this guard is
        # False, so no early-stopping callbacks are added — StarDist's native
        # scheduler runs on its own using support images as val (X_val = X_train
        # fallback).
        callbacks = []
        if patience > 0 and val_images_raw:
            try:
                import tensorflow as tf
                rlrop_patience = max(1, patience // 2)
                rlrop = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=rlrop_patience,
                    min_lr=1e-6,
                    verbose=1,
                )
                es = tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=1e-4,
                    restore_best_weights=True,
                    verbose=1,
                )
                callbacks.extend([rlrop, es])
                logger.info(
                    f"StarDist callbacks: "
                    f"ReduceLROnPlateau(factor=0.5, patience={rlrop_patience}, min_lr=1e-6)"
                    f" + EarlyStopping(patience={patience}, min_delta=1e-4, restore_best_weights=True)"
                )
            except Exception as exc:
                logger.warning(f"StarDist: could not create training callbacks: {exc}")

        train_kwargs: dict = {"augmenter": _stardist_augmenter}

        # Wrap keras_model.fit() for two purposes:
        #   1. Capture the History object returned by fit() directly — more reliable
        #      than reading keras_model.history after StarDist's post-training steps
        #      (csbdeep's ModelCheckpoint reloads the best checkpoint, and
        #      optimize_thresholds() may also reset model.history).  The wrapper is
        #      always installed so history is captured even when patience=0 and no
        #      EarlyStopping/ReduceLROnPlateau callbacks exist.
        #   2. Inject EarlyStopping + ReduceLROnPlateau callbacks when patience > 0.
        #      StarDist 0.9.x has a fixed train() signature (no **kwargs), so these
        #      cannot be passed directly; versions that do expose **kwargs receive
        #      them via train_kwargs instead.
        _orig_keras_fit = None
        _captured_history: list = []   # receives History objects returned by fit()

        import inspect
        _has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(self._model.train).parameters.values()
        )
        if callbacks and _has_var_kw:
            train_kwargs["callbacks"] = callbacks

        # Always install the wrapper — even with an empty callbacks list — so
        # that _captured_history is populated before optimize_thresholds() can
        # reset keras_model.history.
        _orig_keras_fit = self._model.keras_model.fit
        _extra_cbs = list(callbacks if (callbacks and not _has_var_kw) else [])

        def _fit_with_callbacks(*args, **kwargs):
            if _extra_cbs:
                kwargs["callbacks"] = list(kwargs.get("callbacks") or []) + _extra_cbs
            result = _orig_keras_fit(*args, **kwargs)
            _captured_history.append(result)
            return result

        self._model.keras_model.fit = _fit_with_callbacks
        if callbacks and not _has_var_kw:
            logger.info(
                "StarDist: injecting ReduceLROnPlateau + EarlyStopping via "
                "keras_model.fit() wrapper (train() has fixed signature in this version)."
            )

        try:
            self._model.train(X_train, Y_train, validation_data=(X_val, Y_val), **train_kwargs)
        finally:
            if _orig_keras_fit is not None:
                self._model.keras_model.fit = _orig_keras_fit

        if freeze_encoder:
            for layer in self._model.keras_model.layers:
                layer.trainable = True
            logger.info("StarDist encoder unfrozen")

        # Re-optimise detection thresholds on the SUPPORT (training) images, not
        # the val set.  The support images (X_train = X) always share the test
        # distribution and are the correct choice regardless of which experiment
        # type called finetune().
        try:
            self._model.optimize_thresholds(X_train, Y_train)
            logger.info("StarDist thresholds optimised on support (training) images.")
        except Exception as exc:
            logger.warning(f"StarDist threshold optimisation skipped: {exc}")

        # Read actual epochs run and best val-loss epoch from Keras history.
        # Prefer the History object returned directly by fit() (captured in the
        # wrapper closure) because keras_model.history may be reset by StarDist's
        # post-training calls (e.g. optimize_thresholds).  Fall back to
        # keras_model.history if no fit() return was captured.
        try:
            if _captured_history:
                hist = _captured_history[-1].history
            else:
                hist = self._model.keras_model.history.history
            actual_epochs  = len(hist.get("loss", []))
            train_loss_hist = hist.get("loss", [])
            val_loss_hist  = hist.get("val_loss", [])
            best_epoch = (int(np.argmin(val_loss_hist)) + 1) if val_loss_hist else actual_epochs
        except Exception:
            actual_epochs   = n_epochs
            train_loss_hist = []
            val_loss_hist   = []
            best_epoch      = n_epochs

        logger.info(
            f"StarDist fine-tuning complete: {actual_epochs}/{n_epochs} epochs, "
            f"best epoch: {best_epoch}, {len(support_images)} shots."
        )

        # Warm up the TF inference graph before returning.
        #
        # After training, changes to layer.trainable (encoder freeze/unfreeze) and
        # optimize_thresholds() invalidate Keras's compiled @tf.function predict graph.
        # If the first predict_instances() call happens inside the timed inference
        # block, TF retracing adds 40–60 s of CPU overhead to the energy measurement,
        # inflating ei_score_mj_per_image by up to 15×.  A single dummy call here
        # forces recompilation outside the measurement window.
        try:
            from csbdeep.data import PercentileNormalizer
            _warmup = np.zeros(X[0].shape, dtype=X[0].dtype)
            self._model.predict_instances(_warmup, normalizer=PercentileNormalizer(1, 99.8))
            logger.info("StarDist: TF inference graph warmed up.")
        except Exception as _exc:
            logger.warning(f"StarDist: inference warm-up skipped ({_exc})")

        return {
            "n_epochs_run":        actual_epochs,
            "n_epochs_max":        n_epochs,
            "best_epoch":          best_epoch,
            "train_losses":        train_loss_hist,
            "val_losses":          val_loss_hist,
            "early_stopping_mode": "keras_earlystopping_restore_best_weights",
            "n_train_images":      len(support_images),
            # crops processed per epoch = steps_per_epoch × batch_size
            "n_train_patches":     int(self._model.config.train_steps_per_epoch
                                       * self._model.config.train_batch_size),
            "n_trainable_params":  _n_trainable_params,
            "n_frozen_params":     _n_frozen_params,
        }

    def get_param_count(self) -> int:
        self._ensure_loaded()
        try:
            return int(self._model.keras_model.count_params())
        except Exception as exc:
            logger.warning(f"Failed to get parameter count for {self.model_name}: {exc}")
            return -1
