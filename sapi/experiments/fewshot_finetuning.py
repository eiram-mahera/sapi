"""
Tissue-stratified fewshot fine-tuning experiment (fewshot_finetuning).

Selects exactly ONE random image per tissue group as the support set, giving a
maximally diverse calibration set that mirrors real-world deployment (a user
typically picks one representative patch per tissue type).

Unified execution — both freeze_encoder modes (full_model and frozen_encoder)
run in a single experiment, producing a single result.json containing zero-shot
baseline plus both fine-tuning modes' metrics and efficiency data.

Workflow
--------
  STEP 1: Zero-shot inference on the test set (pretrained model)
  STEP 2: Full-model fine-tuning on the support set (fixed epoch budget)
  STEP 3: Full-model post-fine-tuning inference on the test set
  STEP 4: Frozen-encoder fine-tuning on the support set (identical seed, same support images)
  STEP 5: Frozen-encoder post-fine-tuning inference on the test set
  STEP 6: Compute SAPI_ZS (zero-shot) and SAPI_LC (full_model, frozen_encoder)
          components from all three scenarios' metrics and efficiency data

Key design choices:
  * Unified execution: both modes run in one Python process. After each
    of the three phases (zero-shot, full_model, frozen_encoder) finishes, its
    model instance is deleted and GPU/CPU memory cleared; the next phase then
    builds a brand-new model instance, so every phase starts from an
    identical pretrained checkpoint rather than inheriting in-memory state.
  * No validation image: no held-out signal, early stopping disabled
    (patience=0). Training runs for a fixed number of epochs (config.n_epochs).
  * Fixed epoch budget: same n_epochs for all models, regardless of support-set
    size. Each epoch exposes every support image once plus augmentation, giving
    principled energy-comparable training budget across datasets.
  * Identical seeding: both freeze_encoder modes use config.seed for
    tissue selection, augmentation, and initialization. Modes differ ONLY in whether
    the encoder is frozen, isolating freeze_encoder's effect without confounding.
  * Tissue labels from configs/tissue_labels.json (full 100% coverage across datasets).
  * Energy measurement excludes cooldown: GPU cooldown between modes
    happens outside TimedBlock, so it doesn't inflate Et values. Both modes' energy
    totals reflect only the actual training/inference work, not idle waiting.
  * Et normalised by support set size only: training energy per image
    = total_training_it_energy / n_train_images. NOT divided by n_epochs, so Et
    reflects per-image amortised cost independent of epoch budget, enabling fair
    cross-model comparison at different epoch counts (10 vs 100 epochs).

Support/test split
------------------
  support:  1 image per tissue group, chosen via random.Random(seed).choice()
             (same support set used for both full_model and frozen_encoder)
  test:     all remaining images (not in support)

Result
------
Returns a single ExperimentResult (one per execution, both modes included).
performance_metrics contains zero_shot, full_model, and frozen_encoder.
training_metadata contains efficiency and loss info for full_model and frozen_encoder.
Energy measurements (inference_efficiency, training_efficiency) stored separately.
"""

from __future__ import annotations

import logging
from typing import List

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from .base import BaseExperiment, ExperimentResult
from .tissue_split import tissue_stratified_split

logger = logging.getLogger(__name__)

# Epoch budget for tissue-stratified runs.
# n_epochs is always read from ExperimentConfig.n_epochs, which resolves
# (in priority order): --n_epochs CLI flag → model YAML fewshot_finetuning.n_epochs
# → YAML fewshot.n_epochs → built-in default (100).

# Max batch size during tissue fine-tuning (support sets are tiny).
_TISSUE_MAX_BATCH: int = 8


class FewShotFinetuningExperiment(BaseExperiment):
    """Unified execution: runs both full_model and frozen_encoder fine-tuning in one experiment.

    Tissue groups are read from configs/tissue_labels.json.  One image per
    group is chosen at random (seed=config.seed); all remaining images form the
    test set.  Training uses a fixed epoch budget with no early stopping.

    Both freeze_encoder modes (full_model and frozen_encoder) are executed in
    a single experiment, storing all results (zero-shot, full_model FT, frozen_encoder FT)
    in one result.json under <dataset>/fewshot_finetuning/<timestamp>/.
    """

    experiment_type = "fewshot_finetuning"
    measures_energy = True
    yaml_section_chain = ("fewshot_finetuning", "fewshot")

    def execute(self) -> List[ExperimentResult]:
        self.config.patience = 0  # fewshot_finetuning always trains for a fixed epoch budget
        hardware_info = self._collect_hardware_info()

        # The tissue split + image loading run before the energy-profiled section
        # (which has its own try/except in _run_tissue_split).  Guard them here so
        # a setup failure — missing tissue label, empty test set, data-load error —
        # is recorded as a failed result instead of crashing run_experiment.py
        # with no result written and no structured status.
        try:
            dataset = self._load_all_data()

            split = tissue_stratified_split(
                dataset, self.config.dataset_name, self.config.seed
            )

            # ── Load images ────────────────────────────────────────────────────
            # needs_raw_he_image must be read from the model INSTANCE, not the
            # class: it is variant-dependent for models whose variants share one
            # wrapper class (stardist_fluo needs the hematoxylin channel,
            # stardist_he needs raw RGB — both are StarDistModel, and the class
            # attribute cannot describe both).  Building the model here costs
            # nothing extra — every wrapper lazy-loads its checkpoint on first
            # use — and the instance is reused for training below.
            model = self._build_model()
            raw_he = getattr(model, "needs_raw_he_image", False)
            all_images, all_masks = dataset.get_images_and_masks(raw_he=raw_he)

            support_images = [all_images[i] for i in split.support_idx]
            support_masks  = [all_masks[i]  for i in split.support_idx]
            test_images    = [all_images[i] for i in split.test_idx]
            test_masks     = [all_masks[i]  for i in split.test_idx]
            test_names     = [dataset.sample_names[i] for i in split.test_idx]
        except Exception as exc:
            logger.exception(f"  Tissue fewshot setup failed: {exc}")
            result = self._make_result_base(hardware_info)
            result.mark_failed(exc, "setup")
            return [result]

        # ── Seed and run unified execution (both modes) ──────────────────────
        self._seed_everything(self.config.seed)

        result = self._run_unified_fewshot(
            model=model,
            support_images=support_images,
            support_masks=support_masks,
            test_images=test_images,
            test_masks=test_masks,
            test_names=test_names,
            support_names=split.support_names,
            support_tissues=split.support_tissues,
            support_indices=split.support_idx,
            total_samples=split.total_samples,
            hardware_info=hardware_info,
        )
        return [result]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_unified_fewshot(
        self,
        model,
        support_images: list,
        support_masks: list,
        test_images: list,
        test_masks: list,
        test_names: list,
        support_names: list,
        support_tissues: list,
        support_indices: list,
        total_samples: int,
        hardware_info: dict,
    ) -> ExperimentResult:
        """Run unified fewshot experiment: zero-shot + both freeze_encoder modes in one execution.

        Returns a single ExperimentResult containing:
        - Pre-finetuning (zero-shot) inference on test set
        - Full-model fine-tuning + post-FT inference on test set
        - Frozen-encoder fine-tuning + post-FT inference on test set
        All three pieces stored in one result.json.
        """
        result = self._make_result_base(hardware_info)
        tb_zs = None
        tb_ft_full = None
        tb_inf_full = None
        tb_ft_frozen = None
        tb_inf_frozen = None
        stage = "setup"

        try:
            n_support = len(support_images)
            n_test    = len(test_images)
            result.num_shots        = n_support
            result.num_train_images = n_support
            result.num_test_images  = n_test
            result.n_dataset_images = total_samples

            # Validate support and test sets are non-empty
            if n_support == 0:
                raise ValueError(f"Support set is empty after tissue stratification (n_tissue_groups={len(support_tissues)}). Cannot fine-tune with zero images.")
            if n_test == 0:
                raise ValueError(f"Test set is empty; all {total_samples} images were selected for support. Need at least 1 test image for evaluation.")

            if not model.supports_finetune:
                raise NotImplementedError(
                    f"Model '{self.config.model_name}' does not support fine-tuning."
                )

            n_epochs   = self.config.n_epochs
            batch_size = max(1, min(n_support, _TISSUE_MAX_BATCH))

            # Model-specific sub-keys (e.g. InstanSeg's hotstart_epochs)
            _model_cfg = self.config.model_config or {}
            _extra_ft: dict = {}
            if "hotstart_epochs" in _model_cfg:
                _extra_ft["hotstart_epochs"] = _model_cfg["hotstart_epochs"]
            if "grad_accum_micro_batch" in _model_cfg:
                _extra_ft["grad_accum_micro_batch"] = _model_cfg["grad_accum_micro_batch"]

            from sapi.efficiency.profiler import TimedBlock, wait_for_gpu_cooldown

            profiler = self._build_profiler()

            # ── STEP 1: Zero-shot inference (shared, runs once) ──────────────
            stage = "zero_shot_inference"
            logger.info("  STEP 1: Zero-shot inference on test set (both modes use same)")
            model._ensure_loaded()  # load pretrained weights before the timed block starts
            tb_zs = TimedBlock(profiler, phase="inference", n_images=n_test)
            tb_zs.start()
            pred_masks_zs = model.predict_batch(test_images, batch_size=self.config.batch_size)
            zs_result = tb_zs.stop()
            zs_result.model_param_count = model.get_param_count()
            logger.info(
                f"    Zero-shot inference: {zs_result.wall_time_s:.2f}s  "
                f"{zs_result.time_per_image_s * 1000:.1f} ms/image"
            )

            from sapi.metrics import (
                compute_all_metrics,
                compute_raw_components,
                compute_sapi_zs,
                compute_sapi_lc,
            )
            seg_zs = compute_all_metrics(pred_masks_zs, test_masks, image_ids=test_names)
            perf_zs = seg_zs.summary_dict()
            logger.info(
                f"    Zero-shot: AJI+={perf_zs.get('aji_plus', 0):.4f}  "
                f"PQ={perf_zs.get('pq', 0):.4f}  NSD={perf_zs.get('nsd', 0):.4f}"
            )

            # Zero-shot inference finished — clear memory before fine-tuning.
            del model
            if _HAS_TORCH and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()
            wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)
            model = self._build_model()
            model._ensure_loaded()  # load pretrained weights before the timed block starts

            # ── STEP 2: Full-model fine-tuning ──────────────────────────────
            stage = "finetuning_full_model"
            logger.info("  STEP 2: Fine-tuning full model (freeze_encoder=False)")
            # Reseed immediately before training so this phase's augmentation
            # stream starts from a known state (config.seed), not whatever
            # state the global RNG happens to be in after setup/zero-shot.
            self._seed_everything(self.config.seed)
            tb_ft_full = TimedBlock(profiler, phase="finetuning", n_images=n_support)
            tb_ft_full.start()
            ft_metadata_full = model.finetune(
                support_images,
                support_masks,
                val_images=[],
                val_masks=[],
                patience=0,
                n_epochs=n_epochs,
                learning_rate=self.config.learning_rate,
                batch_size=batch_size,
                weight_decay=self.config.weight_decay,
                freeze_encoder=False,  # Full model fine-tuning
                **_extra_ft,
            )
            ft_result_full = tb_ft_full.stop()
            param_count = model.get_param_count()
            ft_result_full.model_param_count = param_count
            logger.info(
                f"    Full-model fine-tuning: {ft_result_full.wall_time_s:.1f}s  "
                f"epochs={n_epochs}  batch_size={batch_size}"
            )

            if _HAS_TORCH and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()
            wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)

            # ── STEP 3: Full-model evaluation and metrics ────────────────────
            stage = "inference_full_model"
            logger.info("  STEP 3: Evaluation of full-model fine-tuned weights")
            tb_inf_full = TimedBlock(profiler, phase="inference", n_images=n_test)
            tb_inf_full.start()
            pred_masks_full = model.predict_batch(test_images, batch_size=self.config.batch_size)
            inf_result_full = tb_inf_full.stop()
            inf_result_full.model_param_count = param_count
            logger.info(
                f"    Post-FT inference (full model): {inf_result_full.wall_time_s:.2f}s  "
                f"{inf_result_full.time_per_image_s * 1000:.1f} ms/image"
            )

            # Compute metrics for full-model
            seg_full = compute_all_metrics(pred_masks_full, test_masks, image_ids=test_names)
            perf_full = seg_full.summary_dict()
            logger.info(
                f"    Full-model metrics: AJI+={perf_full.get('aji_plus', 0):.4f}  "
                f"PQ={perf_full.get('pq', 0):.4f}  NSD={perf_full.get('nsd', 0):.4f}"
            )

            # ── STEP 4: Frozen-encoder fine-tuning ──────────────────────────
            # _ensure_loaded() is a lazy-init guard: it no-ops once a model is
            # already loaded, so it cannot "reload" a model that full_model
            # fine-tuning just mutated in place. Discard that instance and
            # build a fresh one instead, so frozen_encoder starts from the
            # same pretrained checkpoint as full_model did, not from
            # full_model's fine-tuned weights.
            stage = "finetuning_frozen_encoder"
            logger.info("  STEP 4: Fine-tuning frozen encoder (freeze_encoder=True)")
            del model
            if _HAS_TORCH and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()
            wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)
            model = self._build_model()
            model._ensure_loaded()  # load pretrained weights before the timed block starts
            # Reseed so frozen_encoder's augmentation stream starts from the
            # same state full_model's did (see STEP 2), isolating freeze_encoder's
            # effect as the module docstring documents — without this, frozen_encoder
            # would continue drawing from wherever full_model's training left the
            # global RNG.
            self._seed_everything(self.config.seed)

            tb_ft_frozen = TimedBlock(profiler, phase="finetuning", n_images=n_support)
            tb_ft_frozen.start()
            ft_metadata_frozen = model.finetune(
                support_images,
                support_masks,
                val_images=[],
                val_masks=[],
                patience=0,
                n_epochs=n_epochs,
                learning_rate=self.config.learning_rate,
                batch_size=batch_size,
                weight_decay=self.config.weight_decay,
                freeze_encoder=True,  # Frozen encoder fine-tuning
                **_extra_ft,
            )
            ft_result_frozen = tb_ft_frozen.stop()
            ft_result_frozen.model_param_count = param_count
            logger.info(
                f"    Frozen-encoder fine-tuning: {ft_result_frozen.wall_time_s:.1f}s  "
                f"epochs={n_epochs}  batch_size={batch_size}"
            )

            if _HAS_TORCH and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()
            wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)

            # ── STEP 5: Frozen-encoder evaluation and metrics ────────────────
            stage = "inference_frozen_encoder"
            logger.info("  STEP 5: Evaluation of frozen-encoder fine-tuned weights")
            tb_inf_frozen = TimedBlock(profiler, phase="inference", n_images=n_test)
            tb_inf_frozen.start()
            pred_masks_frozen = model.predict_batch(test_images, batch_size=self.config.batch_size)
            inf_result_frozen = tb_inf_frozen.stop()
            inf_result_frozen.model_param_count = param_count
            logger.info(
                f"    Post-FT inference (frozen encoder): {inf_result_frozen.wall_time_s:.2f}s  "
                f"{inf_result_frozen.time_per_image_s * 1000:.1f} ms/image"
            )

            # Compute metrics for frozen-encoder
            seg_frozen = compute_all_metrics(pred_masks_frozen, test_masks, image_ids=test_names)
            perf_frozen = seg_frozen.summary_dict()
            logger.info(
                f"    Frozen-encoder metrics: AJI+={perf_frozen.get('aji_plus', 0):.4f}  "
                f"PQ={perf_frozen.get('pq', 0):.4f}  NSD={perf_frozen.get('nsd', 0):.4f}"
            )

            # Frozen-encoder inference finished — the model is no longer
            # needed (remaining steps only compute metrics from predictions
            # already captured above), so clear memory now. Cooldown here too:
            # this experiment's last training (frozen-encoder FT) may leave the
            # GPU hot, and whatever runs next (another experiment, launched as
            # a separate process by an external sweep) must not inherit that
            # heat into its own baseline/inference measurement.
            del model
            if _HAS_TORCH and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
            import gc as _gc; _gc.collect()
            wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)

            # ── STEP 6: Compute SAPI components for all three scenarios ──────
            stage = "sapi_components"
            logger.info("  STEP 6: Computing SAPI scores for all three scenarios")

            # Compute SAPI_ZS for zero-shot (inference-only, no training energy)
            spec_raw_zs = compute_raw_components(
                perf_metrics=perf_zs,
                inf_profiler=zs_result.to_dict(),
                n_test_images=n_test,
                train_profiler=None,  # Zero-shot has no training
                n_train_images=0,
                n_params=param_count,
            )
            sapi_zs = compute_sapi_zs(
                spec_raw_zs["p_score"],
                spec_raw_zs["ei_score_mj_per_image"],
                spec_raw_zs["m_raw"],
            )
            spec_raw_zs["sapi_zs"] = sapi_zs

            # Full-model and frozen-encoder results are stored separately, never
            # averaged together — they are different fine-tuning configurations.
            n_run_full = (ft_metadata_full or {}).get("n_epochs_run") or n_epochs
            n_run_frozen = (ft_metadata_frozen or {}).get("n_epochs_run") or n_epochs

            # Full-model SAPI_LC components
            spec_raw_full = compute_raw_components(
                perf_metrics=perf_full,
                inf_profiler=inf_result_full.to_dict(),
                n_test_images=n_test,
                train_profiler=ft_result_full.to_dict(),
                n_train_images=n_support,
                n_params=param_count,
            )
            sapi_lc_full = compute_sapi_lc(
                spec_raw_full["p_score"],
                spec_raw_full["ei_score_mj_per_image"],
                spec_raw_full["et_score_mj_per_image"],
                spec_raw_full["m_raw"],
            )
            spec_raw_full["sapi_lc"] = sapi_lc_full

            # Frozen-encoder SAPI_LC components
            spec_raw_frozen = compute_raw_components(
                perf_metrics=perf_frozen,
                inf_profiler=inf_result_frozen.to_dict(),
                n_test_images=n_test,
                train_profiler=ft_result_frozen.to_dict(),
                n_train_images=n_support,
                n_params=param_count,
            )
            sapi_lc_frozen = compute_sapi_lc(
                spec_raw_frozen["p_score"],
                spec_raw_frozen["ei_score_mj_per_image"],
                spec_raw_frozen["et_score_mj_per_image"],
                spec_raw_frozen["m_raw"],
            )
            spec_raw_frozen["sapi_lc"] = sapi_lc_frozen

            # Build performance_metrics with hierarchical structure (all separate, no averaging)
            metrics = {
                "model_param_count": param_count,
                "zero_shot": {
                    **perf_zs,
                    **spec_raw_zs,  # Includes ei_score_mj_per_image, m_raw, p_score, sapi_zs
                },
                "full_model": {
                    **perf_full,
                    **spec_raw_full,  # Includes ei/et_score_mj_per_image, m_raw, p_score, sapi_lc
                },
                "frozen_encoder": {
                    **perf_frozen,
                    **spec_raw_frozen,  # Includes ei/et_score_mj_per_image, m_raw, p_score, sapi_lc
                },
            }

            result.performance_metrics = metrics

            # Per-image records for all three phases (zero_shot, full_model, frozen_encoder)
            per_image_records = []
            # Zero-shot records
            for rec in self.build_per_image_records(seg_zs, stage="zero_shot"):
                per_image_records.append(rec)
            # Full-model records
            for rec in self.build_per_image_records(seg_full, stage="full_model"):
                per_image_records.append(rec)
            # Frozen-encoder records
            for rec in self.build_per_image_records(seg_frozen, stage="frozen_encoder"):
                per_image_records.append(rec)
            result.performance_metrics_per_image = per_image_records

            # Store efficiency results for all three phases (separate, not averaged)
            result.inference_efficiency = {
                "zero_shot": zs_result.to_dict(),
                "full_model": inf_result_full.to_dict(),
                "frozen_encoder": inf_result_frozen.to_dict(),
            }
            result.training_efficiency = {
                "full_model": ft_result_full.to_dict(),
                "frozen_encoder": ft_result_frozen.to_dict(),
            }

            # Store metadata for both modes (simplified: no checkpoints, no val_losses)
            # E_t = training_efficiency[mode]["it_energy_mj"] / n_support
            _tm = {
                "num_shots": n_support,
                "n_tissue_groups": n_support,
                "tissue_groups": support_tissues,
                "support_sample_names": support_names,
                "support_sample_indices": support_indices,
                "n_epochs": n_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": batch_size,
                "weight_decay": self.config.weight_decay,
                "seed": self.config.seed,
                "patience": 0,
                # Full-model training metadata (100 epochs, no checkpoints)
                "full_model": {
                    "n_epochs": n_run_full,
                    "train_losses": ft_metadata_full.get("train_losses", []) if ft_metadata_full else [],
                    "final_train_loss": (ft_metadata_full.get("train_losses", []) or [None])[-1],
                    "p_score": perf_full.get("p_score"),
                    # E_t_mJ = training_efficiency[full_model]["it_energy_mj"] / n_support
                    "e_t_mj_per_image": ft_result_full.it_energy_mj / n_support,
                },
                # Frozen-encoder training metadata (100 epochs, no checkpoints)
                "frozen_encoder": {
                    "n_epochs": n_run_frozen,
                    "train_losses": ft_metadata_frozen.get("train_losses", []) if ft_metadata_frozen else [],
                    "final_train_loss": (ft_metadata_frozen.get("train_losses", []) or [None])[-1],
                    "p_score": perf_frozen.get("p_score"),
                    # E_t_mJ = training_efficiency[frozen_encoder]["it_energy_mj"] / n_support
                    "e_t_mj_per_image": ft_result_frozen.it_energy_mj / n_support,
                },
            }
            result.training_metadata = _tm
            # Store the actual epochs run (both modes train for same budget, so they should be identical)
            # Use max as fallback in case they differ due to early stopping or other issues
            result.num_epochs = max(int(n_run_full), int(n_run_frozen))
            result.num_epochs_max = n_epochs

            logger.info(
                f"\n  ── UNIFIED FEWSHOT SUMMARY ──────────────────────────────\n"
                f"    Zero-shot:     AJI+={perf_zs.get('aji_plus', 0):.4f}  "
                f"PQ={perf_zs.get('pq', 0):.4f}  NSD={perf_zs.get('nsd', 0):.4f}  "
                f"SAPI_ZS={sapi_zs:.4f}\n"
                f"    Full-model FT: AJI+={perf_full.get('aji_plus', 0):.4f}  "
                f"PQ={perf_full.get('pq', 0):.4f}  NSD={perf_full.get('nsd', 0):.4f}  "
                f"SAPI_LC={sapi_lc_full:.4f}\n"
                f"    Frozen-enc FT: AJI+={perf_frozen.get('aji_plus', 0):.4f}  "
                f"PQ={perf_frozen.get('pq', 0):.4f}  NSD={perf_frozen.get('nsd', 0):.4f}  "
                f"SAPI_LC={sapi_lc_frozen:.4f}"
            )

            # Predictions from both modes are stored (not just full_model).
            # Validate length matching to prevent silent data loss from zip()
            assert len(test_names) == len(pred_masks_full), \
                f"Prediction length mismatch: {len(test_names)} test names vs {len(pred_masks_full)} full_model predictions"
            assert len(test_names) == len(pred_masks_frozen), \
                f"Prediction length mismatch: {len(test_names)} test names vs {len(pred_masks_frozen)} frozen_encoder predictions"

            result.predictions = {
                "full_model": {
                    name: mask for name, mask in zip(test_names, pred_masks_full)
                },
                "frozen_encoder": {
                    name: mask for name, mask in zip(test_names, pred_masks_frozen)
                },
            }

        except Exception as exc:
            logger.exception(f"  Unified fewshot failed at stage '{stage}': {exc}")
            result.mark_failed(exc, stage)
            # Clean up any active profilers. TimedBlock.stop() raises if called
            # out of order or twice, so each call is wrapped individually below.
            for _tb in (tb_zs, tb_ft_full, tb_inf_full, tb_ft_frozen, tb_inf_frozen):
                try:
                    if _tb is not None:
                        _tb.stop()
                except Exception:
                    pass
            if "model" in locals():
                del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            import gc as _gc; _gc.collect()
            try:
                from sapi.efficiency.profiler import wait_for_gpu_cooldown
                wait_for_gpu_cooldown(gpu_index=self.config.gpu_index)
            except Exception:
                pass

        return result

    # ── CLI summary ──────────────────────────────────────────────────────────

    def summarize(self, result: ExperimentResult) -> str:
        if result.error:
            return (
                f"Tissue fewshot failed [{result.failure_type}] "
                f"at stage '{result.failure_stage}': {result.error}"
            )
        p  = result.performance_metrics or {}
        e  = result.inference_efficiency or {}
        tm = result.training_metadata or {}
        cfg = result.config or {}

        # Extract metrics from hierarchical structure
        p_zs = p.get("zero_shot", {})
        p_fm = p.get("full_model", {})
        p_fe = p.get("frozen_encoder", {})

        e_fm = e.get("full_model", {})
        e_fe = e.get("frozen_encoder", {})

        lines = [
            "── TISSUE-STRATIFIED FEWSHOT RESULTS (UNIFIED EXECUTION) ──────",
            f"  Tissue groups ({tm.get('n_tissue_groups', result.num_shots)}): "
            f"{tm.get('tissue_groups', [])}",
            f"  Support images: {tm.get('support_sample_names', [])}",
            f"  Epochs: {result.num_epochs}  |  lr={tm.get('learning_rate', '?')}  |  patience=0",
            "",
            f"  Zero-shot baseline:",
            f"    AJI+={p_zs.get('aji_plus', 0):.4f}  PQ={p_zs.get('pq', 0):.4f}  "
            f"NSD={p_zs.get('nsd', 0):.4f}  P={p_zs.get('p_score', 0):.4f}",
            f"    Energy={p_zs.get('ei_score_mj_per_image', 0):.1f} mJ/img  "
            f"SAPI_ZS={p_zs.get('sapi_zs', float('nan')):.4f}",
            "",
            f"  Full-model fine-tuning (freeze_encoder=False):",
            f"    AJI+={p_fm.get('aji_plus', 0):.4f}  PQ={p_fm.get('pq', 0):.4f}  "
            f"NSD={p_fm.get('nsd', 0):.4f}  P={p_fm.get('p_score', 0):.4f}",
            f"    E_i={p_fm.get('ei_score_mj_per_image', 0):.1f}  "
            f"E_t={p_fm.get('et_score_mj_per_image', 0):.1f} mJ/img  "
            f"SAPI_LC={p_fm.get('sapi_lc', float('nan')):.4f}",
            "",
            f"  Frozen-encoder fine-tuning (freeze_encoder=True):",
            f"    AJI+={p_fe.get('aji_plus', 0):.4f}  PQ={p_fe.get('pq', 0):.4f}  "
            f"NSD={p_fe.get('nsd', 0):.4f}  P={p_fe.get('p_score', 0):.4f}",
            f"    E_i={p_fe.get('ei_score_mj_per_image', 0):.1f}  "
            f"E_t={p_fe.get('et_score_mj_per_image', 0):.1f} mJ/img  "
            f"SAPI_LC={p_fe.get('sapi_lc', float('nan')):.4f}",
        ]
        return "\n".join(lines)
