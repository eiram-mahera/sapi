"""
Experiment Type 1 - Pretrained inference.

The pretrained model is run on ALL available samples in the dataset.
No training, no data split. This gives a complete picture of how well
the off-the-shelf model performs on the target modality.
"""

from __future__ import annotations

import logging
from typing import List

from .base import BaseExperiment, ExperimentResult

logger = logging.getLogger(__name__)


class InferenceExperiment(BaseExperiment):
    """
    Loads the pretrained model and runs inference on every sample in the dataset.
    All performance and efficiency metrics are computed over the full dataset.
    """

    experiment_type = "inference"
    measures_energy = True
    yaml_section_chain = ()           # no training hyperparameters

    def execute(self) -> List[ExperimentResult]:
        self._seed_everything(self.config.seed)
        hardware_info = self._collect_hardware_info()
        result = self._make_result_base(hardware_info)
        tb = None
        stage = "build_model"

        try:
            # 1. Build pretrained model (weights loaded automatically).
            #    Must happen before data loading so we can query needs_raw_he_image.
            model = self._build_model()

            # 2. Load all data — no train/test split for inference evaluation.
            #    raw_he=True skips stain separation and returns full RGB for models
            #    that need colour H&E input (e.g. stardist_he).
            stage = "load_data"
            dataset = self._load_all_data()
            raw_he = getattr(model, "needs_raw_he_image", False)
            images, gt_masks = dataset.get_images_and_masks(raw_he=raw_he)
            n = len(images)
            logger.info(f"  Inference on all {n} images (dataset: {self.config.dataset_name})")

            # 3. Profile inference
            stage = "inference"
            profiler = self._build_profiler()
            from sapi.efficiency.profiler import TimedBlock

            model._ensure_loaded()  # load pretrained weights before the timed block starts
            tb = TimedBlock(profiler, phase="inference", n_images=n)
            tb.start()
            pred_masks = model.predict_batch(images, batch_size=self.config.batch_size)
            inf_result = tb.stop()

            # Query param count after predict_batch so lazy-loading models (e.g. CellViT)
            # have called _ensure_loaded() and _model is populated.
            param_count = model.get_param_count()
            if param_count > 0:
                logger.info(f"  Model: {self.config.model_name}  params: {param_count:,}")
            inf_result.model_param_count = param_count

            logger.info(
                f"  Inference: {inf_result.wall_time_s:.2f}s total  "
                f"{inf_result.time_per_image_s * 1000:.1f} ms/image  "
                f"energy={inf_result.it_energy_mj:.3f} mJ  "
                f"CO2={inf_result.co2_eq_g:.4f} g"
            )

            # 4. Compute segmentation metrics
            stage = "metrics"
            from sapi.metrics import compute_all_metrics, compute_raw_components, compute_sapi_zs

            seg = compute_all_metrics(
                pred_masks, gt_masks, image_ids=dataset.sample_names
            )
            logger.info(
                f"  AJI+={seg.aji_plus:.4f}  PQ={seg.pq:.4f}  NSD={seg.nsd:.4f}  "
                f"(included={seg.num_included}/{seg.num_images}, "
                f"both_empty={seg.num_both_empty})"
            )

            # 5. SAPI raw components + SAPI_ZS (computed at runtime, not analysis time)
            # Energy composition: E_IT = E_GPU + E_CPU + E_RAM
            # Per-image inference energy: Ei = E_IT / N_test
            spec_raw = compute_raw_components(
                perf_metrics=seg.summary_dict(),
                inf_profiler=inf_result.to_dict(),
                n_test_images=n,
                train_profiler=None,
                n_params=param_count,
            )
            # Energy values are now in mJ from the profiler, no conversion needed
            sapi_zs = compute_sapi_zs(
                spec_raw["p_score"],
                spec_raw["ei_score_mj_per_image"],  # Ei = E_IT / N_test (in mJ/image)
                spec_raw["m_raw"],
            )
            spec_raw["sapi_zs"] = sapi_zs
            logger.info(
                f"  P={spec_raw['p_score']:.4f}  "
                f"Ei={spec_raw['ei_score_mj_per_image']:.3f} mJ/img  "
                f"S={spec_raw['s_score_co2_per_image']*1e6:.3f} µgCO2/img  "
                f"SAPI_ZS={sapi_zs:.4f}"
            )

            metrics = seg.summary_dict()
            metrics.update(spec_raw)
            metrics["model_param_count"] = param_count

            result.performance_metrics = metrics
            result.performance_metrics_per_image = self.build_per_image_records(seg)
            result.inference_efficiency = inf_result.to_dict()
            result.num_test_images = n
            result.n_dataset_images = n
            # Predicted masks for every image, saved to predictions/<sample>.npz
            # by ResultLogger.
            result.predictions = {
                name: mask for name, mask in zip(dataset.sample_names, pred_masks)
            }

        except Exception as exc:
            logger.exception(f"  Inference failed at stage '{stage}': {exc}")
            result.mark_failed(exc, stage)
            try:
                if tb is not None and tb._monitor is not None and tb._monitor.is_alive():
                    tb.stop()
            except Exception:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return [result]

    # ── CLI summary ──────────────────────────────────────────────────────────

    def summarize(self, result: ExperimentResult) -> str:
        if result.error:
            return (
                f"Experiment failed [{result.failure_type}] "
                f"at stage '{result.failure_stage}': {result.error}"
            )

        p = result.performance_metrics
        e = result.inference_efficiency
        lines: List[str] = []

        lines.append("── SEGMENTATION PERFORMANCE ──────────────────────────────────")
        for label, key in [
            ("AJI+", "aji_plus"), ("PQ", "pq"), ("NSD", "nsd"),
        ]:
            v = p.get(key)
            if v is not None:
                lines.append(f"  {label:<20} {v:.4f}")

        lines.append("\n── SAPI RAW COMPONENTS ───────────────────────────────────────")
        lines.append(f"  {'P (performance)':28} {p.get('p_score', 0):.6f}")
        # Ei: mJ is the native unit now
        ei_mj = p.get('ei_score_mj_per_image', 0)
        lines.append(f"  {'Ei (inf energy, mJ/img)':28} {ei_mj:.4f}")
        # Et: mJ is the native unit now
        et_mj = p.get('et_score_mj_per_image', 0)
        lines.append(f"  {'Et (train energy, mJ/img)':28} {et_mj:.4f}")
        lines.append(f"  {'S (CO2, µgCO2/img)':28} {p.get('s_score_co2_per_image', 0) * 1e6:.4f}")
        lines.append(f"  {'M (log10 params)':28} {p.get('m_raw', 0):.4f}")
        sapi_zs_val = p.get("sapi_zs")
        sapi_zs_str = f"{sapi_zs_val:.6f}" if sapi_zs_val is not None else "n/a"
        lines.append(f"  {'SAPI_ZS':28} {sapi_zs_str}")

        lines.append("\n── INFERENCE EFFICIENCY ──────────────────────────────────────")
        for label, key, scale, unit in [
            ("Total time", "wall_time_s", 1, "s"),
            ("Time / image", "time_per_image_s", 1000, "ms"),
            ("Throughput", "throughput_img_per_s", 1, "img/s"),
            ("GPU peak", "peak_gpu_memory_mb", 1, "MB"),
            ("CPU energy", "cpu_energy_mj", 1, "mJ"),
            ("GPU energy", "gpu_energy_mj", 1, "mJ"),
            ("RAM energy", "ram_energy_mj", 1, "mJ"),
            ("IT energy", "it_energy_mj", 1, "mJ"),
            ("PUE", "pue", 1, ""),
            ("Facility energy", "total_facility_energy_mj", 1, "mJ"),
            ("CO2", "co2_eq_g", 1, "gCO2eq"),
            ("CO2 / image", "co2_eq_g_per_image", 1e6, "µgCO2eq"),
            ("Carbon intens.", "carbon_intensity_g_per_kwh", 1, "gCO2/kWh"),
        ]:
            v = e.get(key)
            if v is not None:
                lines.append(f"  {label:<22} {float(v) * scale:.4f} {unit}")
        loc = e.get("co2_location", "")
        if loc:
            lines.append(f"  {'CO2 location':<22} {loc}")

        n_params = p.get("model_param_count", -1)
        if n_params and n_params > 0:
            lines.append(f"\n  Model parameters: {int(n_params):,}")
        lines.append(f"  Test images:      {result.num_test_images}")
        lines.append(f"\n  Seed:            {(result.config or {}).get('seed', '?')}")
        lines.append(f"  Experiment time: {result.experiment_datetime}")
        return "\n".join(lines)
