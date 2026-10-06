"""
Collect hardware and software environment information.

Called once at the start of each experiment so every result file contains
enough of a snapshot of the machine it was produced on for reproducibility.
Only fields that are actually load-bearing for reproducing a run or
interpreting its energy numbers are collected — no live/transient telemetry
(instantaneous VRAM/RAM usage, temperature, power draw, CPU frequency, etc.),
which reflects shared-machine state at metadata-collection time rather than
anything about the run itself (that's what the energy profiler measures).
"""

from __future__ import annotations

import logging
import platform
import sys

logger = logging.getLogger(__name__)


def collect_hardware_info() -> dict:
    """
    Return a dict containing the CPU, RAM, GPU, OS, and software version
    fields needed to reproduce a run or interpret its energy numbers.
    Fields that cannot be read are omitted rather than raising exceptions.
    """
    info: dict = {}

    _add_os_info(info)
    _add_cpu_info(info)
    _add_ram_info(info)
    _add_gpu_info(info)
    _add_software_versions(info)

    return info


# ---------------------------------------------------------------------------
# OS
# ---------------------------------------------------------------------------

def _add_os_info(info: dict):
    info["os_name"] = platform.system()
    info["python_version"] = sys.version.split()[0]


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------

def _add_cpu_info(info: dict):
    try:
        import psutil
        info["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        info["cpu_logical_cores"] = psutil.cpu_count(logical=True)
    except Exception as exc:
        logger.debug(f"psutil CPU info unavailable: {exc}")

    # Detailed model name via py-cpuinfo (optional dependency)
    try:
        import cpuinfo  # py-cpuinfo
        cpu = cpuinfo.get_cpu_info()
        info["cpu_model"] = cpu.get("brand_raw", "unknown")
    except ImportError:
        info["cpu_model"] = platform.processor() or "unknown"
    except Exception as exc:
        logger.debug(f"py-cpuinfo unavailable: {exc}")
        info["cpu_model"] = platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# RAM
# ---------------------------------------------------------------------------

def _add_ram_info(info: dict):
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1024 ** 3, 2)
    except Exception as exc:
        logger.debug(f"psutil RAM info unavailable: {exc}")


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

def _add_gpu_info(info: dict):
    # NVIDIA GPU info via pynvml. A failure reading one GPU (or one field of
    # one GPU) must not discard GPUs that were already read successfully —
    # each level has its own try/except, and nvmlShutdown() always runs once
    # nvmlInit() has succeeded, even if enumeration fails partway through.
    info["gpu_count"] = 0
    info["gpus"] = []
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug(f"pynvml GPU info unavailable: {exc}")
    else:
        try:
            try:
                count = pynvml.nvmlDeviceGetCount()
            except Exception as exc:
                logger.debug(f"pynvml device count unavailable: {exc}")
                count = 0

            try:
                info["gpu_driver_version"] = _decode(pynvml.nvmlSystemGetDriverVersion())
            except Exception as exc:
                logger.debug(f"pynvml driver version unavailable: {exc}")

            gpus = []
            for i in range(count):
                try:
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                    g: dict = {
                        "index": i,
                        "name": _decode(pynvml.nvmlDeviceGetName(h)),
                        "vram_total_gb": round(mem.total / 1024 ** 3, 2),
                    }
                    try:
                        g["power_limit_w"] = round(
                            pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0, 1
                        )
                    except Exception:
                        pass
                    try:
                        g["compute_capability"] = ".".join(
                            str(x)
                            for x in pynvml.nvmlDeviceGetCudaComputeCapability(h)
                        )
                    except Exception:
                        pass
                    gpus.append(g)
                except Exception as exc:
                    logger.debug(f"pynvml info for GPU {i} unavailable: {exc}")

            info["gpu_count"] = len(gpus)
            info["gpus"] = gpus
        except Exception as exc:
            logger.debug(f"pynvml GPU enumeration unavailable: {exc}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    # CUDA / cuDNN versions — PyTorch first, then TF/pynvml fallback for TF-only envs
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda or "n/a"
        cudnn_version = torch.backends.cudnn.version()
        info["cudnn_version"] = str(cudnn_version) if cudnn_version is not None else "n/a"
    except ImportError:
        # PyTorch absent (e.g. tf215env / tf28env) — detect GPU via pynvml or TF
        gpu_present = False
        try:
            import pynvml as _pynvml
            _pynvml.nvmlInit()
            try:
                gpu_present = _pynvml.nvmlDeviceGetCount() > 0
            finally:
                _pynvml.nvmlShutdown()
        except Exception:
            try:
                import tensorflow as tf
                gpu_present = len(tf.config.list_physical_devices("GPU")) > 0
            except Exception:
                pass
        info["cuda_available"] = gpu_present
    except Exception as exc:
        logger.debug(f"PyTorch CUDA info unavailable: {exc}")


# ---------------------------------------------------------------------------
# Software versions
# ---------------------------------------------------------------------------

def _add_software_versions(info: dict):
    packages = [
        "torch", "torchvision", "numpy", "scipy",
        "tifffile", "psutil",
        "cellpose", "stardist", "deepcell",
        "skimage", "cv2",
    ]
    versions: dict = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "installed")
        except Exception:
            pass  # only record packages that are actually present
    info["software_versions"] = versions


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _decode(val) -> str:
    """Decode bytes to str if necessary."""
    return val.decode("utf-8") if isinstance(val, bytes) else str(val)
