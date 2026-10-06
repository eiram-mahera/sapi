"""
Hardware-domain efficiency and sustainability profiler for the benchmark.

PRIMARY ENERGY RELATIONSHIPS
=============================

All energy is measured in millijoules (mJ) for SAPI score calculations.
The fundamental composition of IT equipment energy is:

    E_IT = E_GPU + E_CPU + E_RAM     (sum of all hardware domains)

For inference (zero-shot) and fine-tuning experiments, energy is normalized per image:

    E_i  = E_IT / N_test             (IT inference energy per test image)
    E_ft = E_IT / N_support          (IT fine-tuning energy per training image)

Facility energy accounts for infrastructure (cooling, UPS, distribution):

    E_facility = E_IT × PUE          (PUE = Power Usage Effectiveness = 1.3)

Primary energy measurements for an exclusively allocated compute node:
    CPU energy   – start/end differences of Intel RAPL package counters
    GPU energy   – start/end difference of the NVML cumulative device-energy
                   counter, with timestamped power integration as a fallback
    RAM energy   – start/end differences of Intel RAPL DRAM counters

PUE (Power Usage Effectiveness):
    PUE = Total Facility Power / IT Equipment Power
    IT energy × PUE = total facility energy (accounts for cooling, UPS, etc.)
    The benchmark uses the fixed value PUE = 1.3 for every experiment.

Carbon emissions:
    Default location label: Bielefeld, Germany (North Rhine-Westphalia, DEU)
    Germany 2025 grid intensity ≈ 309 gCO₂eq/kWh (Umweltbundesamt)
    Electricity Maps, a timeline file, or an explicit override supplies the
    operational carbon intensity; the German value is retained as a fallback.

Measurement scope:
    NVML and RAPL expose device/package/DRAM-domain energy, not genuine
    per-process energy.  When the node and selected GPU are exclusively allocated
    to this benchmark, the complete domain energy observed during the synchronized
    timed interval is attributed to the workload.  Static/idle energy is retained.
    The fixed node contains two GPU cards, but each experiment uses exactly one
    selected card.  The profiler rejects CPU-only configuration, RAPL coverage
    inconsistent with the node's detected CPU-package topology, foreign GPU
    processes anywhere on the node, and
    benchmark processes using a non-selected GPU.  It does not apply CPU-time,
    GPU-memory, or GPU-utilization attribution heuristics.

Usage:
    profiler = EfficiencyProfiler(device="cuda", pue=1.3)
    tb = TimedBlock(profiler, phase="inference", n_images=100)
    tb.start()
    predictions = model.predict_batch(images)   ← the actual work
    result = tb.stop()                          ← returns ProfilerResult
    print(result.summary())
"""

from __future__ import annotations

import csv
import gc
import glob
import json
import logging
import math
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import psutil

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fixed benchmark hardware and facility assumptions
# ---------------------------------------------------------------------------

FIXED_PUE: float = 1.3
EXPECTED_NODE_GPU_COUNT: int = 2


# ---------------------------------------------------------------------------
# Germany / Bielefeld carbon intensity (fallback constant)
# ---------------------------------------------------------------------------
# Source: Umweltbundesamt (Federal Environment Agency), 2025 German grid average.
# Bielefeld is served by the German national grid (North Rhine-Westphalia, DEU).
GERMANY_CARBON_INTENSITY_G_PER_KWH: float = 309.0
LOCATION_LABEL: str = "Bielefeld, Germany (North Rhine-Westphalia, DEU)"

# Energy unit conversion factors
KWH_TO_MJ: float = 3.6e9  # 1 kWh = 3.6e9 millijoules

# GPU cooldown constants (in units shown)
GPU_COOLDOWN_TARGET_C: int = 45      # °C threshold to proceed
GPU_COOLDOWN_MANDATORY_S: int = 60   # unconditional sleep before first temperature check
GPU_COOLDOWN_RETRY_S: int = 30       # sleep between retries when temp > target

# Profiler monitoring constants
MONITOR_INTERVAL_S: float = 0.2      # GPU/CPU polling interval (seconds)
GPU_BASELINE_WARNING_THRESHOLD_W: int = 100  # warning if idle GPU power exceeds this (W)

# Electricity Maps API endpoint (v3)
_ELECTRICITYMAPS_URL = "https://api.electricitymap.org/v3/carbon-intensity/latest?zone={zone}"


# ---------------------------------------------------------------------------
# Electricity Maps – real-time carbon intensity
# ---------------------------------------------------------------------------

def fetch_electricitymaps_intensity(
    token: str,
    zone: str = "DE",
    timeout: float = 8.0,
    proxy_url: str = "",
) -> tuple[float | None, str | None]:
    """
    Fetch current carbon intensity from the Electricity Maps API.

    The carbon intensity of Germany's grid varies significantly by time of day
    (lower at midday when solar is online, higher at night when fossil fuels
    dominate). This function retrieves the current value rather than using the
    annual average static fallback (309 gCO₂eq/kWh, Umweltbundesamt 2025).

    Args:
        token:     Electricity Maps API token (free tier supports 1 req/hour).
                   Obtain at https://electricitymaps.com/free-tier
        zone:      Electricity Maps zone code. "DE" = Germany (whole country).
                   More granular regional zones exist but "DE" covers Bielefeld.
        timeout:   HTTP timeout in seconds.
        proxy_url: Optional HTTP/HTTPS proxy URL, e.g. "http://proxy.hpc.edu:3128".
                   Useful on HPC compute nodes that have no direct internet access
                   but can reach the internet via a campus proxy.
                   If empty, the system default proxy settings are used
                   (respects HTTP_PROXY / HTTPS_PROXY environment variables).

    Returns:
        (carbon_intensity_g_per_kwh, reading_datetime) where reading_datetime is
        the ISO-8601 validity time of the CI reading from the API response
        (payload "datetime"/"updatedAt"), or None when it could not be read.
        Returns ``(None, None)`` if unavailable.  The caller applies and labels
        the static fallback so that a failed request cannot be misreported as a
        live Electricity Maps observation.
    """
    url = _ELECTRICITYMAPS_URL.format(zone=zone)
    req = urllib.request.Request(url, headers={"auth-token": token})

    if proxy_url:
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
        )
    else:
        opener = urllib.request.build_opener()

    try:
        with opener.open(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        intensity = float(payload["carbonIntensity"])
        if not math.isfinite(intensity) or intensity <= 0.0:
            raise ValueError(
                f"invalid carbonIntensity value {payload['carbonIntensity']!r}"
            )
        # Validity time of the reading (Electricity Maps returns "datetime" for
        # the CI value and "updatedAt" for when the record was last refreshed).
        reading_dt = payload.get("datetime") or payload.get("updatedAt")
        logger.info(
            f"Electricity Maps [{zone}]: live carbon intensity = "
            f"{intensity:.1f} gCO₂eq/kWh (reading time: {reading_dt})"
        )
        return intensity, reading_dt
    except urllib.error.HTTPError as exc:
        logger.warning(
            f"Electricity Maps API HTTP {exc.code} for zone={zone}; "
            f"falling back to static {GERMANY_CARBON_INTENSITY_G_PER_KWH} g/kWh (Germany 2025)"
        )
    except urllib.error.URLError as exc:
        logger.warning(
            f"Electricity Maps API unreachable ({exc.reason}); "
            f"falling back to static {GERMANY_CARBON_INTENSITY_G_PER_KWH} g/kWh (Germany 2025)"
        )
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            f"Electricity Maps API response parse error ({exc}); "
            f"falling back to static {GERMANY_CARBON_INTENSITY_G_PER_KWH} g/kWh (Germany 2025)"
        )
    except Exception as exc:
        logger.warning(
            f"Electricity Maps API error ({exc}); "
            f"falling back to static {GERMANY_CARBON_INTENSITY_G_PER_KWH} g/kWh (Germany 2025)"
        )
    return None, None


# ---------------------------------------------------------------------------
# CI timeline file — offline / HPC fallback
# ---------------------------------------------------------------------------

def read_ci_from_timeline(path: str, now: datetime | None = None) -> tuple[float, float, str] | None:
    """
    Read a carbon intensity timeline CSV and return
    (intensity_g_per_kwh, age_seconds, entry_timestamp_iso) for the entry whose
    timestamp is closest to `now` (defaults to current UTC time).

    The expected CSV format is what collect_carbon_intensity.py produces:
        timestamp_utc,zone,carbon_intensity_g_per_kwh
        2025-05-04T12:00:00+00:00,DE,245.3
        ...

    Returns None if the file does not exist, is empty, or cannot be parsed.
    The caller decides whether the age is acceptable.
    """
    if not path:
        return None

    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    try:
        entries: list[tuple[datetime, float]] = []
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp_utc"])
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    ci = float(row["carbon_intensity_g_per_kwh"])
                    if not math.isfinite(ci) or ci <= 0.0:
                        continue
                    entries.append((ts, ci))
                except (KeyError, ValueError):
                    continue

        if not entries:
            return None

        closest_ts, closest_ci = min(entries, key=lambda e: abs((e[0] - now).total_seconds()))
        age_s = abs((now - closest_ts).total_seconds())
        return closest_ci, age_s, closest_ts.isoformat()

    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning(f"CI timeline read error ({exc}); file: {path}")
        return None


# ---------------------------------------------------------------------------
# RAPL accessibility check
# ---------------------------------------------------------------------------

_RAPL_SYSFS_ROOT = "/sys/class/powercap"
_CPU_SYSFS_ROOT = "/sys/devices/system/cpu"
_CPUINFO_PATH = "/proc/cpuinfo"


@dataclass(frozen=True)
class _RaplDomain:
    """One readable Linux powercap RAPL energy domain."""

    energy_path: str
    name: str
    kind: str
    max_energy_range_uj: int
    package_id: int = -1


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read().strip()


def _parse_rapl_package_id(name: str) -> int | None:
    """Return the physical package identifier encoded by a RAPL package name."""
    match = re.fullmatch(r"package(?:[-_\s]+)?(\d+)", name.strip().lower())
    return int(match.group(1)) if match else None


def _discover_cpu_package_ids(
    cpu_root: str = _CPU_SYSFS_ROOT,
    cpuinfo_path: str = _CPUINFO_PATH,
) -> set[int]:
    """
    Return physical CPU-package IDs visible to the node.

    Linux CPU-topology files are preferred because they provide a direct,
    bounded mapping from each logical CPU to its physical package.  ``/proc``
    is used only as a compatibility fallback.  Failure is explicit: silently
    assuming one or two sockets could make incomplete RAPL coverage appear
    valid.
    """
    package_ids: set[int] = set()
    topology_pattern = os.path.join(
        cpu_root,
        "cpu[0-9]*",
        "topology",
        "physical_package_id",
    )
    for package_id_path in glob.glob(topology_pattern):
        try:
            package_id = int(_read_text(package_id_path))
        except (OSError, ValueError):
            continue
        if package_id >= 0:
            package_ids.add(package_id)
    if package_ids:
        return package_ids

    current_package_id: int | None = None
    try:
        with open(cpuinfo_path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    if current_package_id is not None:
                        package_ids.add(current_package_id)
                    current_package_id = None
                    continue
                if line.startswith("physical id") and ":" in line:
                    try:
                        current_package_id = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        current_package_id = None
        if current_package_id is not None:
            package_ids.add(current_package_id)
    except OSError:
        pass

    if not package_ids:
        raise RuntimeError(
            "Physical CPU-package topology could not be determined from "
            f"{cpu_root} or {cpuinfo_path}; complete RAPL coverage cannot be "
            "verified without assuming a socket count."
        )
    return package_ids


def _discover_rapl_domains(root: str = _RAPL_SYSFS_ROOT) -> list[_RaplDomain]:
    """
    Return readable CPU-package and DRAM RAPL domains without double counting.

    Linux may expose the same powercap tree through more than one symlinked
    pathname.  Domains are therefore deduplicated by real path.

    Discovery is deliberately bounded to package directories and their
    immediate children.  Recursive filesystem walks must not be used in sysfs:
    powercap domains contain ``device`` and ``subsystem`` symlinks that point
    back into the device/class trees, so a recursive glob can follow cycles and
    spend an unbounded amount of time constructing duplicate paths.

    The two layouts handled here are:

    * ``<root>/intel-rapl:N`` and ``<root>/intel-rapl:N/intel-rapl:N:M``
    * ``<root>/intel-rapl/intel-rapl:N`` and its immediate children

    Only ``package-*`` domains contribute to CPU energy; immediate child
    ``dram`` domains contribute to RAM energy.  Core/uncore child domains are
    not summed because their energy is already represented by the package
    counter.
    """
    domains: list[_RaplDomain] = []
    seen: set[str] = set()

    def add_domain(
        domain_dir: str,
        expected_kind: str,
        parent_package_id: int | None = None,
    ) -> _RaplDomain | None:
        candidate = os.path.join(domain_dir, "energy_uj")
        real_energy_path = os.path.realpath(candidate)
        if real_energy_path in seen:
            return next(
                domain
                for domain in domains
                if domain.energy_path == real_energy_path
            )

        domain_dir = os.path.dirname(real_energy_path)
        try:
            name = _read_text(os.path.join(domain_dir, "name")).lower()
            # Confirm that the counter is readable now, rather than failing only
            # after a long training phase.
            int(_read_text(real_energy_path))
            max_range_path = os.path.join(domain_dir, "max_energy_range_uj")
            max_range = (
                int(_read_text(max_range_path))
                if os.path.isfile(max_range_path)
                else 0
            )
        except (OSError, ValueError):
            return None

        if expected_kind == "dram" and name == "dram":
            kind = "dram"
            if parent_package_id is None:
                return None
            package_id = parent_package_id
        elif (
            expected_kind == "package"
            and (name == "package" or name.startswith("package-"))
        ):
            kind = "package"
            package_id = _parse_rapl_package_id(name)
            if package_id is None:
                return None
        else:
            return None

        seen.add(real_energy_path)
        domain = _RaplDomain(
            energy_path=real_energy_path,
            name=name,
            kind=kind,
            max_energy_range_uj=max_range,
            package_id=package_id,
        )
        domains.append(domain)
        return domain

    # Some kernels expose package directories directly below the powercap class;
    # this node exposes them below an additional ``intel-rapl`` directory.
    containers = [root, os.path.join(root, "intel-rapl")]
    package_dirs: set[str] = set()
    for container in containers:
        for path in glob.glob(os.path.join(container, "intel-rapl:*")):
            basename = os.path.basename(path)
            suffix = basename.removeprefix("intel-rapl:")
            if suffix.isdigit() and os.path.isdir(path):
                package_dirs.add(path)

    for package_dir in sorted(package_dirs):
        package_domain = add_domain(package_dir, "package")
        if package_domain is None:
            continue

        # Inspect exactly one level below the package.  Do not descend through
        # the package's ``device`` or ``subsystem`` symlinks.
        for child_dir in sorted(
            glob.glob(os.path.join(package_dir, "intel-rapl:*:*"))
        ):
            basename = os.path.basename(child_dir)
            suffix = basename.removeprefix("intel-rapl:")
            parts = suffix.split(":")
            if (
                len(parts) == 2
                and all(part.isdigit() for part in parts)
                and os.path.isdir(child_dir)
            ):
                add_domain(
                    child_dir,
                    "dram",
                    parent_package_id=package_domain.package_id,
                )

    return domains


def _validate_rapl_coverage(
    domains: list[_RaplDomain],
    cpu_package_ids: set[int],
) -> tuple[bool, str]:
    """
    Verify one package counter per physical package and DRAM coverage for each.

    More than one distinct DRAM domain for a package is accepted and summed;
    some platforms partition memory energy into multiple non-overlapping
    domains.  Duplicate package counters are rejected because summing them
    would double count CPU-package energy.
    """
    if not cpu_package_ids:
        return False, "CPU topology reported no physical package IDs."

    package_counts: dict[int, int] = {}
    dram_counts: dict[int, int] = {}
    for domain in domains:
        if domain.kind == "package":
            target = package_counts
        elif domain.kind == "dram":
            target = dram_counts
        else:
            return False, f"Unsupported RAPL domain kind={domain.kind!r}."
        target[domain.package_id] = target.get(domain.package_id, 0) + 1

    expected = set(cpu_package_ids)
    measured_packages = set(package_counts)
    dram_packages = set(dram_counts)
    duplicate_package_ids = sorted(
        package_id
        for package_id, count in package_counts.items()
        if count != 1
    )
    missing_packages = sorted(expected - measured_packages)
    unexpected_packages = sorted(measured_packages - expected)
    missing_dram = sorted(expected - dram_packages)
    unexpected_dram = sorted(dram_packages - expected)

    valid = not (
        duplicate_package_ids
        or missing_packages
        or unexpected_packages
        or missing_dram
        or unexpected_dram
    )
    if valid:
        return True, ""

    details = [
        f"CPU topology package IDs={sorted(expected)}",
        f"RAPL package IDs={sorted(measured_packages)}",
        f"DRAM-covered package IDs={sorted(dram_packages)}",
    ]
    if duplicate_package_ids:
        details.append(
            f"duplicate RAPL package counter IDs={duplicate_package_ids}"
        )
    if missing_packages:
        details.append(f"missing RAPL package IDs={missing_packages}")
    if unexpected_packages:
        details.append(f"unexpected RAPL package IDs={unexpected_packages}")
    if missing_dram:
        details.append(f"packages without a readable DRAM domain={missing_dram}")
    if unexpected_dram:
        details.append(
            f"DRAM domains not mapped to visible CPU packages={unexpected_dram}"
        )
    return False, "; ".join(details) + "."


def _read_rapl_snapshot(domains: list[_RaplDomain]) -> dict[str, int]:
    """Read all selected RAPL counters as close together as practicable."""
    snapshot: dict[str, int] = {}
    for domain in domains:
        snapshot[domain.energy_path] = int(_read_text(domain.energy_path))
    return snapshot


def _counter_delta(start: int, end: int, maximum: int) -> int:
    """Return a cumulative-counter delta, accounting for one wrap."""
    if end >= start:
        return end - start
    if maximum > 0:
        return (maximum - start) + end
    raise RuntimeError(
        "A RAPL counter wrapped, but max_energy_range_uj was unavailable."
    )


def _rapl_energy_j(
    domains: list[_RaplDomain],
    start: dict[str, int],
    end: dict[str, int],
    kind: str,
    intermediate: list[dict[str, int]] | None = None,
) -> float:
    """
    Sum energy differences for one RAPL domain type.

    Intermediate snapshots make the calculation robust to repeated counter
    wraparound during long fine-tuning phases.  At the profiler's polling
    interval, a valid RAPL counter cannot wrap more than once between adjacent
    snapshots.
    """
    total_uj = 0
    snapshots = [start, *(intermediate or []), end]
    for domain in domains:
        if domain.kind != kind:
            continue
        values = [
            snapshot[domain.energy_path]
            for snapshot in snapshots
            if domain.energy_path in snapshot
        ]
        if len(values) < 2:
            raise RuntimeError(
                f"RAPL domain disappeared during measurement: {domain.energy_path}"
            )
        total_uj += sum(
            _counter_delta(left, right, domain.max_energy_range_uj)
            for left, right in zip(values, values[1:])
        )
    return total_uj / 1_000_000.0


# ---------------------------------------------------------------------------
# GPU helpers via pynvml
# ---------------------------------------------------------------------------

def _has_gpu() -> bool:
    # PyTorch (fastest path, most common env)
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    # NVML direct — works in TF-only environments (e.g. tf215env / tf28env)
    # where PyTorch is not installed but an NVIDIA GPU is present.
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count > 0
    except (RuntimeError, OSError):
        pass
    return False


def _cuda_visible_device_tokens() -> list[str] | None:
    """
    Return CUDA_VISIBLE_DEVICES tokens, or ``None`` when no visibility mask exists.

    An explicit empty value or ``-1`` means that CUDA exposes no GPU.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw or raw in ("-1", "NoDevFiles"):
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _nvml_handle_for_logical_index(nvml, logical_index: int):
    """
    Resolve a CUDA logical index to the corresponding NVML device handle.

    Numeric and UUID-based CUDA_VISIBLE_DEVICES masks are both supported.
    Invalid logical indices raise instead of silently selecting another card.
    """
    if not isinstance(logical_index, int) or isinstance(logical_index, bool):
        raise TypeError("gpu_index must be an integer CUDA logical device index.")
    if logical_index < 0:
        raise ValueError("gpu_index must be non-negative.")

    visible = _cuda_visible_device_tokens()
    if visible is None:
        return nvml.nvmlDeviceGetHandleByIndex(logical_index)
    if logical_index >= len(visible):
        raise RuntimeError(
            f"CUDA logical GPU {logical_index} is unavailable: "
            f"CUDA_VISIBLE_DEVICES exposes {len(visible)} device(s)."
        )

    token = visible[logical_index]
    if token.lstrip("+").isdigit():
        return nvml.nvmlDeviceGetHandleByIndex(int(token))

    # CUDA accepts full or abbreviated GPU UUIDs. Try the direct NVML lookup,
    # then resolve an unambiguous abbreviation against the two physical cards.
    try:
        return nvml.nvmlDeviceGetHandleByUUID(token.encode("utf-8"))
    except Exception as direct_error:
        matches = []
        for physical_index in range(int(nvml.nvmlDeviceGetCount())):
            handle = nvml.nvmlDeviceGetHandleByIndex(physical_index)
            uuid_value = nvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid_value, bytes):
                uuid_value = uuid_value.decode("utf-8")
            if str(uuid_value).startswith(token):
                matches.append(handle)
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"Cannot resolve CUDA_VISIBLE_DEVICES token {token!r} uniquely "
            "through NVML."
        ) from direct_error


def _resolve_physical_gpu_index(logical_index: int) -> int:
    """
    Convert a PyTorch logical GPU index to the physical NVML device index.

    On HPC clusters, SLURM sets CUDA_VISIBLE_DEVICES to the physical GPU IDs
    allocated to the job (e.g. "2,3"). PyTorch re-numbers these as 0, 1, …
    but pynvml always uses physical indices — nvmlDeviceGetHandleByIndex(0)
    reads GPU 0 regardless of CUDA_VISIBLE_DEVICES.  Without this translation
    the profiler would silently measure the wrong GPU on any multi-GPU node.
    """
    nvml = _nvml_init()
    if nvml is None:
        raise RuntimeError("NVML is unavailable; the selected GPU cannot be resolved.")
    try:
        handle = _nvml_handle_for_logical_index(nvml, logical_index)
        return int(nvml.nvmlDeviceGetIndex(handle))
    finally:
        _nvml_shutdown(nvml)


def _nvml_init():
    try:
        import pynvml
        pynvml.nvmlInit()
        return pynvml
    except (ImportError, RuntimeError, OSError):
        return None


def _nvml_shutdown(pynvml_mod):
    try:
        pynvml_mod.nvmlShutdown()
    except (RuntimeError, OSError):
        pass


def _get_nvml_device_count() -> int | None:
    """Return the number of physical NVIDIA devices visible to NVML."""
    nvml = _nvml_init()
    if nvml is None:
        return None
    try:
        return int(nvml.nvmlDeviceGetCount())
    except (RuntimeError, OSError, TypeError):
        return None
    finally:
        _nvml_shutdown(nvml)


def _get_gpu_power_w(gpu_index: int = 0) -> float:
    """Return current GPU power draw in Watts (0.0 if unavailable)."""
    nvml = _nvml_init()
    if nvml is None:
        return 0.0
    try:
        h = _nvml_handle_for_logical_index(nvml, gpu_index)
        pw = nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
        return pw
    except (RuntimeError, OSError, ZeroDivisionError):
        return 0.0
    finally:
        _nvml_shutdown(nvml)


def _get_gpu_total_energy_mj(gpu_index: int = 0) -> int | None:
    """
    Return NVML's cumulative device energy in millijoules.

    The counter is available on supported Volta-or-newer NVIDIA devices.  ``None``
    means that the driver/device does not expose it, in which case the profiler
    uses timestamped device-power integration.
    """
    nvml = _nvml_init()
    if nvml is None:
        return None
    try:
        handle = _nvml_handle_for_logical_index(nvml, gpu_index)
        return int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    except (RuntimeError, OSError, TypeError):
        return None
    finally:
        _nvml_shutdown(nvml)


def _get_process_gpu_memory_mb(pid: int, gpu_index: int = 0) -> float:
    """Return GPU memory used by a specific process (MB)."""
    nvml = _nvml_init()
    if nvml is None:
        return 0.0
    try:
        h = _nvml_handle_for_logical_index(nvml, gpu_index)
        procs = nvml.nvmlDeviceGetComputeRunningProcesses(h)
        for p in procs:
            if p.pid == pid:
                return p.usedGpuMemory / 1024 ** 2
    except (RuntimeError, OSError, ZeroDivisionError, AttributeError):
        pass
    finally:
        _nvml_shutdown(nvml)
    return 0.0


def _get_cpu_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def _owned_process_ids(root_pid: int) -> set[int]:
    """Return the benchmark process ID and all currently existing descendants."""
    owned = {root_pid}
    try:
        owned.update(
            child.pid
            for child in psutil.Process(root_pid).children(recursive=True)
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
        pass
    return owned


def _inspect_node_gpu_processes(
    selected_gpu_index: int = 0,
    root_pid: int | None = None,
) -> tuple[int, int]:
    """
    Inspect compute processes on both node GPUs.

    Returns ``(foreign_process_count, owned_processes_on_other_gpus)``.  Foreign
    GPU activity anywhere on the node invalidates package/DRAM attribution,
    while benchmark-owned activity on another card violates the single-GPU
    execution boundary.

    NVML failures raise: returning zero on an unavailable interface would
    incorrectly certify exclusivity.
    """
    nvml = _nvml_init()
    if nvml is None:
        raise RuntimeError("NVML is unavailable; GPU-process exclusivity is unknown.")
    root_pid = root_pid or os.getpid()
    try:
        device_count = int(nvml.nvmlDeviceGetCount())
        selected_handle = _nvml_handle_for_logical_index(nvml, selected_gpu_index)
        selected_physical_index = int(nvml.nvmlDeviceGetIndex(selected_handle))
        owned_pids = _owned_process_ids(root_pid)
        foreign_pids: set[int] = set()
        owned_on_other_pids: set[int] = set()

        for physical_index in range(device_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(physical_index)
            processes = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for process in processes:
                pid = int(process.pid)
                if pid not in owned_pids:
                    foreign_pids.add(pid)
                elif physical_index != selected_physical_index:
                    owned_on_other_pids.add(pid)

        return len(foreign_pids), len(owned_on_other_pids)
    finally:
        _nvml_shutdown(nvml)


def _get_process_pss_mb() -> float:
    """Return process PSS (Proportional Set Size) in MB. Falls back to RSS."""
    try:
        return psutil.Process(os.getpid()).memory_full_info().pss / 1024 ** 2
    except Exception:
        try:
            return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
        except (psutil.NoSuchProcess, psutil.AccessDenied, ZeroDivisionError, AttributeError):
            return 0.0


def _get_gpu_p_state(gpu_index: int = 0) -> str:
    """Return current GPU performance state as string (e.g. 'P0'). '' if unavailable."""
    nvml = _nvml_init()
    if nvml is None:
        return ""
    try:
        h = _nvml_handle_for_logical_index(nvml, gpu_index)
        p_state = nvml.nvmlDeviceGetPerformanceState(h)
        return f"P{int(p_state)}"
    except (RuntimeError, OSError, TypeError, ValueError):
        return ""
    finally:
        _nvml_shutdown(nvml)


def _synchronize_selected_gpu(gpu_index: int) -> None:
    """
    Wait for queued work in already-loaded CUDA frameworks to complete.

    Importing a framework solely for synchronization can create another CUDA
    context, so only frameworks already present in ``sys.modules`` are touched.
    PyTorch synchronization is explicitly device-scoped. TensorFlow is normally
    synchronous in eager mode; ``async_wait`` also covers explicit asynchronous
    execution when that API is available.
    """
    errors: list[str] = []
    synchronized = False

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize(device=gpu_index)
                synchronized = True
        except Exception as exc:
            errors.append(f"PyTorch: {exc}")

    tensorflow = sys.modules.get("tensorflow")
    if tensorflow is not None:
        try:
            async_wait = getattr(
                getattr(tensorflow, "experimental", None),
                "async_wait",
                None,
            )
            if callable(async_wait):
                async_wait()
            else:
                from tensorflow.python.eager import context as tf_context
                tf_context.async_wait()
            synchronized = True
        except Exception as exc:
            errors.append(f"TensorFlow: {exc}")

    if errors:
        raise RuntimeError(
            "CUDA synchronization failed at the energy-measurement boundary ("
            + "; ".join(errors)
            + ")."
        )
    if not synchronized:
        logger.debug(
            "No loaded framework exposed a synchronization hook; TimedBlock "
            "assumes the benchmark call itself is synchronous."
        )


# GPU thermal-settling constants (used by wait_for_gpu_cooldown).


def wait_for_gpu_cooldown(gpu_index: int = 0) -> None:
    """Enforce a thermal settling period between training and the inference baseline.

    Energy-measuring experiments call this after fine-tuning and before the
    inference baseline so the idle-power baseline is not contaminated by a GPU
    still hot (200–280 W) from training.

    Protocol:
      1. Sleep GPU_COOLDOWN_MANDATORY_S (60 s) unconditionally.
      2. Read GPU temperature.  If ≤ GPU_COOLDOWN_TARGET_C (45 °C), proceed.
      3. Otherwise sleep GPU_COOLDOWN_RETRY_S (30 s) and re-check (no hard
         timeout — thermal settling must complete before the baseline).

    Silently returns if pynvml is unavailable.
    """
    pynvml = _nvml_init()
    if pynvml is None:
        logger.debug("GPU cooldown skipped (pynvml unavailable)")
        return
    try:
        handle = _nvml_handle_for_logical_index(pynvml, gpu_index)

        logger.info(
            f"GPU cooldown: mandatory {GPU_COOLDOWN_MANDATORY_S}s sleep before "
            "temperature check."
        )
        time.sleep(GPU_COOLDOWN_MANDATORY_S)

        while True:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            if temp <= GPU_COOLDOWN_TARGET_C:
                logger.info(
                    f"GPU cooldown complete: {temp}°C ≤ {GPU_COOLDOWN_TARGET_C}°C — "
                    "proceeding to inference baseline."
                )
                return
            logger.info(
                f"GPU cooldown: {temp}°C > {GPU_COOLDOWN_TARGET_C}°C — "
                f"waiting {GPU_COOLDOWN_RETRY_S}s before re-check."
            )
            time.sleep(GPU_COOLDOWN_RETRY_S)
    except Exception as exc:
        logger.debug(f"GPU cooldown skipped (pynvml error): {exc}")
    finally:
        _nvml_shutdown(pynvml)


def _measure_disk_write_speed(target_dir: str = "", duration_s: float = 0.5) -> float:
    """
    Measure disk write speed by writing a temp file for ~duration_s seconds.
    Returns MB/s write speed. Returns 0.0 if measurement fails.
    """
    chunk = b"\x00" * (256 * 1024)  # 256 KB per write call
    total_bytes = 0
    try:
        fd, tmp_path = tempfile.mkstemp(dir=target_dir or None, prefix="_bm_disktest_")
        try:
            t_start = time.perf_counter()
            t_end_target = t_start + duration_s
            with os.fdopen(fd, "wb") as fh:
                while time.perf_counter() < t_end_target:
                    fh.write(chunk)
                    total_bytes += len(chunk)
            elapsed = time.perf_counter() - t_start
            return total_bytes / elapsed / (1024 * 1024) if elapsed > 0 else 0.0
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except (OSError, ZeroDivisionError):
        return 0.0


# Social cost of carbon (SCC) lookup — Ricke et al. 2018, Nature Climate Change.
# Model: SSP2, BHM short-run, 2% PRTP, η=1.5, RCP6.0, median. Units: USD/tonne CO₂.
_SCC_USD_PER_TONNE_CO2: dict = {
    "DEU": 52.0, "USA": 48.0, "CHN": 24.0, "IND": 86.0,
    "GBR": 60.0, "FRA": 65.0, "JPN": 69.0, "BRA": 41.0,
}
_SCC_GLOBAL_MEDIAN: float = 51.0   # Ricke et al. 2018 global median

# Electricity Maps zone (ISO2 prefix, before any "-" subregion) → ISO3 country
# code, so the SCC lookup above follows wherever the experiment's carbon
# intensity was actually measured instead of being hardcoded to Germany.
_ZONE_PREFIX_TO_ISO3: dict = {
    "DE": "DEU", "US": "USA", "CN": "CHN", "IN": "IND",
    "GB": "GBR", "FR": "FRA", "JP": "JPN", "BR": "BRA",
}


def _country_code_from_zone(zone: str) -> str:
    """Map an Electricity Maps zone (e.g. "DE", "US-CAL-CISO") to an ISO3 code.

    Falls back to "DEU" only when the zone's ISO2 prefix isn't in the lookup
    table (i.e. we have no SCC estimate for that country either way, so
    _SCC_USD_PER_TONNE_CO2.get() would fall back to the global median anyway).
    """
    prefix = (zone or "").split("-")[0].upper()
    return _ZONE_PREFIX_TO_ISO3.get(prefix, "DEU")


def _compute_social_cost_carbon_usd(co2_eq_g: float, country_code: str = "DEU") -> float:
    """
    Compute social cost of CO₂ in USD.

    co2_eq_g: total CO₂-equivalent in grams
    country_code: ISO3 code (DEU for Germany, our default location)
    Returns: social cost in USD (Ricke et al. 2018 median SCC)
    """
    scc = _SCC_USD_PER_TONNE_CO2.get(country_code.upper(), _SCC_GLOBAL_MEDIAN)
    return (co2_eq_g / 1e6) * scc   # g → tonnes × USD/tonne


# ---------------------------------------------------------------------------
# Background monitoring thread
# ---------------------------------------------------------------------------

class _BackgroundMonitor(threading.Thread):
    """
    Timestamped diagnostic sampler for GPU power/utilization and process memory.

    Device power samples are used for GPU energy only if NVML's cumulative
    device-energy counter is unavailable.  No per-process power attribution is
    attempted.  The same process-memory/utilization fields are retained for
    compatibility with existing benchmark logs.
    """

    def __init__(
        self,
        interval_s: float = MONITOR_INTERVAL_S,
        gpu_index: int = 0,
        pid: int = 0,
        rapl_domains: list[_RaplDomain] | None = None,
    ):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self.pid = pid or os.getpid()
        self.rapl_domains = list(rapl_domains or [])
        # NOTE: must NOT be named _stop – that shadows threading.Thread._stop()
        self._halt = threading.Event()
        self._ready = threading.Event()
        self.gpu_sample_timestamps_s: list = []
        self.gpu_power_samples: list = []             # total device power (W)
        self.gpu_memory_samples: list = []            # this process's GPU memory (MB)
        self.cpu_memory_samples: list = []
        self.gpu_sm_util_samples: list = []           # whole-device SM util (%)
        self.rapl_sample_timestamps_s: list = []
        self.rapl_samples: list[dict[str, int]] = []
        self.foreign_gpu_process_counts: list[int] = []
        self.owned_processes_on_other_gpu_counts: list[int] = []
        self.gpu_process_inspection_errors: int = 0

    def run(self):
        nvml = _nvml_init()
        handle = None
        all_handles: list = []
        selected_physical_index: int | None = None
        if nvml is not None:
            try:
                handle = _nvml_handle_for_logical_index(nvml, self.gpu_index)
                selected_physical_index = int(nvml.nvmlDeviceGetIndex(handle))
                all_handles = [
                    nvml.nvmlDeviceGetHandleByIndex(index)
                    for index in range(int(nvml.nvmlDeviceGetCount()))
                ]
            except (RuntimeError, OSError, TypeError):
                _nvml_shutdown(nvml)
                nvml = None

        next_tick = time.perf_counter()
        while not self._halt.is_set():
            power_w = 0.0
            mem_mb = 0.0
            sm_util = 0.0
            foreign_count = 0
            owned_other_count = 0

            if nvml is not None and handle is not None:
                try:
                    power_w = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    try:
                        procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        proc_mem = {p.pid: p.usedGpuMemory for p in procs}
                        our_mem_bytes = proc_mem.get(self.pid, 0)
                    except (RuntimeError, OSError, AttributeError):
                        our_mem_bytes = 0
                    mem_mb = our_mem_bytes / 1024 ** 2

                    try:
                        sm_util = float(
                            nvml.nvmlDeviceGetUtilizationRates(handle).gpu
                        )
                    except (RuntimeError, OSError, TypeError, ValueError, AttributeError):
                        sm_util = 0.0
                except (RuntimeError, OSError, ZeroDivisionError, TypeError):
                    power_w = mem_mb = sm_util = 0.0

                try:
                    owned_pids = _owned_process_ids(self.pid)
                    foreign_pids: set[int] = set()
                    owned_other_pids: set[int] = set()
                    for physical_index, device_handle in enumerate(all_handles):
                        processes = nvml.nvmlDeviceGetComputeRunningProcesses(
                            device_handle
                        )
                        for process in processes:
                            process_pid = int(process.pid)
                            if process_pid not in owned_pids:
                                foreign_pids.add(process_pid)
                            elif physical_index != selected_physical_index:
                                owned_other_pids.add(process_pid)
                    foreign_count = len(foreign_pids)
                    owned_other_count = len(owned_other_pids)
                except (RuntimeError, OSError, TypeError, AttributeError):
                    self.gpu_process_inspection_errors += 1

            sample_time = time.perf_counter()
            self.gpu_sample_timestamps_s.append(sample_time)
            self.gpu_power_samples.append(power_w)
            self.gpu_memory_samples.append(mem_mb)
            self.gpu_sm_util_samples.append(sm_util)
            self.foreign_gpu_process_counts.append(foreign_count)
            self.owned_processes_on_other_gpu_counts.append(owned_other_count)
            self.cpu_memory_samples.append(_get_cpu_memory_mb())
            if self.rapl_domains:
                try:
                    rapl_sample = _read_rapl_snapshot(self.rapl_domains)
                except (OSError, ValueError):
                    rapl_sample = {}
                self.rapl_sample_timestamps_s.append(time.perf_counter())
                self.rapl_samples.append(rapl_sample)
            self._ready.set()

            next_tick += self.interval_s
            self._halt.wait(max(0.0, next_tick - time.perf_counter()))

        if nvml is not None:
            _nvml_shutdown(nvml)

    def stop(self):
        self._halt.set()
        self.join(timeout=3.0)

    def wait_until_ready(self, timeout_s: float = 3.0) -> bool:
        return self._ready.wait(timeout_s)


def _windowed_samples(
    monitor: _BackgroundMonitor | None,
    values: list,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """Return samples whose timestamps fall inside the timed interval."""
    if monitor is None or not monitor.gpu_sample_timestamps_s or not values:
        return np.array([], dtype=float)
    n = min(len(monitor.gpu_sample_timestamps_s), len(values))
    pairs = [
        float(values[i])
        for i in range(n)
        if start_s <= monitor.gpu_sample_timestamps_s[i] <= end_s
    ]
    return np.asarray(pairs, dtype=float)


def _integrate_gpu_power_j(
    monitor: _BackgroundMonitor | None,
    start_s: float,
    end_s: float,
) -> tuple[float, int]:
    """
    Trapezoidally integrate valid NVML device-power samples over the interval.

    Samples immediately outside the boundary permit linear interpolation at the
    exact start and stop times.  The nearest reading is held constant only when
    one side lacks an exterior sample.
    """
    if (
        monitor is None
        or end_s <= start_s
        or not monitor.gpu_sample_timestamps_s
        or not monitor.gpu_power_samples
    ):
        return 0.0, 0

    n = min(
        len(monitor.gpu_sample_timestamps_s),
        len(monitor.gpu_power_samples),
    )
    valid = [
        (
            float(monitor.gpu_sample_timestamps_s[i]),
            float(monitor.gpu_power_samples[i]),
        )
        for i in range(n)
        if math.isfinite(monitor.gpu_power_samples[i])
        and monitor.gpu_power_samples[i] > 0.0
    ]
    if not valid:
        return 0.0, 0

    times = [x[0] for x in valid]
    powers = [x[1] for x in valid]
    if len(times) == 1:
        return powers[0] * (end_s - start_s), 1

    def interpolate_power(boundary_s: float) -> float:
        if boundary_s <= times[0]:
            return powers[0]
        if boundary_s >= times[-1]:
            return powers[-1]
        for index in range(1, len(times)):
            if times[index] >= boundary_s:
                left_t, right_t = times[index - 1], times[index]
                left_p, right_p = powers[index - 1], powers[index]
                fraction = (boundary_s - left_t) / (right_t - left_t)
                return left_p + fraction * (right_p - left_p)
        return powers[-1]

    points = [(start_s, interpolate_power(start_s))]
    points.extend(
        (sample_t, sample_p)
        for sample_t, sample_p in valid
        if start_s < sample_t < end_s
    )
    points.append((end_s, interpolate_power(end_s)))
    energy_j = sum(
        0.5 * (left_p + right_p) * (right_t - left_t)
        for (left_t, left_p), (right_t, right_p)
        in zip(points, points[1:])
    )
    return max(energy_j, 0.0), len(valid)


# ---------------------------------------------------------------------------
# ProfilerResult – all measured values
# ---------------------------------------------------------------------------

@dataclass
class ProfilerResult:
    """Complete efficiency and sustainability record for one timed block.

    ENERGY COMPOSITION
    ==================
    E_IT (IT total energy) = E_GPU + E_CPU + E_RAM  (sum of hardware domains, in mJ)

    Energy per image is computed as:
        E_i  = E_IT / n_images   (for inference, i.e. E_i = IT energy ÷ N_test)
        E_ft = E_IT / n_images   (for fine-tuning, i.e. E_ft = IT energy ÷ N_support)

    Facility energy:
        E_facility = E_IT × PUE   (where PUE = 1.3 for this benchmark)

    All energy fields (*_energy_mj) are in millijoules. All energies represent
    complete start-to-end hardware-domain measurements with no per-process
    attribution; measurement accuracy depends on the node and GPU being
    exclusively allocated to this workload.
    """

    # ── Identity ──────────────────────────────────────────────────
    phase: str = ""               # "inference" | "training" | "finetuning"
    n_images: int = 0

    # ── Time ──────────────────────────────────────────────────────
    wall_time_s: float = 0.0
    time_per_image_s: float = 0.0

    # ── GPU memory (all figures are per-process / this PID only) ─
    peak_gpu_memory_mb: float = 0.0      # peak memory allocated by this process
    mean_gpu_memory_mb: float = 0.0      # mean memory allocated by this process
    process_gpu_memory_mb: float = 0.0   # point-in-time snapshot at run end (this process)

    # ── CPU / RAM memory ──────────────────────────────────────────
    peak_cpu_memory_mb: float = 0.0
    mean_cpu_memory_mb: float = 0.0

    # ── Power (Watts) ─────────────────────────────────────────────
    cpu_power_w: float = 0.0             # mean CPU-package power over the interval
    gpu_power_w: float = 0.0             # mean selected-GPU device power
    ram_power_w: float = 0.0             # mean RAPL DRAM-domain power
    baseline_gpu_power_w: float = 0.0    # idle GPU power measured before experiment (diagnostics only; not subtracted)

    # ── Component energy (mJ, exclusive hardware domains) ───────
    # E_IT = E_GPU + E_CPU + E_RAM
    cpu_energy_mj: float = 0.0           # complete RAPL package-domain energy (E_CPU)
    gpu_energy_mj: float = 0.0           # complete selected-GPU device energy (E_GPU)
    ram_energy_mj: float = 0.0           # complete RAPL DRAM-domain energy (E_RAM)

    # ── Measurement provenance and validity ──────────────────────
    energy_measurement_scope: str = "exclusive hardware domains"
    cpu_energy_source: str = "RAPL package counters"
    gpu_energy_source: str = ""
    ram_energy_source: str = "RAPL DRAM counters"
    rapl_package_domain_count: int = 0
    rapl_dram_domain_count: int = 0
    cpu_topology_package_count: int = 0
    cpu_topology_package_ids: str = ""
    rapl_package_ids: str = ""
    rapl_dram_package_ids: str = ""
    rapl_sample_error_count: int = 0
    gpu_power_sample_count: int = 0
    competing_gpu_process_count: int = 0
    nonselected_gpu_process_count: int = 0
    gpu_process_inspection_error_count: int = 0
    node_gpu_count: int = 0
    selected_gpu_logical_index: int = 0
    selected_gpu_physical_index: int = 0
    cpu_measurement_duration_s: float = 0.0
    gpu_measurement_duration_s: float = 0.0
    ram_measurement_duration_s: float = 0.0
    measurement_valid: bool = True

    # ── IT total energy (mJ) ───────────────────────────────────────
    # E_IT = E_GPU + E_CPU + E_RAM (in mJ)
    # Per-image: E_i = E_IT / n_images
    it_energy_mj: float = 0.0            # E_IT = cpu_energy_mj + gpu_energy_mj + ram_energy_mj (IT equipment only)

    # ── Power Usage Effectiveness (PUE) ───────────────────────────
    # PUE = Total Facility Power / IT Equipment Power; fixed at 1.3 for every result.
    pue: float = FIXED_PUE
    total_facility_energy_mj: float = 0.0  # it_energy × pue (in mJ)
    power_usage_effectiveness: float = FIXED_PUE  # = pue (reported explicitly)

    # ── Carbon emissions (Bielefeld, Germany) ─────────────────────
    co2_location: str = LOCATION_LABEL
    carbon_intensity_g_per_kwh: float = GERMANY_CARBON_INTENSITY_G_PER_KWH
    carbon_intensity_source: str = ""            # how CI was resolved: API / timeline / static fallback / pre-supplied
    carbon_intensity_datetime: str = ""          # ISO validity time of the CI reading (API "datetime" or timeline entry); "" if unknown (static/override)
    carbon_intensity_fetched_at: str = ""        # ISO wall-clock time the CI value was resolved at run start
    co2_eq_g: float = 0.0               # gCO₂eq = total_facility_energy × intensity
    co2_eq_g_per_image: float = 0.0     # co2 / n_images

    # ── Throughput ────────────────────────────────────────────────
    throughput_img_per_s: float = 0.0
    throughput_img_per_j: float = 0.0   # images per Joule of IT energy

    # ── Process utilisation ───────────────────────────────────────
    process_cpu_percent: float = 0.0     # % of total CPU used by this process
    process_ram_percent: float = 0.0     # % of total RAM used by this process

    # ── Extended resource metrics (pynvml / psutil) ───────────────
    gpu_sm_utilization_pct: float = 0.0  # mean GPU SM utilization (%) during run
    gpu_p_state: str = ""                # GPU performance state at run end (e.g. "P0")
    ram_pss_mb: float = 0.0             # process PSS RAM (MB); fair shared-memory attribution
    cpu_hours: float = 0.0              # actual CPU-time hours (user + system time)
    gpu_hours: float = 0.0              # utilization-weighted GPU hours (SM util × time)
    system_load_avg_5min: float = 0.0   # 5-minute system load average at run end
    disk_write_speed_mbs: float = 0.0   # disk write throughput (MB/s), measured pre-run
    social_cost_carbon_usd: float = 0.0 # social cost of CO₂ in USD (Ricke et al. 2018)

    # ── Model ─────────────────────────────────────────────────────
    model_param_count: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"[{self.phase}] {self.n_images} images in {self.wall_time_s:.2f}s "
            f"| CPU {self.cpu_energy_mj:.3f}mJ "
            f"| GPU {self.gpu_energy_mj:.3f}mJ "
            f"| RAM {self.ram_energy_mj:.3f}mJ "
            f"| PUE={self.pue:.2f} "
            f"| CO₂={self.co2_eq_g:.3f}gCO₂eq [{self.co2_location}]"
        )


# ---------------------------------------------------------------------------
# EfficiencyProfiler – configuration object
# ---------------------------------------------------------------------------

class EfficiencyProfiler:
    """
    Holds profiler configuration shared across multiple TimedBlock instances.

    Args:
        device:                    CUDA device descriptor; CPU mode is unsupported
        monitor_interval:          GPU/CPU polling interval (seconds)
        gpu_index:                 GPU device index (0-based)
        pue:                       must equal 1.3 (the benchmark's fixed value)
        baseline_duration_s:       seconds to sample idle GPU power before each experiment
        electricitymaps_api_token: Electricity Maps API token for real-time carbon intensity.
                                   If empty, falls back to the static German grid average
                                   (309 gCO₂eq/kWh, Umweltbundesamt 2025).
                                   Free tier: https://electricitymaps.com/free-tier
        electricitymaps_zone:      Electricity Maps zone code (default "DE" for Germany).
        electricitymaps_proxy:     HTTP/HTTPS proxy URL for reaching the Electricity Maps API
                                   from HPC compute nodes with no direct internet access,
                                   e.g. "http://proxy.hpc.edu:3128". Leave empty to use
                                   system proxy settings (HTTP_PROXY / HTTPS_PROXY env vars).
        carbon_intensity_override: If > 0, use this value (gCO₂eq/kWh) directly and skip
                                   the Electricity Maps API call entirely. Intended for HPC
                                   nodes where neither a proxy nor direct internet is available:
                                   pre-fetch the intensity on the login node with
                                   `curl -H "auth-token: <token>" https://api.electricitymap.org/v3/carbon-intensity/latest?zone=DE`
                                   then pass the value via --electricitymaps_intensity.
        ci_timeline_path:          Path to a carbon intensity timeline CSV produced by
                                   collect_carbon_intensity.py. When set, the entry whose
                                   timestamp is closest to the current time is used,
                                   avoiding any API call. This is the recommended approach
                                   for HPC compute nodes with no internet access: run
                                   collect_carbon_intensity.py on the login node, then
                                   point all compute jobs at the resulting file.
                                   Takes precedence over the Electricity Maps API call but
                                   not over carbon_intensity_override.
    """

    def __init__(
        self,
        device: str = "cuda",
        monitor_interval: float = MONITOR_INTERVAL_S,
        gpu_index: int = 0,
        pue: float = FIXED_PUE,
        baseline_duration_s: float = 2.0,
        electricitymaps_api_token: str = "",
        electricitymaps_zone: str = "DE",
        electricitymaps_proxy: str = "",
        carbon_intensity_override: float = 0.0,
        ci_timeline_path: str = "",
    ):
        device_text = str(device).lower()
        if "cuda" not in device_text and "gpu" not in device_text:
            raise ValueError(
                "This benchmark has no CPU-only models; EfficiencyProfiler "
                "requires a CUDA device."
            )
        if not math.isclose(float(pue), FIXED_PUE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"PUE is fixed at {FIXED_PUE} for every benchmark experiment; "
                f"received {pue!r}."
            )
        if not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0:
            raise ValueError("gpu_index must be a non-negative integer.")
        monitor_interval_value = float(monitor_interval)
        baseline_duration_value = float(baseline_duration_s)
        carbon_override_value = float(carbon_intensity_override)
        if (
            not math.isfinite(monitor_interval_value)
            or monitor_interval_value <= 0.0
        ):
            raise ValueError("monitor_interval must be a positive finite number.")
        if (
            not math.isfinite(baseline_duration_value)
            or baseline_duration_value < 0.0
        ):
            raise ValueError("baseline_duration_s must be a non-negative finite number.")
        if (
            not math.isfinite(carbon_override_value)
            or carbon_override_value < 0.0
        ):
            raise ValueError(
                "carbon_intensity_override must be zero or a positive finite value."
            )
        if device_text.startswith("cuda:"):
            try:
                device_index = int(device_text.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"Invalid CUDA device descriptor: {device!r}.") from exc
            if device_index != gpu_index:
                raise ValueError(
                    f"device={device!r} selects CUDA GPU {device_index}, but "
                    f"gpu_index={gpu_index}. These must identify the same card."
                )
        self.device = device
        self.monitor_interval = monitor_interval_value
        self.gpu_index = gpu_index
        self.pue = FIXED_PUE
        self.baseline_duration_s = baseline_duration_value
        self.electricitymaps_api_token = electricitymaps_api_token
        self.electricitymaps_zone = electricitymaps_zone
        self.electricitymaps_proxy = electricitymaps_proxy
        self.carbon_intensity_override = carbon_override_value
        self.ci_timeline_path = ci_timeline_path


# ---------------------------------------------------------------------------
# TimedBlock – the main public interface
# ---------------------------------------------------------------------------

class TimedBlock:
    """
    Explicit start / stop profiler for a single timed code block.

    Usage:
        tb = TimedBlock(profiler, phase="inference", n_images=100)
        tb.start()          ← measures 2s idle baseline, then begins tracking
        model.predict(...)  ← the code being profiled
        result = tb.stop()  ← returns ProfilerResult
    """

    def __init__(
        self,
        profiler: EfficiencyProfiler,
        phase: str = "inference",
        n_images: int = 0,
    ):
        self.profiler = profiler
        self.phase = phase
        self.n_images = n_images

        # Validate n_images — must be > 0 to avoid division by zero in result calculations
        if self.n_images <= 0:
            logger.warning(
                f"TimedBlock initialized with n_images={self.n_images} for phase '{phase}'. "
                "This will result in per-image metrics being zero. "
                "Ensure n_images is set correctly when creating TimedBlock instances."
            )

        self._monitor: Optional[_BackgroundMonitor] = None
        self._t_start: float = 0.0
        self._rapl_domains: list[_RaplDomain] = []
        self._cpu_package_ids: set[int] = set()
        self._rapl_start: dict[str, int] = {}
        self._rapl_start_counter_time_s: float = 0.0
        self._gpu_energy_start_mj: int | None = None
        self._gpu_start_counter_time_s: float = 0.0
        self._competing_gpu_process_count: int = 0
        self._nonselected_gpu_process_count: int = 0
        self._node_gpu_count: int = 0
        self._selected_gpu_physical_index: int = -1
        self._baseline_gpu_power_w: float = 0.0
        self._live_carbon_intensity: Optional[float] = None  # set if Electricity Maps used
        self._ci_source: str = "static fallback"
        self._ci_reading_datetime: Optional[str] = None      # validity time of the CI reading
        self._ci_fetched_at: str = ""                        # wall-clock time CI was resolved
        self._cpu_times_start: Optional[object] = None   # psutil CpuTimes snapshot at start
        self._disk_write_speed: float = 0.0              # measured once before baseline
        self._started: bool = False
        self._stopped: bool = False

    # ------------------------------------------------------------------
    # Baseline measurement (before experiment)
    # ------------------------------------------------------------------

    def _measure_baseline_gpu_power(self, duration_s: float) -> float:
        """
        Measure total GPU power for `duration_s` seconds BEFORE the experiment starts.
        Stored as baseline_gpu_power_w for diagnostics (shows what was running on the
        GPU before this experiment).  It is not subtracted from gpu_energy_mj:
        the reported value is total selected-device energy over the interval.
        A warning is logged when the baseline exceeds 100 W (GPU not at idle).
        """
        if not _has_gpu():
            return 0.0
        samples = []
        t_end = time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            pw = _get_gpu_power_w(self.profiler.gpu_index)
            if pw > 0:
                samples.append(pw)
            time.sleep(0.1)
        val = float(np.mean(samples)) if samples else 0.0
        logger.debug(f"Baseline GPU power: {val:.1f}W (measured over {duration_s}s)")
        return val

    # ------------------------------------------------------------------
    # start()
    # ------------------------------------------------------------------

    def start(self):
        """
        Begin profiling.

        Steps:
            1. Validate the fixed two-GPU node and selected CUDA device.
            2. Flush selected-GPU state and run garbage collection.
            3. Fetch carbon intensity (override → timeline file → API → static).
            4. Record idle GPU power baseline (diagnostic only).
            5. Verify node-wide GPU exclusivity and complete RAPL coverage.
            6. Start the timestamped diagnostic/power sampler.
            7. Record RAPL and NVML counters at the synchronized boundary.
        """
        if self._started and not self._stopped:
            raise RuntimeError("TimedBlock.start() was called more than once.")
        if self._stopped:
            raise RuntimeError("A completed TimedBlock cannot be restarted.")

        gc.collect()

        # 1. Validate the fixed two-card node and resolve the selected card.
        node_gpu_count = _get_nvml_device_count()
        if node_gpu_count is None:
            raise RuntimeError(
                "NVML is unavailable; the mandatory selected-GPU energy "
                "measurement cannot start."
            )
        self._node_gpu_count = node_gpu_count
        if node_gpu_count != EXPECTED_NODE_GPU_COUNT:
            raise RuntimeError(
                f"The fixed benchmark node must expose {EXPECTED_NODE_GPU_COUNT} "
                f"GPU cards through NVML; found {node_gpu_count}."
            )
        self._selected_gpu_physical_index = _resolve_physical_gpu_index(
            self.profiler.gpu_index
        )

        # 2. Flush the selected GPU only.
        _synchronize_selected_gpu(self.profiler.gpu_index)
        torch = sys.modules.get("torch")
        if torch is not None:
            try:
                torch.cuda.reset_peak_memory_stats(self.profiler.gpu_index)
                with torch.cuda.device(self.profiler.gpu_index):
                    torch.cuda.empty_cache()
            except (RuntimeError, AttributeError):
                pass
        gc.collect()

        # 3. Resolve carbon intensity.  Done before the baseline measurement so
        #    any network or file-read latency is outside the profiled block.
        #
        #    Priority order:
        #      a) carbon_intensity_override  — explicit single-value override
        #      b) CI timeline file           — nearest timestamp entry; avoids
        #                                     API timeout waste on HPC nodes
        #      c) Electricity Maps live API  — falls back to static if unreachable
        #      d) Static fallback            — 309 gCO₂eq/kWh (Germany 2025)
        self._live_carbon_intensity = None
        self._ci_source = "static fallback"
        # Validity time of the CI reading itself (API "datetime" / timeline entry
        # timestamp); None for override/static. Distinct from _ci_fetched_at,
        # the wall-clock time we resolved it (set below).
        self._ci_reading_datetime = None

        if self.profiler.carbon_intensity_override > 0.0:
            # (a) Explicit pre-supplied value — highest priority, no lookup needed
            self._live_carbon_intensity = self.profiler.carbon_intensity_override
            self._ci_source = "pre-supplied (--electricitymaps_intensity)"

        elif self.profiler.ci_timeline_path:
            # (b) Timeline file — use nearest timestamp; skip API entirely so HPC
            #     nodes don't waste time waiting for a connection that will fail
            timeline_result = read_ci_from_timeline(self.profiler.ci_timeline_path)
            if timeline_result is not None:
                ci_val, age_s, entry_ts = timeline_result
                age_str = (f"{age_s/60:.0f} min" if age_s < 7200
                           else f"{age_s/3600:.1f} h")
                self._live_carbon_intensity = ci_val
                self._ci_reading_datetime = entry_ts
                self._ci_source = f"timeline file (entry age: {age_str})"
                if age_s > 7200:
                    logger.warning(
                        f"CI timeline entry is {age_str} old — "
                        f"consider re-running collect_carbon_intensity.py"
                    )
            else:
                logger.warning(
                    f"CI timeline file not found or empty: "
                    f"{self.profiler.ci_timeline_path}. "
                    f"Trying Electricity Maps API."
                )
                if self.profiler.electricitymaps_api_token:
                    self._live_carbon_intensity, self._ci_reading_datetime = fetch_electricitymaps_intensity(
                        token=self.profiler.electricitymaps_api_token,
                        zone=self.profiler.electricitymaps_zone,
                        proxy_url=self.profiler.electricitymaps_proxy,
                    )
                    if self._live_carbon_intensity is not None:
                        self._ci_source = f"Electricity Maps API (zone: {self.profiler.electricitymaps_zone})"

        elif self.profiler.electricitymaps_api_token:
            # (c) Live API — used when no timeline file is present
            self._live_carbon_intensity, self._ci_reading_datetime = fetch_electricitymaps_intensity(
                token=self.profiler.electricitymaps_api_token,
                zone=self.profiler.electricitymaps_zone,
                proxy_url=self.profiler.electricitymaps_proxy,
            )
            if self._live_carbon_intensity is not None:
                self._ci_source = f"Electricity Maps API (zone: {self.profiler.electricitymaps_zone})"

        # Wall-clock time we resolved the CI value (always set), so the record
        # shows both when the reading is valid for and when we looked it up.
        self._ci_fetched_at = datetime.now().astimezone().isoformat()

        # Resolved CI value (None here means stop() will use the static fallback)
        _ci_display = (self._live_carbon_intensity
                       if self._live_carbon_intensity is not None
                       else GERMANY_CARBON_INTENSITY_G_PER_KWH)
        if self._live_carbon_intensity is None:
            self._ci_source = f"static fallback ({GERMANY_CARBON_INTENSITY_G_PER_KWH} gCO₂eq/kWh, Germany 2025)"
        logger.info(f"  Carbon intensity [{self.phase}]: {_ci_display:.1f} gCO₂eq/kWh  |  source: {self._ci_source}")
        print(f"  ⚡ Carbon intensity [{self.phase}]: {_ci_display:.1f} gCO₂eq/kWh  |  {self._ci_source}", flush=True)

        # 3b. Disk write speed — measured once outside the profiled block.
        self._disk_write_speed = _measure_disk_write_speed(duration_s=0.5)
        logger.debug(f"Disk write speed: {self._disk_write_speed:.1f} MB/s")

        # 4. Baseline GPU power is diagnostic only; it is not subtracted.
        if self.profiler.baseline_duration_s > 0:
            self._baseline_gpu_power_w = self._measure_baseline_gpu_power(
                self.profiler.baseline_duration_s
            )

        # 5. Verify node-wide GPU exclusivity before opening the timed boundary.
        (
            self._competing_gpu_process_count,
            self._nonselected_gpu_process_count,
        ) = _inspect_node_gpu_processes(
            selected_gpu_index=self.profiler.gpu_index,
            root_pid=os.getpid(),
        )
        if self._competing_gpu_process_count > 0:
            raise RuntimeError(
                "Foreign GPU compute processes were detected on the node. "
                "They would invalidate selected-GPU and host RAPL attribution."
            )
        if self._nonselected_gpu_process_count > 0:
            raise RuntimeError(
                "The benchmark process is using a non-selected GPU. Every "
                "experiment must use exactly one GPU card."
            )

        # Discover the physical CPU-package topology and the corresponding
        # package/DRAM RAPL domains. Fail before the workload if any visible
        # package lacks either component.
        self._cpu_package_ids = _discover_cpu_package_ids()
        self._rapl_domains = _discover_rapl_domains()
        n_package = sum(d.kind == "package" for d in self._rapl_domains)
        n_dram = sum(d.kind == "dram" for d in self._rapl_domains)
        complete_rapl, rapl_coverage_detail = _validate_rapl_coverage(
            self._rapl_domains,
            self._cpu_package_ids,
        )
        if not complete_rapl:
            raise RuntimeError(
                "Incomplete or inconsistent RAPL coverage: "
                f"{rapl_coverage_detail}"
            )

        # 6. Start timestamped sampling before the timed boundary so the power-
        #    integration fallback has a reading on the left side of the boundary.
        self._monitor = _BackgroundMonitor(
            interval_s=self.profiler.monitor_interval,
            gpu_index=self.profiler.gpu_index,
            pid=os.getpid(),
            rapl_domains=self._rapl_domains,
        )
        self._monitor.start()
        if not self._monitor.wait_until_ready():
            self._monitor.stop()
            raise RuntimeError(
                "The NVML background sampler did not produce an initial sample."
            )

        # Validate the GPU power interface before running the workload.  The
        # cumulative-energy counter may legitimately be unsupported because the
        # timestamped power sampler provides the fallback.
        if _get_gpu_power_w(self.profiler.gpu_index) <= 0.0:
            self._monitor.stop()
            raise RuntimeError(
                "NVML did not provide a positive power reading for the selected "
                "GPU; GPU energy cannot be measured."
            )

        # 7. Capture process-resource diagnostics.
        try:
            self._cpu_times_start = psutil.Process(os.getpid()).cpu_times()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self._cpu_times_start = None

        # 8. Synchronized measurement boundary. CUDA work is complete before
        #    opening the boundary.  Counter reads are deliberately ordered as
        #    RAPL→GPU at start and GPU→RAPL at stop to bracket the workload
        #    symmetrically. Workload wall time starts after the start counters;
        #    component mean powers use their own counter-interval durations.
        try:
            _synchronize_selected_gpu(self.profiler.gpu_index)
            rapl_read_start_s = time.perf_counter()
            self._rapl_start = _read_rapl_snapshot(self._rapl_domains)
            rapl_read_end_s = time.perf_counter()
            self._rapl_start_counter_time_s = 0.5 * (
                rapl_read_start_s + rapl_read_end_s
            )
            gpu_read_start_s = time.perf_counter()
            self._gpu_energy_start_mj = _get_gpu_total_energy_mj(
                self.profiler.gpu_index
            )
            gpu_read_end_s = time.perf_counter()
            self._gpu_start_counter_time_s = 0.5 * (
                gpu_read_start_s + gpu_read_end_s
            )
            self._t_start = time.perf_counter()
            self._started = True
        except (RuntimeError, OSError, ImportError) as e:
            if self._monitor is not None:
                self._monitor.stop()
            raise

    # ------------------------------------------------------------------
    # stop()
    # ------------------------------------------------------------------

    def stop(self) -> ProfilerResult:
        """
        End profiling and return a complete ProfilerResult.

        Returns complete GPU-device, CPU-package, and DRAM-domain energy over
        the timed interval.  These hardware-domain values are attributable to
        the benchmark only under exclusive allocation.
        """
        if not self._started:
            raise RuntimeError("TimedBlock.stop() was called before start().")
        if self._stopped:
            raise RuntimeError("TimedBlock.stop() was called more than once.")

        try:
            _synchronize_selected_gpu(self.profiler.gpu_index)
            t_end = time.perf_counter()

            gpu_read_start_s = time.perf_counter()
            gpu_energy_end_mj = _get_gpu_total_energy_mj(
                self.profiler.gpu_index
            )
            gpu_read_end_s = time.perf_counter()
            gpu_end_counter_time_s = 0.5 * (
                gpu_read_start_s + gpu_read_end_s
            )

            rapl_read_start_s = time.perf_counter()
            rapl_end = _read_rapl_snapshot(self._rapl_domains)
            rapl_read_end_s = time.perf_counter()
            rapl_end_counter_time_s = 0.5 * (
                rapl_read_start_s + rapl_read_end_s
            )
        except (RuntimeError, OSError, ValueError) as e:
            if self._monitor is not None:
                self._monitor.stop()
            self._stopped = True
            raise
        wall_time_s = max(t_end - self._t_start, 1e-9)
        self._stopped = True

        # Verify that exclusivity and one-card execution held through the end of
        # the phase, not only at start(). This check is performed immediately
        # after the end counters, before the fallback sampler's exterior point.
        try:
            (
                final_foreign_count,
                final_owned_other_count,
            ) = _inspect_node_gpu_processes(
                selected_gpu_index=self.profiler.gpu_index,
                root_pid=os.getpid(),
            )
        except (RuntimeError, OSError, TypeError) as e:
            if self._monitor is not None:
                self._monitor.stop()
            raise
        sampled_foreign_count = 0
        sampled_owned_other_count = 0
        if self._monitor is not None:
            for index, timestamp in enumerate(
                self._monitor.gpu_sample_timestamps_s
            ):
                if not (
                    self._gpu_start_counter_time_s
                    <= timestamp
                    <= gpu_end_counter_time_s
                ):
                    continue
                if index < len(self._monitor.foreign_gpu_process_counts):
                    sampled_foreign_count = max(
                        sampled_foreign_count,
                        self._monitor.foreign_gpu_process_counts[index],
                    )
                if index < len(
                    self._monitor.owned_processes_on_other_gpu_counts
                ):
                    sampled_owned_other_count = max(
                        sampled_owned_other_count,
                        self._monitor.owned_processes_on_other_gpu_counts[index],
                    )
        gpu_process_inspection_error_count = (
            self._monitor.gpu_process_inspection_errors
            if self._monitor is not None
            else 0
        )
        self._competing_gpu_process_count = max(
            self._competing_gpu_process_count,
            sampled_foreign_count,
            final_foreign_count,
        )
        self._nonselected_gpu_process_count = max(
            self._nonselected_gpu_process_count,
            sampled_owned_other_count,
            final_owned_other_count,
        )

        # Let the sampler obtain a point immediately to the right of the stop
        # boundary for the power-integration fallback, then stop it. This wait
        # occurs outside the measured interval and its process-state samples are
        # excluded above.
        if self._monitor:
            time.sleep(max(self.profiler.monitor_interval, 0.0))
            self._monitor.stop()

        # ------------------------------------------------------------------
        # GPU stats — all memory figures are per-process (our PID only)
        # ------------------------------------------------------------------
        peak_gpu_mb = 0.0
        mean_gpu_mb = 0.0

        torch = sys.modules.get("torch")
        if torch is not None:
            # torch.cuda.max_memory_allocated is per-process and device-specific.
            try:
                peak_gpu_mb = torch.cuda.max_memory_allocated(self.profiler.gpu_index) / 1024 ** 2
            except (RuntimeError, AttributeError, ZeroDivisionError):
                pass
        memory_window = _windowed_samples(
            self._monitor,
            self._monitor.gpu_memory_samples if self._monitor else [],
            self._t_start,
            t_end,
        )
        if memory_window.size:
            mean_gpu_mb = float(np.mean(memory_window))
            peak_gpu_mb = max(peak_gpu_mb, float(np.max(memory_window)))

        # Per-process GPU memory snapshot at end (point-in-time, for process_gpu_memory_mb field)
        process_gpu_mb = _get_process_gpu_memory_mb(os.getpid(), self.profiler.gpu_index)

        # ------------------------------------------------------------------
        # CPU / RAM memory stats
        # ------------------------------------------------------------------
        peak_cpu_mb = 0.0
        mean_cpu_mb = 0.0
        cpu_memory_window = _windowed_samples(
            self._monitor,
            self._monitor.cpu_memory_samples if self._monitor else [],
            self._t_start,
            t_end,
        )
        if cpu_memory_window.size:
            peak_cpu_mb = float(np.max(cpu_memory_window))
            mean_cpu_mb = float(np.mean(cpu_memory_window))

        # ------------------------------------------------------------------
        # Direct hardware-domain energy
        # ------------------------------------------------------------------
        n_package = sum(d.kind == "package" for d in self._rapl_domains)
        n_dram = sum(d.kind == "dram" for d in self._rapl_domains)
        intermediate_rapl = []
        rapl_sample_error_count = 0
        if self._monitor is not None:
            for timestamp, sample in zip(
                self._monitor.rapl_sample_timestamps_s,
                self._monitor.rapl_samples,
            ):
                if (
                    self._rapl_start_counter_time_s < timestamp
                    < rapl_end_counter_time_s
                ):
                    if sample:
                        intermediate_rapl.append(sample)
                    else:
                        rapl_sample_error_count += 1

        cpu_energy_j = _rapl_energy_j(
            self._rapl_domains,
            self._rapl_start,
            rapl_end,
            "package",
            intermediate=intermediate_rapl,
        )
        ram_energy_j = _rapl_energy_j(
            self._rapl_domains,
            self._rapl_start,
            rapl_end,
            "dram",
            intermediate=intermediate_rapl,
        )
        cpu_energy_mj = cpu_energy_j * 1000.0  # convert joules to millijoules
        ram_energy_mj = ram_energy_j * 1000.0  # convert joules to millijoules
        rapl_measurement_duration_s = max(
            rapl_end_counter_time_s - self._rapl_start_counter_time_s,
            1e-9,
        )
        cpu_power_w = cpu_energy_j / rapl_measurement_duration_s  # watts (unchanged)
        ram_power_w = ram_energy_j / rapl_measurement_duration_s  # watts (unchanged)

        integrated_gpu_j, gpu_sample_count = _integrate_gpu_power_j(
            self._monitor,
            self._gpu_start_counter_time_s,
            gpu_end_counter_time_s,
        )
        gpu_energy_source = "NVML timestamped power integration"
        if (
            self._gpu_energy_start_mj is not None
            and gpu_energy_end_mj is not None
            and gpu_energy_end_mj > self._gpu_energy_start_mj
        ):
            gpu_energy_j = (
                gpu_energy_end_mj - self._gpu_energy_start_mj
            ) / 1000.0
            gpu_energy_source = "NVML cumulative device-energy counter"
        else:
            gpu_energy_j = integrated_gpu_j

        gpu_measurement_duration_s = max(
            gpu_end_counter_time_s - self._gpu_start_counter_time_s,
            1e-9,
        )
        gpu_energy_mj = gpu_energy_j * 1000.0  # convert joules to millijoules
        gpu_power_w = (
            gpu_energy_j / gpu_measurement_duration_s
            if gpu_energy_j > 0
            else 0.0
        )

        sm_window = _windowed_samples(
            self._monitor,
            self._monitor.gpu_sm_util_samples if self._monitor else [],
            self._t_start,
            t_end,
        )
        mean_sm_util = float(np.mean(sm_window)) if sm_window.size else 0.0

        if self._baseline_gpu_power_w > GPU_BASELINE_WARNING_THRESHOLD_W:
            logger.warning(
                "GPU baseline was elevated (%.0f W) before %s; the complete "
                "device energy is retained because no idle subtraction is applied.",
                self._baseline_gpu_power_w,
                self.phase,
            )

        complete_rapl, rapl_coverage_detail = _validate_rapl_coverage(
            self._rapl_domains,
            self._cpu_package_ids,
        )
        gpu_fallback_valid = (
            gpu_energy_source != "NVML timestamped power integration"
            or gpu_sample_count >= 2
        )
        measurement_valid = bool(
            complete_rapl
            and gpu_energy_j > 0.0
            and gpu_fallback_valid
            and self._competing_gpu_process_count == 0
            and self._nonselected_gpu_process_count == 0
            and gpu_process_inspection_error_count == 0
            and rapl_sample_error_count == 0
        )
        if not measurement_valid:
            raise RuntimeError(
                "Invalid hardware-domain measurement: "
                f"RAPL package/DRAM domains={n_package}/{n_dram} "
                f"({rapl_coverage_detail or 'complete topology-derived coverage'}), "
                f"GPU energy={gpu_energy_j:.6f} J, "
                f"GPU source={gpu_energy_source!r}, valid GPU samples="
                f"{gpu_sample_count}, foreign GPU processes="
                f"{self._competing_gpu_process_count}, benchmark processes on "
                f"non-selected GPUs={self._nonselected_gpu_process_count}, "
                f"GPU-process inspection errors="
                f"{gpu_process_inspection_error_count}, RAPL sample errors="
                f"{rapl_sample_error_count}."
            )

        # Diagnostics retain process utilization and memory occupancy, but neither
        # quantity scales the measured hardware-domain energy.
        process_cpu_pct = 0.0
        process_ram_pct = 0.0
        try:
            proc = psutil.Process(os.getpid())
            end_cpu_times = proc.cpu_times()
            if self._cpu_times_start is not None:
                process_cpu_s = max(
                    0.0,
                    (end_cpu_times.user - self._cpu_times_start.user)
                    + (end_cpu_times.system - self._cpu_times_start.system),
                )
                logical_cpus = psutil.cpu_count(logical=True) or 1
                process_cpu_pct = (
                    process_cpu_s
                    / (wall_time_s * logical_cpus)
                    * 100.0
                )
            process_ram_pct = (
                proc.memory_info().rss / psutil.virtual_memory().total * 100.0
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, ZeroDivisionError, AttributeError):
            pass

        carbon_intensity = (
            self._live_carbon_intensity
            if self._live_carbon_intensity is not None
            else GERMANY_CARBON_INTENSITY_G_PER_KWH
        )
        if self.profiler.carbon_intensity_override > 0.0:
            location = (
                f"user-supplied intensity "
                f"(zone: {self.profiler.electricitymaps_zone})"
            )
        elif self._ci_source.startswith("Electricity Maps API"):
            location = (
                f"Electricity Maps live "
                f"(zone: {self.profiler.electricitymaps_zone})"
            )
        elif self._ci_source.startswith("timeline file"):
            location = (
                f"Electricity Maps timeline "
                f"(zone: {self.profiler.electricitymaps_zone})"
            )
        else:
            location = (
                f"{LOCATION_LABEL} "
                f"[non-live CI estimate — source: {self._ci_source}]"
            )

        # ------------------------------------------------------------------
        # IT total energy (exclusive hardware domains, in mJ)
        # ==================================================================
        # ENERGY EQUATION: E_IT = E_GPU + E_CPU + E_RAM
        # E_IT is the sum of all hardware-domain energy components.
        # Per-image energy is computed by the experiment class:
        #   E_i  = E_IT / N_test     (inference energy per test image)
        #   E_ft = E_IT / N_support  (fine-tuning energy per training image)
        # ------------------------------------------------------------------
        it_energy_mj = cpu_energy_mj + gpu_energy_mj + ram_energy_mj

        # ------------------------------------------------------------------
        # PUE – total facility energy (in mJ)
        # ------------------------------------------------------------------
        # Use the benchmark constant directly so later mutation of the public
        # compatibility attribute cannot alter an experiment's facility energy.
        pue = FIXED_PUE
        total_facility_mj = it_energy_mj * pue

        # ------------------------------------------------------------------
        # Carbon emissions (Bielefeld, Germany)
        # Note: carbon_intensity is in g/kWh, so convert mJ back to kWh
        # ------------------------------------------------------------------
        total_facility_kwh = total_facility_mj / KWH_TO_MJ  # convert mJ to kWh for CO2 calc
        co2_g = total_facility_kwh * carbon_intensity
        co2_g_per_image = co2_g / self.n_images if self.n_images > 0 else 0.0

        # ------------------------------------------------------------------
        # Throughput (images per joule)
        # ------------------------------------------------------------------
        throughput_ips = self.n_images / wall_time_s if self.n_images > 0 else 0.0
        it_energy_j = it_energy_mj / 1000.0  # convert mJ to J for throughput calculation
        throughput_ipj = (
            self.n_images / it_energy_j if it_energy_j > 0 and self.n_images > 0 else 0.0
        )

        # ------------------------------------------------------------------
        # Extended resource metrics (experiment-impact-tracker integration)
        # ------------------------------------------------------------------
        # (mean_sm_util already computed above in the GPU energy block)

        # GPU performance state at the end of the run
        gpu_p_state = _get_gpu_p_state(self.profiler.gpu_index)

        # Process PSS RAM (fair shared-memory attribution; falls back to RSS)
        ram_pss_mb = _get_process_pss_mb()

        # Actual CPU-time hours: delta of (user + system) process CPU time
        cpu_hours = 0.0
        try:
            cpu_times_end = psutil.Process(os.getpid()).cpu_times()
            if self._cpu_times_start is not None:
                delta_user = cpu_times_end.user - self._cpu_times_start.user
                delta_sys  = cpu_times_end.system - self._cpu_times_start.system
                cpu_hours = max(0.0, delta_user + delta_sys) / 3600.0
        except (TypeError, AttributeError, ZeroDivisionError):
            pass

        # GPU-hours: utilization-weighted for the one selected card only.
        gpu_hours = (mean_sm_util / 100.0) * wall_time_s / 3600.0

        # 5-minute system load average at experiment end
        try:
            system_load_5min = float(psutil.getloadavg()[1])
        except (OSError, TypeError, ValueError, IndexError):
            system_load_5min = 0.0

        # Social cost of carbon (Ricke et al. 2018), keyed off the zone the live
        # carbon intensity was actually measured in (falls back to DEU only if
        # that zone has no SCC estimate of its own).
        scc_country = _country_code_from_zone(self.profiler.electricitymaps_zone)
        social_cost_usd = _compute_social_cost_carbon_usd(co2_g, country_code=scc_country)

        return ProfilerResult(
            # Identity
            phase=self.phase,
            n_images=self.n_images,
            # Time
            wall_time_s=round(wall_time_s, 4),
            time_per_image_s=round(wall_time_s / self.n_images, 6) if self.n_images > 0 else 0.0,
            # GPU memory
            peak_gpu_memory_mb=round(peak_gpu_mb, 2),
            mean_gpu_memory_mb=round(mean_gpu_mb, 2),
            process_gpu_memory_mb=round(process_gpu_mb, 2),
            # CPU/RAM memory
            peak_cpu_memory_mb=round(peak_cpu_mb, 2),
            mean_cpu_memory_mb=round(mean_cpu_mb, 2),
            # Power
            cpu_power_w=round(cpu_power_w, 3),
            gpu_power_w=round(gpu_power_w, 3),
            ram_power_w=round(ram_power_w, 3),
            baseline_gpu_power_w=round(self._baseline_gpu_power_w, 3),
            # Component energy (exclusive hardware domains, mJ)
            cpu_energy_mj=round(cpu_energy_mj, 6),
            gpu_energy_mj=round(gpu_energy_mj, 6),
            ram_energy_mj=round(ram_energy_mj, 6),
            energy_measurement_scope=(
                "GPU device + CPU package + DRAM domains; attributed to the "
                "benchmark under exclusive allocation; idle energy retained"
            ),
            cpu_energy_source="Linux powercap RAPL package counters",
            gpu_energy_source=gpu_energy_source,
            ram_energy_source="Linux powercap RAPL DRAM counters",
            rapl_package_domain_count=n_package,
            rapl_dram_domain_count=n_dram,
            cpu_topology_package_count=len(self._cpu_package_ids),
            cpu_topology_package_ids=",".join(
                str(package_id)
                for package_id in sorted(self._cpu_package_ids)
            ),
            rapl_package_ids=",".join(
                str(package_id)
                for package_id in sorted(
                    {
                        domain.package_id
                        for domain in self._rapl_domains
                        if domain.kind == "package"
                    }
                )
            ),
            rapl_dram_package_ids=",".join(
                str(package_id)
                for package_id in sorted(
                    {
                        domain.package_id
                        for domain in self._rapl_domains
                        if domain.kind == "dram"
                    }
                )
            ),
            rapl_sample_error_count=rapl_sample_error_count,
            gpu_power_sample_count=gpu_sample_count,
            competing_gpu_process_count=self._competing_gpu_process_count,
            nonselected_gpu_process_count=self._nonselected_gpu_process_count,
            gpu_process_inspection_error_count=gpu_process_inspection_error_count,
            node_gpu_count=self._node_gpu_count,
            selected_gpu_logical_index=self.profiler.gpu_index,
            selected_gpu_physical_index=self._selected_gpu_physical_index,
            cpu_measurement_duration_s=round(rapl_measurement_duration_s, 6),
            gpu_measurement_duration_s=round(gpu_measurement_duration_s, 6),
            ram_measurement_duration_s=round(rapl_measurement_duration_s, 6),
            measurement_valid=measurement_valid,
            # IT total (mJ)
            it_energy_mj=round(it_energy_mj, 6),
            # PUE
            pue=round(pue, 3),
            total_facility_energy_mj=round(total_facility_mj, 6),
            power_usage_effectiveness=round(pue, 3),
            # Carbon (Bielefeld, Germany)
            co2_location=location,
            carbon_intensity_g_per_kwh=round(carbon_intensity, 2),
            carbon_intensity_source=self._ci_source,
            carbon_intensity_datetime=self._ci_reading_datetime or "",
            carbon_intensity_fetched_at=self._ci_fetched_at or "",
            co2_eq_g=round(co2_g, 6),
            co2_eq_g_per_image=round(co2_g_per_image, 8),
            # Throughput
            throughput_img_per_s=round(throughput_ips, 4),
            throughput_img_per_j=round(throughput_ipj, 4),
            # Process utilisation
            process_cpu_percent=round(process_cpu_pct, 2),
            process_ram_percent=round(process_ram_pct, 2),
            # Extended resource metrics
            gpu_sm_utilization_pct=round(mean_sm_util, 2),
            gpu_p_state=gpu_p_state,
            ram_pss_mb=round(ram_pss_mb, 2),
            cpu_hours=round(cpu_hours, 8),
            gpu_hours=round(gpu_hours, 8),
            system_load_avg_5min=round(system_load_5min, 3),
            disk_write_speed_mbs=round(self._disk_write_speed, 2),
            social_cost_carbon_usd=round(social_cost_usd, 10),
        )
