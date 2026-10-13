# sapi — Sustainability-Aware Performance Index for Cell and Nucleus Instance Segmentation

> **This repository is the official implementation of the experiments and
> results reported in the paper:**
>
> **[Performance at What Cost? A Sustainability-Aware Performance Index for Cell and Nucleus Instance Segmentation](https://arxiv.org/pdf/2610.10324)**

`sapi` benchmarks cell / nucleus instance segmentation models on **accuracy and
sustainability**. It runs 19 pretrained model variants on the six datasets of
[CellBinDB](https://doi.org/10.5281/zenodo.15370205). For every run it measures
segmentation quality, energy use (GPU, CPU and DRAM) and CO₂ emissions, and it
combines performance, energy and model size into a single **SAPI** score.

## What it does

There are two experiment types:

| Experiment | What happens | Score |
|---|---|---|
| `inference` | The pretrained model segments every image of a dataset (zero-shot, no training). | **SAPI_ZS** |
| `fewshot_finetuning` | One image per tissue type is used to fine-tune the model, and the remaining images are used for evaluation. Each run reports zero-shot, full-model fine-tuning and frozen-encoder fine-tuning results. | **SAPI_FT+I** |

```
SAPI_ZS   = P^(1/2) · (1 / log10(E_i))^(1/3) · (1 / m)^(1/6)
SAPI_FT+I = P^(1/2) · (1 / log10(E))^(1/3)   · (1 / m)^(1/6),   E = 0.8 · E_i + 0.2 · E_ft
```

- `P` is the harmonic mean of AJI+, PQ and NSD.
- `E_i` is the inference energy per image, and `E_ft` is the fine-tuning energy per training image, both in mJ.
- `m = log10(number of parameters)`.


## Requirements

- Linux with an NVIDIA GPU (energy is read through NVML).
- Intel RAPL energy counters readable by your user, including DRAM. They are usually root-only, so ask your admin for read access.
- **An exclusive node.** Energy is measured for the whole node, so run only one sweep per node at a time and nothing else on its GPUs.
- Conda, and Python ≥ 3.9 in every environment.

## Setup

### 1. Create the model environments

The models have conflicting dependencies (e.g.: TensorFlow 2.8 vs 2.15, Cellpose 3 vs 4), so you create a separate environment for each model, or for each group of models whose dependencies are compatible. Each environment needs the model's own package(s) and the name of the environment for each model must be set in the `configs/cluster_config.sh` script.

Install `sapi` requirements using:

```bash
pip install -r requirements.txt
```

### 2. Download the source-only models

HoVer-Net, MEDIAR and CellSAM are run from their source code. Download them
into `third_party/`:

| Model | Repository | Folder |
|---|---|---|
| HoVer-Net | https://github.com/vqdang/hover_net | `third_party/hover_net-master/` |
| MEDIAR | https://github.com/Lee-Gihun/MEDIAR | `third_party/MEDIAR-main/` |
| CellSAM | https://github.com/vanvalenlab/cellSAM | `third_party/cellSAM-master/` |

### 3. Download the pretrained weights

> **All original pretrained weights must be downloaded before you run experiments.** A model with missing weights may run without them and produce meaningless scores.

| Model(s) | Weights | Location |
|---|---|---|
| StarDist, Cellpose, Cellpose-SAM | downloaded by the library on first use | library cache |
| µSAM, PathoSAM | downloaded by `micro_sam` on first use | `~/.cache/micro_sam/models/` |
| DeepCell, Mesmer | downloaded by `deepcell` on first use (needs `DEEPCELL_ACCESS_TOKEN`, see step 5) | `~/.deepcell/models/` |
| CellSAM | `cellsam_general.pt` (v1.2) | `~/.deepcell/models/cellsam_v1.2/` |
| InstanSeg | `single_channel_nuclei` → `instanseg.pt` (BioImage Model Zoo) | `~/.instanseg/instanseg.pt` |
| HoVer-Net | `hovernet_original_kumar_notype_pytorch.tar` (HoVer-Net README, *Model Weights*) | `~/.hovernet/` |
| MEDIAR | `from_phase2.pth` (MEDIAR *Releases*, **not** `phase2.pth`) | `~/.mediar/finetuned/` |
| CellViT-256 | [`CellViT-256-x20.pth`](https://drive.google.com/uc?export=download&id=1w99U4sxDQgOSuiHMyvS_NYBiz6ozolN2) | any; set `weight_path` in `configs/models/cellvit_256.yaml` |
| CellViT-SAM-H | [`CellViT-SAM-H-x20.pth`](https://drive.google.com/uc?export=download&id=1wP4WhHLNwyJv97AK42pWK8kPoWlrqi30) | any; set `weight_path` in `configs/models/cellvit_sam_h.yaml` |

The locations can be changed in `configs/cluster_config.sh`.

### 4. Download the data

> **The original CellBinDB data must be downloaded before you run
> experiments.** `sapi` does not download it.

Download CellBinDB from its official source on Zenodo:
https://doi.org/10.5281/zenodo.15370205. Place it under `data/CellBinDB/`, keeping the original folder and file names:

```
data/CellBinDB/
├── DAPI/
│   ├── <sample>/
│   │   ├── <sample>-img.tif
│   │   └── <sample>-instancemask.tif
│   └── ...
├── ssDNA/   
├── mIF/   
├── HE/   
├── 10xGenomics_DAPI/   
├── 10xGenomics_HE/
```

### 5. Configure

- **`configs/cluster_config.sh`:** set `_TORCH_ROOT` and `_TF_ROOT` to the folders that contain your conda environments. If your environment names or grouping differ from the paper's, edit the env names and the `MODEL_ENV` table (model → env).
- **DeepCell token** (DeepCell, Mesmer and CellSAM downloads): get a free token at https://users.deepcell.org, then run
  ```bash
  export DEEPCELL_ACCESS_TOKEN=<YOUR_DEEPCELL_TOKEN>
  ```
- **Carbon intensity (optional).** By default a fixed 309 g CO₂eq/kWh (Germany) is used. To use live values, use one of these options:
  - Create `configs/electricity_maps.json` containing `{"api_token": "<YOUR_TOKEN>", "zone": "DE"}`.
  - On an offline node, provide a CSV timeline (`timestamp_utc,zone,carbon_intensity_g_per_kwh`) via `--ci_timeline` or as `configs/ci_timeline.csv`.
  - Pass a fixed value with `--electricitymaps_intensity`.


## Running experiments

### Full benchmark sweeps

The sweep scripts run every model on every compatible dataset, each model in
its own environment:

```bash
# Zero-shot inference sweep (SAPI_ZS), all 19 models
./scripts/run_zero_shot_inference.sh --seed 42 --gpu 0

# Few-shot fine-tuning sweep (SAPI_FT+I), all fine-tunable models
./scripts/run_fewshot_finetuning_benchmark.sh --seed 42 --gpu 0

# Narrow it down to a model family, variant and/or dataset
./scripts/run_fewshot_finetuning_benchmark.sh --seed 42 --model cellpose --variant cyto3 --dataset DAPI


### A single run

Activate the model's environment, load the paths, and call `run_experiment.py`:

```bash
conda activate cellpose3_env
BENCH_ROOT=$PWD source configs/cluster_config.sh   # sets model source and weight paths

python run_experiment.py --model cellpose_cyto3 --dataset DAPI --experiment_type inference
python run_experiment.py --model cellpose_cyto3 --dataset DAPI --experiment_type fewshot_finetuning
```

Run `python run_experiment.py --help` for all options, for example `--seed`,
`--data_root`, `--output_dir` and `--save_predictions`.

## Results

Each run gets its own timestamped folder; earlier runs are never overwritten:

```
results_zero_shot/<family>/<variant>/<dataset>/zero_shot_inference/<run>/
results/<family>/<variant>/<dataset>/fewshot_finetuning/<run>/
```

## Citation

If you use this code, please cite the paper:



## License

The `sapi` code is released under the [MIT License](LICENSE). The third-party models, their weights, and the CellBinDB dataset are covered by their own licences.
