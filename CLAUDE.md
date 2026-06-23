# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Requires **Python 3.12**. The project uses a virtual environment at `.venv/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The convex-optimisation heading correction requires a valid MOSEK license at `~/mosek/mosek.lic`. The simpler `naive_heading_correction` does not require MOSEK.

## Data

Experimental data must be downloaded separately from Dryad:

**Training data** (van Breugel et al. 2014) — Parquet format:
```
experimentaldata/
├── 30cms/  # flight_trajectories_3d_HCS_odor_horizon_matched.parquet, body_orientations_HCS_odor_horizon_matched.parquet, body_trajec_matches.parquet
├── 40cms/
└── 60cms/
```

**External evaluation data** (David et al.) — CSV format, from https://datadryad.org/dataset/doi:10.5061/dryad.zw3r228fw:
```
experimentaldata/external/
├── orco_laminar1.csv
├── orco_traj2.csv
└── orco_traj3.csv
```

## Architecture

This project predicts *Drosophila* body heading angles from flight trajectory data using a neural network trained on data from van Breugel et al. (2014).

### Notebooks (primary workflow)

Run in order:

- **`notebooks/data_pipeline.ipynb`** — Full preprocessing pipeline: merges raw Parquet data across wind speeds (30/40/60 cm/s), corrects 180° heading ambiguities, filters and smooths trajectories, and saves the final dataset to `pipelinedata/06_final/`.
- **`notebooks/model_training.ipynb`** — Loads preprocessed data from `pipelinedata/06_final/`, constructs time-delay–embedded features, runs a hyperparameter grid search, trains the Keras neural network, saves `models/drosophila_body_orientation_predictor.keras`, and evaluates predictions on both the training dataset and an external dataset (`pipelinedata/external/`).

### Python modules (in `utils/`)

Both notebooks add `../utils` to `sys.path` so `from utils import ...` resolves correctly.

- **`utils/utils.py`** — All shared functions. Two parallel APIs for the same computations:
  - **Top-level functions** (used by the notebooks): `naive_heading_correction`, `convex_opt_heading_correction`, `augment_fly_trajectory`, `smooth_trajectory`, `augment_with_time_delay_embedding`, `plot_trajectory`, `plot_trajectory_with_predicted_heading`, `create_model`, `custom_density_plots`.
  - **`compute` class** (static methods): `compute.angular_velocity`, `compute.linear_acceleration`, `compute.thrust`, `compute.heading_angle_corrected`, `compute.heading_angle_convex_opt` — lower-level, return arrays rather than augmented DataFrames.
- **`utils/fly_plot_lib_plot.py`** — Wrapper around `FlyPlotLib` for trajectory visualisation with heading arrows.

### Pipeline data stages (`pipelinedata/`)

Stages 01–06 are written by `data_pipeline.ipynb`; stages 07–08 are written by `model_training.ipynb`. `figures/` and `pipelinedata/` are generated outputs and are not tracked in git.

| Directory | Written by | Contents |
|---|---|---|
| `01_merged/` | `data_pipeline.ipynb` | Per-wind-speed CSVs + combined `all_wind_heading_and_trajectories.csv` |
| `02_augmented/` | `data_pipeline.ipynb` | Kinematic features added (groundspeed, airspeed, thrust, acceleration) |
| `03_corrected/` | `data_pipeline.ipynb` | 180° heading ambiguities resolved |
| `04_filtered/` | `data_pipeline.ipynb` | Short/uncorrectable trajectories removed; `rejected_trajectories.csv` |
| `05_smoothed/` | `data_pipeline.ipynb` | Savitzky–Golay smoothing applied |
| `06_final/` | `data_pipeline.ipynb` | `heading_angle_x`/`heading_angle_y` columns added; ready for training |
| `07_time_delay_embedded/` | `model_training.ipynb` | Time-delay–embedded feature matrix (no wind augmentation); `traj_augment_original.csv` |
| `08_wind_augmented/` | `model_training.ipynb` | Time-delay–embedded feature matrix with random wind-direction rotation; `traj_augment_wind.csv` |
| `external/` | `model_training.ipynb` | Augmented external datasets (David's data) |

### Trained models (`models/`)

- **`models/drosophila_body_orientation_predictor.keras`** — Canonical published model trained on Floris et al. data.
- **`models/model_CEM_all-angle-rotate.keras`** — Variant trained with `wind_augment=True` in `augment_with_time_delay_embedding`, which applies random wind-direction rotations to remove wind-direction bias.

### Key data pipeline steps

1. **Raw data**: Parquet files for trajectory, body orientation, and key table per wind speed.
2. **Merging**: Join trajectory + body orientation via key table; concatenate across wind speeds.
3. **Augmentation**: Compute groundspeed, airspeed, thrust (mass=0.25e-6 kg, dragcoeff=mass/0.170), linear acceleration, heading components (cos/sin). Differentiation uses `pynumdiff` Savitzky-Golay with params `[2, 10, 10]`.
4. **Heading correction**: Fix 180° ambiguities using `naive_heading_correction` (np.unwrap) then `convex_opt_heading_correction` (cvxpy + MOSEK, minimizes total variation + thrust-alignment).
5. **Filtering**: Remove trajectories with residual π-flips above threshold.
6. **Smoothing**: Unwrap heading signal then apply Savitzky–Golay filter (default params `[1, 5, 5]`).
7. **Time-delay embedding**: `augment_with_time_delay_embedding` creates lookback window features (window=4) across [groundspeed, groundspeed_angle, airspeed, airspeed_angle, thrust, thrust_angle] → 24 input features.
8. **Model**: Keras fully-connected network (default: 1 hidden layer, 50 neurons, ReLU, `UnitNormalization` output layer) predicts [heading_angle_x, heading_angle_y] (unit vector components); heading angle recovered via `arctan2`.

### Coordinate frame convention

The model was trained with:
- Positive x = upwind direction
- Wind heading = negative x direction (coming from the right)
- Positive y = crosswind such that x × y points out of screen

External data must be transformed to match this frame before prediction.
