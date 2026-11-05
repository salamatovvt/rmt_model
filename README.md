# Almaty greening pipeline — code and outputs for the Results section

`almaty_greening_pipeline.py` implements the three-stage design of the paper on a **synthetic** dataset
(5 reference stations + 38 low-cost sensors, 2019–2023, 500 m grid with 2,716 cells, 8 districts).
The numbers in `results.tex` are taken from `outputs/` of the full run below and are reported in the
paper as expected results (the Kazhydromet archive has not been received yet).

## Run

```bash
pip install numpy pandas scipy scikit-learn matplotlib xgboost shap libpysal esda tensorflow
python almaty_greening_pipeline.py              # full run, seed 42 (8.2 min on one Xeon core)
python almaty_greening_pipeline.py --quick      # smoke run (~4.5 min)
python almaty_greening_pipeline.py --no-lstm    # skip the LSTM baseline
python almaty_greening_pipeline.py --seed 7     # another seed
```

## Outputs → paper objects

| File | Paper object |
|---|---|
| `table_dataset.csv`, `table_target_distribution.csv`, `table_fold_sizes.csv` | Table I (dataset) |
| `table_models_pooled_ci.csv`, `table_dm_tests.csv`, `table_models.csv`, `table_models_by_fold.csv`, `table_loso_by_station.csv`, `table_ablation.csv` | Table II (main results, CIs, DM tests, LOSO, ablation) |
| `fig_timeseries.pdf` | Fig. 1, held-out 2023 at the Almaly reference station |
| `table_hotspots.csv`, `table_hotspot_class_by_district.csv`, `seasonal_surfaces.csv` | Table III, hotspot counts by district |
| `fig_shap_ndvi.pdf`, `table_ndvi_effect.csv`, `table_h1_contrast.csv` | Fig. 2 and Table IV (RQ1 / H1) |
| `fig_maps.pdf`, `grid_results.csv`, `table_top100_by_district.csv`, `table_sensitivity*.csv`, `table_criteria_correlation.csv` | Fig. 3 (RQ2 / H2) |
| `fig_errors.pdf`, `table_confusion_*.csv`, `table_exceedance_by_season.csv`, `table_error_by_episode_day.csv`, `table_worst_station_days.csv` | Fig. 4 and error analysis |
| `summary.json` | all headline numbers |
| `reproducibility.json` | library versions, hardware, seed, running time per stage |
| `fig_districts.pdf` | not used in the paper (counts are given in the text) |

## What changed against the previous version

* All four forecasting approaches are scored on the same 60,277 test station-days: LSTM test windows take
  their first 14 days of history from the end of the training period. LSTM: up to 20 epochs with early stopping.
* Uncertainty: moving-block bootstrap 95 % CIs (7-day blocks, 1,000 resamples) for RMSE and F1;
  Diebold–Mariano tests (Newey–West lag 7, HLN correction) with Holm adjustment.
* Additional analyses: feature-group ablation; Gi* at 99 % next to the planned 95 %.
* Error analysis: confusion matrices, per-season exceedance metrics with base rates, error by day of an
  inversion episode, composition of the 1 % largest errors.
* RQ1: NDVI effect for all four seasons, in µg/m³ and % of the class mean, with cell-bootstrap CIs, the H1
  contrast against the non-heating season, and the latent effect of the simulation for comparison.
* Stage 3: cells without plantable land are labelled "Excluded" instead of an empty tier.
* Dataset bookkeeping (records removed and why) and a reproducibility record.

## Before submitting

* Push this folder to the group's repository and put its URL into the Reproducibility paragraph of
  `results.tex` (marked `TODO`).
* When the real archive arrives, replace Stage 0 with a data loader; nothing downstream needs to change.
