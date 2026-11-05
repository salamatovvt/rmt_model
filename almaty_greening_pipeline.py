#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seasonal PM2.5 forecasting, hotspot detection and greening prioritisation for Almaty.
Implements Section "Methods and Materials" of the paper on a SIMULATED dataset
(Sect. "Data basis of the results"): 5 reference stations + 38 low-cost sensors,
five heating seasons, 500 m grid, Getis-Ord Gi*, AHP prioritisation, SHAP attribution.

Stages
  0. Simulation of the city, the networks and the daily series
  1. Seasonal forecasting at monitoring points (XGBoost, LSTM, persistence, climatology)
     under temporal blocking and leave-one-station-out        -> Table "models"
  2. Spatial generalisation to the grid, seasonal maps, Moran's I, Gi*, bootstrap
     stability, persistent / winter-only classification        -> Table "hotspots"
     SHAP dependence of PM2.5 on NDVI by season x built-up class -> fig_shap_ndvi.pdf
  3. AHP weights, priority score, sensitivity analysis, district aggregation
                                                                -> fig_districts.pdf
Run:
  python almaty_greening_pipeline.py            # full run (LSTM + 100 bootstrap resamples)
  python almaty_greening_pipeline.py --quick    # smoke run (small LSTM, 20 resamples)
Outputs go to ./outputs/ : figures (pdf+png), tables (csv), summary.json, reproducibility.json

Additions for the Results section (Assignment 3):
  * dataset table (records per network, removals and why, fold sizes, target distribution)
  * all models scored on the same test rows (LSTM windows take their history from the training years)
  * moving-block bootstrap 95 % CIs for RMSE / F1 and Diebold-Mariano tests (HLN correction, Holm)
  * Stage-1 feature-group ablation
  * error analysis: confusion matrix, error by day of inversion episode, worst station-days
  * Gi* at 95 % (planned) and 99 % (sensitivity)
  * NDVI attribution per season of the heating calendar, in ug/m3 and % of seasonal mean, with
    cell-bootstrap CIs (H1)
  * reproducibility record: library versions, hardware, seed, running time per stage
"""

import argparse, json, os, platform, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_fscore_support
import xgboost as xgb
import shap
from libpysal.weights import lat2W
from esda.moran import Moran
from esda.getisord import G_Local

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- config
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="fast smoke run")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", default="outputs")
parser.add_argument("--no-lstm", action="store_true", help="skip the LSTM baseline")
ARGS = parser.parse_args()

SEED = ARGS.seed
rng = np.random.default_rng(SEED)
OUT = ARGS.out
os.makedirs(OUT, exist_ok=True)

N_YEARS = 5                       # five complete years = five heating seasons
N_REF, N_SENSORS = 5, 38          # reference stations and low-cost sensors
GRID_NX, GRID_NY = 52, 53         # 500 m grid, masked below to the built city (target 2,716 cells)
N_CELLS_TARGET = 2716
MPC = 35.0                        # Kazakhstan daily maximum permissible concentration, ug/m3
N_BOOT = 20 if ARGS.quick else 100
STABILITY_THRESHOLD = 0.70
LSTM_EPOCHS = 3 if ARGS.quick else 20      # upper bound; early stopping on a 10 % validation split
N_CI_BOOT = 200 if ARGS.quick else 1000   # moving-block bootstrap resamples for CIs
BLOCK_DAYS = 7                            # block length for the moving-block bootstrap and the DM HAC lag
T_START = time.time(); STAGE_TIMES = {}
LSTM_SEQ = 14
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, tree_method="hist")
# the surface model (no lag features) is regularised harder so that it cannot memorise the 43 points
# through their exact coordinates-like features and must use smooth spatial relations instead
SURFACE_PARAMS = dict(XGB_PARAMS, max_depth=4, min_child_weight=200, colsample_bytree=0.6, n_estimators=500, learning_rate=0.05)

DISTRICTS = {  # approximate district centroids (lat, lon) for a Voronoi assignment
    "Alatau": (43.330, 76.860), "Turksib": (43.350, 76.980), "Zhetysu": (43.300, 76.950),
    "Auezov": (43.230, 76.845), "Almaly": (43.250, 76.920), "Nauryzbay": (43.175, 76.800),
    "Bostandyk": (43.205, 76.915), "Medeu": (43.225, 77.000),
}
CHP = {"CHP-2": (43.335, 76.870), "CHP-1": (43.300, 76.965)}   # coal-fired combined heat and power plants
LAT0, LAT1, LON0, LON1 = 43.16, 43.40, 76.78, 77.04             # study bounding box

SEASON_NAMES = ["early_heating", "deep_winter", "late_heating", "non_heating"]
SEASON_LABELS = {"early_heating": "Early heating", "deep_winter": "Deep winter",
                 "late_heating": "Late heating", "non_heating": "Non-heating"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def season_of(doy):
    """Heating calendar: mid-Oct..mid-Dec early, mid-Dec..end-Feb deep, Mar..mid-Apr late, rest non-heating."""
    doy = np.asarray(doy)
    s = np.full(doy.shape, "non_heating", dtype=object)
    s[(doy >= 288) | (doy <= 349)] = "early_heating"        # 15 Oct - 15 Dec
    s[(doy >= 350) | (doy <= 59)] = "deep_winter"           # 16 Dec - 28 Feb
    s[(doy >= 60) & (doy <= 105)] = "late_heating"          # 1 Mar - 15 Apr
    s[(doy >= 106) & (doy <= 287)] = "non_heating"
    return s


# =========================================================================== STAGE 0
# Simulated city, networks and series, tuned to the published statistics of Almaty.
log("Stage 0: simulating city grid")

lat_edges = np.linspace(LAT0, LAT1, GRID_NY + 1)
lon_edges = np.linspace(LON0, LON1, GRID_NX + 1)
lat_c = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_c = (lon_edges[:-1] + lon_edges[1:]) / 2
LON, LAT = np.meshgrid(lon_c, lat_c)                        # shape (NY, NX), row 0 = south
grid = pd.DataFrame({"row": np.repeat(np.arange(GRID_NY), GRID_NX),
                     "col": np.tile(np.arange(GRID_NX), GRID_NY),
                     "lat": LAT.ravel(), "lon": LON.ravel()})


def km_between(lat1, lon1, lat2, lon2):
    return np.sqrt(((lat1 - lat2) * 111.0) ** 2 + ((lon1 - lon2) * 111.0 * np.cos(np.radians(43.25))) ** 2)


# elevation: the city rises from ~600 m in the north to >1000 m in the south (mountain foot)
y_norm = (grid.lat - LAT0) / (LAT1 - LAT0)                  # 0 south .. 1 north
x_norm = (grid.lon - LON0) / (LON1 - LON0)
grid["elevation"] = 1150 - 560 * y_norm + 60 * np.sin(4 * np.pi * x_norm) * (1 - y_norm) + rng.normal(0, 15, len(grid))

# distance to CHP plants
for name, (la, lo) in CHP.items():
    grid[f"dist_{name.replace('-', '').lower()}"] = km_between(grid.lat, grid.lon, la, lo)
grid["dist_chp_min"] = grid[["dist_chp2", "dist_chp1"]].min(axis=1)

# built-up fraction: dense centre (multi-storey), low-rise private sector in the north, thin on the slopes
centre = np.exp(-(((grid.lat - 43.255) * 111) ** 2 + ((grid.lon - 76.92) * 81) ** 2) / (2 * 3.2 ** 2))
north_private = np.exp(-(((grid.lat - 43.335) * 111) ** 2 + ((grid.lon - 76.90) * 81) ** 2) / (2 * 2.1 ** 2))
grid["built_up"] = np.clip(0.80 * centre + 0.45 * north_private + 0.15 * (1 - np.abs(y_norm - 0.5)) + rng.normal(0, 0.05, len(grid)), 0.02, 0.95)
grid["low_rise"] = np.clip(north_private * 0.9 + rng.normal(0, 0.05, len(grid)), 0, 1)   # share of private sector (coal stoves)

# road density: follows built-up, extra along the central grid
grid["road_density"] = np.clip(grid.built_up * 9 + 2 * centre + rng.normal(0, 0.7, len(grid)), 0.2, 12)

# NDVI: the mountain-side south is green (parks, orchards), the northern periphery is bare; seasonal composites
# parks, orchards and river corridors add vegetation that does not follow the slope gradient
parks = sum(a * np.exp(-(((grid.lat - la) * 111) ** 2 + ((grid.lon - lo) * 81) ** 2) / (2 * r ** 2))
            for la, lo, a, r in [(43.24, 76.95, 0.30, 1.2), (43.27, 76.88, 0.25, 1.0), (43.31, 76.93, 0.22, 0.9),
                                 (43.20, 76.86, 0.28, 1.4), (43.34, 76.95, 0.20, 1.1), (43.29, 77.00, 0.22, 1.0),
                                 (43.22, 76.80, 0.25, 1.2), (43.36, 76.84, 0.18, 1.0)])
river = 0.15 * np.exp(-((grid.lon - (76.90 + 0.08 * y_norm)) * 81) ** 2 / (2 * 0.5 ** 2))
ndvi_base = np.clip(0.22 + 0.12 * (1 - y_norm) ** 1.3 - 0.20 * grid.built_up + 1.3 * parks + river + rng.normal(0, 0.08, len(grid)), 0.05, 0.75)
grid["ndvi_non_heating"] = ndvi_base
grid["ndvi_early_heating"] = np.clip(ndvi_base * 0.70, 0.03, 0.7)
grid["ndvi_late_heating"] = np.clip(ndvi_base * 0.55, 0.03, 0.7)
grid["ndvi_deep_winter"] = np.clip(ndvi_base * 0.40, 0.03, 0.7)

# population (WorldPop-like counts per 500 m cell) and land availability (open ground, vacant lots)
grid["population"] = np.clip(9000 * centre + 4500 * north_private + 600 + rng.gamma(2, 250, len(grid)), 0, None)
grid["land_available"] = np.clip(0.55 * (1 - grid.built_up) * (0.6 + 0.6 * y_norm) + rng.normal(0, 0.06, len(grid)), 0, 1)

# district assignment (nearest district centroid)
d_names = list(DISTRICTS)
d_xy = np.array([[la, lo] for la, lo in DISTRICTS.values()])
dist_to_d = np.stack([km_between(grid.lat, grid.lon, la, lo) for la, lo in d_xy], axis=1)
grid["district"] = np.array(d_names)[dist_to_d.argmin(axis=1)]

# city mask: drop the highest cells (mountain slopes / protected areas) so that n = 2,716
grid = grid.sort_values("elevation").iloc[:N_CELLS_TARGET].sort_values(["row", "col"]).reset_index(drop=True)
grid["cell_id"] = np.arange(len(grid))
log(f"grid: {len(grid)} cells, districts: {grid.district.value_counts().to_dict()}")


# --- latent "true" spatial pollution factor (unknown to the models) ---------------------------
# Two latent spatial components (unknown to the models):
#   persistent_part - stationary coal sources: CHP plants and the low-rise private sector, active all year;
#   centre_part     - basin-floor / dense-centre load that appears only when a deep-winter cold pool forms.
src = np.exp(-grid.dist_chp2 / 1.7) * 1.0 + np.exp(-grid.dist_chp1 / 1.5) * 0.6
low_elev = np.exp(-(grid.elevation - 600) / 300)
grid["persistent_part"] = 0.55 + 0.7 * src + 0.6 * grid.low_rise + 0.04 * low_elev + 0.05 * grid.built_up
grid["centre_part"] = 2.0 * grid.built_up ** 2 + 0.15 * low_elev
PNORM = float(grid.persistent_part.mean())

# --- monitoring networks ----------------------------------------------------------------------
log("Stage 0: placing stations and sensors")
ref_cells = grid.iloc[[grid.index[(grid.district == d)].to_series().sample(1, random_state=SEED + i).iloc[0]
                       for i, d in enumerate(["Almaly", "Alatau", "Turksib", "Bostandyk", "Auezov"])]].copy()
ref_cells["station"] = [f"REF{i+1}" for i in range(N_REF)]
ref_cells["kind"] = "reference"
_cand = grid.drop(ref_cells.index)
# community sensors sit where people live and where residents complain: dense housing and the coal-heated private sector
_w = 1.2 * _cand.population / _cand.population.mean() + 0.5 + 2.5 * _cand.low_rise + 1.5 * np.exp(-_cand.dist_chp_min / 3)
sensor_cells = _cand.sample(N_SENSORS, weights=_w, random_state=SEED).copy()
sensor_cells["station"] = [f"S{i+1:02d}" for i in range(N_SENSORS)]
sensor_cells["kind"] = "sensor"
stations = pd.concat([ref_cells, sensor_cells]).reset_index(drop=True)
# nearest reference station for each sensor (for calibration)
tree = cKDTree(np.c_[ref_cells.lat * 111, ref_cells.lon * 81])
_, nearest_ref = tree.query(np.c_[stations.lat * 111, stations.lon * 81])
stations["nearest_ref"] = ref_cells.station.values[nearest_ref]

# --- daily calendar and meteorology (ERA5-like, one city-wide state per day) -----------------------
dates = pd.date_range("2019-01-01", periods=365 * N_YEARS + 1, freq="D")  # 2019..2023 inclusive (1,826 days)
dates = dates[dates.year <= 2018 + N_YEARS]
nd = len(dates)
doy = dates.dayofyear.values
year = dates.year.values
heating = np.isin(season_of(doy), ["early_heating", "deep_winter", "late_heating"]).astype(int)

met = pd.DataFrame({"date": dates})
cos_y = np.cos(2 * np.pi * (doy - 15) / 365.25)             # coldest ~ mid January
met["t2m"] = 10 - 14 * cos_y + np.cumsum(rng.normal(0, 1.2, nd)) * 0 + rng.normal(0, 3.0, nd)
ar = np.zeros(nd); e = rng.normal(0, 1, nd)
for i in range(1, nd):
    ar[i] = 0.75 * ar[i - 1] + e[i]
met["t2m"] += 2.5 * ar
met["wind"] = np.clip(1.8 + 0.6 * cos_y + rng.gamma(2.0, 0.5, nd) - 0.2 * ar, 0.2, 9)
met["rh"] = np.clip(60 + 10 * (-cos_y) * (-1) + 8 * rng.normal(0, 1, nd), 15, 100)
met["pressure"] = 922 + 4 * (-cos_y) + rng.normal(0, 3, nd)
met["precip"] = np.where(rng.random(nd) < 0.22, rng.gamma(1.2, 2.5, nd), 0.0)
# boundary-layer height: shallow in winter, and collapses during inversion episodes
inversion = np.zeros(nd); z = 0.0
for i in range(nd):
    p_on = 0.18 if heating[i] else 0.03
    z = 1.0 if (z == 0 and rng.random() < p_on) else (0.0 if (z == 1 and rng.random() < 0.3) else z)
    inversion[i] = z
met["inversion"] = inversion
met["blh"] = np.clip(650 + 450 * (-cos_y) - 350 * inversion + 60 * met.wind + rng.normal(0, 90, nd), 80, 2000)
met["doy"], met["year"], met["heating"] = doy, year, heating
met["season"] = season_of(doy)

# --- true daily PM2.5 at every station (and, later, every cell) -----------------------------------
# climatology: winter mean ~ 90, summer ~ 19 at a typical central site (Kerimray et al. 2020)
clim = 25 + 14 * (1 + cos_y) + 5 * np.clip(cos_y, 0, None) ** 3
meteo_mod = (np.exp(-0.0010 * (met.blh - 600)) *           # shallow BLH -> higher
             np.exp(-0.13 * (met.wind - 2.0)) *              # wind disperses
             np.exp(-0.009 * (met.t2m - 0) * heating) *      # colder winter days -> more coal burnt
             np.exp(-0.03 * met.precip))
episode = 1 + 0.7 * inversion * heating + 0.15 * inversion * (1 - heating)
day_factor = clim * np.clip(meteo_mod, 0.3, 4.0) * episode
# cold-pool strength: zero outside the heating season, peaks in deep winter, stronger on inversion days
coldpool_day = (met.season.values == "deep_winter") * np.clip(cos_y, 0, None) ** 2 * (0.6 + 0.8 * inversion)


def ndvi_effect(ndvi, built_up, heat):
    """Latent greenery effect: negative outside the heating season, weaker in winter,
    flat in dense built-up cells (the structure H1 is designed to detect)."""
    strength = np.where(heat == 1, 0.08, 0.26)
    dense_damp = np.clip(1.4 - 1.6 * built_up, 0.05, 1.0)
    return 1 - strength * dense_damp * (ndvi - 0.3) / 0.3


def simulate_series(cells_df, add_noise=True):
    """Returns long DataFrame (cell_id/station, date, pm25_true) for the given cells.
    Each point gets an unobserved local-source effect and local episode noise, so that
    spatial predictors explain the surface only partly (this is what LOSO is meant to expose)."""
    out = []
    for _, c in cells_df.iterrows():
        local_effect = np.exp(rng.normal(0, 0.12)) if add_noise else 1.0
        loc_ar = np.zeros(nd); ee = rng.normal(0, 0.22, nd)
        for i in range(1, nd):
            loc_ar[i] = 0.6 * loc_ar[i - 1] + ee[i]
        ndvi_day = np.select([met.season == s for s in SEASON_NAMES],
                             [c[f"ndvi_{s}"] for s in SEASON_NAMES])
        base = (day_factor * (c.persistent_part + coldpool_day * c.centre_part) / PNORM
                * ndvi_effect(ndvi_day, c.built_up, heating))
        noise = np.exp(rng.normal(0, 0.26, nd) + loc_ar * heating + 0.3 * loc_ar * (1 - heating)) * local_effect if add_noise else 1.0
        pm = np.clip(base * noise, 2, None)
        out.append(pd.DataFrame({"station": c.get("station", c.cell_id), "cell_id": c.cell_id,
                                 "date": dates, "pm25_true": pm}))
    return pd.concat(out, ignore_index=True)


log("Stage 0: simulating daily series at 43 monitoring points")
obs = simulate_series(stations)
obs = obs.merge(stations[["station", "kind", "nearest_ref", "lat", "lon", "elevation", "dist_chp2", "dist_chp1",
                          "dist_chp_min", "built_up", "low_rise", "road_density", "population"] +
                         [f"ndvi_{s}" for s in SEASON_NAMES]], on="station")
obs = obs.merge(met, on="date")
obs["ndvi"] = np.select([obs.season == s for s in SEASON_NAMES], [obs[f"ndvi_{s}"] for s in SEASON_NAMES])

# reference stations measure the truth with small error; low-cost sensors overestimate at high humidity
is_ref = obs.kind == "reference"
obs["pm25_raw"] = np.where(is_ref,
                           obs.pm25_true * np.exp(rng.normal(0, 0.05, len(obs))),
                           obs.pm25_true * (1.15 + 0.006 * (obs.rh - 50)) * np.exp(rng.normal(0, 0.12, len(obs))))
# completeness rule: ~4 % of station-days fall below 75 % hourly completeness and are dropped
N_RAW = len(obs)
N_RAW_BY_KIND = obs.groupby("kind").size().to_dict()
obs = obs[rng.random(len(obs)) > 0.04].reset_index(drop=True)
N_AFTER_COMPLETENESS_BY_KIND = obs.groupby("kind").size().to_dict()

# --- season-specific linear calibration of each sensor against its nearest reference station -------
log("Stage 0: calibrating low-cost sensors")
ref_series = obs[is_ref.loc[obs.index]][["station", "date", "pm25_raw"]].rename(columns={"station": "nearest_ref", "pm25_raw": "ref_pm"})
obs = obs.merge(ref_series, on=["nearest_ref", "date"], how="left")
obs["pm25"] = obs.pm25_raw.copy()
calib_log = []
for st in stations.station[stations.kind == "sensor"]:
    for s in SEASON_NAMES:
        m = (obs.station == st) & (obs.season == s) & obs.ref_pm.notna()
        if m.sum() < 30:
            continue
        lr = LinearRegression().fit(obs.loc[m, ["pm25_raw"]], obs.loc[m, "ref_pm"])
        obs.loc[(obs.station == st) & (obs.season == s), "pm25"] = np.clip(
            lr.predict(obs.loc[(obs.station == st) & (obs.season == s), ["pm25_raw"]]), 1, None)
        calib_log.append((st, s, lr.coef_[0], lr.intercept_))
calib = pd.DataFrame(calib_log, columns=["station", "season", "slope", "intercept"])
calib.to_csv(f"{OUT}/sensor_calibration.csv", index=False)
obs = obs.drop(columns=["ref_pm"])

# --- satellite proxies: TROPOMI NO2 column and MODIS AOD, missing under winter cloud ----------------
# TROPOMI (5.5 km) and MAIAC AOD (1 km) resolve the city-wide daily load, not the 500 m structure:
# one daily value per day plus a weak road-related term; cloud removes winter days
met["no2_col"] = 0.6 + 0.012 * day_factor + rng.normal(0, 0.15, nd)
met["aod"] = 0.08 + 0.0030 * day_factor + rng.normal(0, 0.05, nd)
cloud_day = rng.random(nd) < np.where(heating == 1, 0.55, 0.20)
met.loc[cloud_day, ["no2_col", "aod"]] = np.nan
obs = obs.merge(met[["date", "no2_col", "aod"]], on="date")
obs["no2_col"] = obs.no2_col + 0.03 * obs.road_density          # passed to the tree model as missing on cloudy days

# --- feature engineering: calendar encodings, lags, rolling means --------------------------------
obs = obs.sort_values(["station", "date"]).reset_index(drop=True)
obs["doy_sin"], obs["doy_cos"] = np.sin(2 * np.pi * obs.doy / 365.25), np.cos(2 * np.pi * obs.doy / 365.25)
obs["dow"] = obs.date.dt.dayofweek
g = obs.groupby("station")["pm25"]
for L in range(1, 8):
    obs[f"lag{L}"] = g.shift(L)
obs["roll7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
obs["aod_ffill"] = obs.groupby("station")["aod"].transform(lambda s: s.ffill())
obs["no2_ffill"] = obs.groupby("station")["no2_col"].transform(lambda s: s.ffill())

MET_FEATS = ["t2m", "wind", "rh", "pressure", "precip", "blh"]
CAL_FEATS = ["doy_sin", "doy_cos", "heating", "dow"]
SAT_FEATS = ["no2_col", "aod"]
SPAT_FEATS = ["elevation", "dist_chp2", "dist_chp1", "dist_chp_min", "built_up", "low_rise", "road_density", "ndvi"]
LAG_FEATS = [f"lag{L}" for L in range(1, 8)] + ["roll7"]
FEATS_TEMPORAL = MET_FEATS + CAL_FEATS + SAT_FEATS + SPAT_FEATS + LAG_FEATS   # Stage 1 forecasting model
FEATS_SURFACE = MET_FEATS + CAL_FEATS + SAT_FEATS + SPAT_FEATS                # Stage 2 surface model (no lags: grid cells have no history)

N_BEFORE_LAGS = len(obs)
data = obs.dropna(subset=LAG_FEATS).reset_index(drop=True)
N_FINAL_BY_KIND = data.groupby("kind").size().to_dict()
data["exceed"] = (data.pm25 > MPC).astype(int)
log(f"modelling table: {len(data):,} station-days, {data.station.nunique()} points, {data.date.min().date()}..{data.date.max().date()}")
data.to_csv(f"{OUT}/station_days.csv.gz", index=False)

# ---- dataset table: what was actually used ---------------------------------------------------------
ds_rows = []
for kind, label in [("reference", "Reference stations"), ("sensor", "Low-cost sensors (calibrated)")]:
    ds_rows.append(dict(network=label, points=int(stations[stations.kind == kind].shape[0]),
                        raw_station_days=int(N_RAW_BY_KIND.get(kind, 0)),
                        removed_completeness=int(N_RAW_BY_KIND.get(kind, 0) - N_AFTER_COMPLETENESS_BY_KIND.get(kind, 0)),
                        removed_lag_warmup=int(N_AFTER_COMPLETENESS_BY_KIND.get(kind, 0) - N_FINAL_BY_KIND.get(kind, 0)),
                        final_station_days=int(N_FINAL_BY_KIND.get(kind, 0))))
pd.DataFrame(ds_rows).to_csv(f"{OUT}/table_dataset.csv", index=False)
dist_rows = []
for s in SEASON_NAMES:
    v = data.loc[data.season == s, "pm25"]
    dist_rows.append(dict(season=SEASON_LABELS[s], n=int(len(v)), mean=v.mean(), median=v.median(),
                          p95=v.quantile(0.95), share_above_mpc=(v > MPC).mean()))
pd.DataFrame(dist_rows).round(3).to_csv(f"{OUT}/table_target_distribution.csv", index=False)
sat_missing = {f: {s: float(data.loc[data.season == s, f].isna().mean()) for s in SEASON_NAMES} for f in SAT_FEATS}
STAGE_TIMES["stage0_simulation_s"] = round(time.time() - T_START, 1)


# =========================================================================== STAGE 1
log("Stage 1: forecasting at monitoring points")


def season_metrics(df, pred_col):
    """RMSE / MAE / R2 for heating and non-heating season, F1 for exceedance of the daily MPC."""
    out = {}
    for name, m in [("heat", df.heating == 1), ("nonheat", df.heating == 0)]:
        y, p = df.loc[m, "pm25"], df.loc[m, pred_col]
        out[f"{name}_rmse"] = np.sqrt(mean_squared_error(y, p))
        out[f"{name}_mae"] = mean_absolute_error(y, p)
        out[f"{name}_r2"] = r2_score(y, p)
    pr, rc, f1, _ = precision_recall_fscore_support(df.exceed, (df[pred_col] > MPC).astype(int),
                                                    average="binary", zero_division=0)
    out.update(exc_precision=pr, exc_recall=rc, exc_f1=f1)
    return out


def fit_xgb(train, feats, n_estimators=None, params=None):
    p = dict(params or XGB_PARAMS)
    if n_estimators:
        p["n_estimators"] = n_estimators
    m = xgb.XGBRegressor(**p)
    m.fit(train[feats], np.log1p(train.pm25))          # log target: multiplicative, right-skewed series
    return m


def predict_xgb(m, df, feats):
    return np.expm1(m.predict(df[feats]))


def fit_predict_lstm(train, test, feats):
    """LSTM baseline on 14-day windows of the same features (satellite gaps forward-filled)."""
    import tensorflow as tf
    from tensorflow import keras
    tf.random.set_seed(SEED)
    f2 = [f.replace("aod", "aod_ffill").replace("no2_col", "no2_ffill") for f in feats if f not in LAG_FEATS]
    f2 = [f for f in f2 if f not in ("roll7",)]
    mu, sd = train[f2 + ["pm25"]].mean(), train[f2 + ["pm25"]].std() + 1e-6

    def windows(df, keep=None):
        X, y, idx = [], [], []
        for st, grp in df.groupby("station"):
            grp = grp.sort_values("date")
            Z = ((grp[f2] - mu[f2]) / sd[f2]).fillna(0).values
            pm = np.log1p(grp.pm25.values)
            pmz = (grp.pm25.values - mu["pm25"]) / sd["pm25"]
            feat = np.c_[Z, np.r_[pmz[0], pmz[:-1]]]          # previous-day PM as an input channel
            for i in range(LSTM_SEQ, len(grp)):
                if keep is not None and grp.index[i] not in keep:
                    continue
                X.append(feat[i - LSTM_SEQ:i]); y.append(pm[i]); idx.append(grp.index[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx)

    Xtr, ytr, _ = windows(train)
    # test windows draw their first 14 days of history from the end of the training period,
    # so the LSTM is scored on exactly the same station-days as the other models
    hist = train.groupby("station").tail(LSTM_SEQ + 1)
    Xte, yte, ite = windows(pd.concat([hist, test]), keep=set(test.index))
    model = keras.Sequential([keras.Input(shape=Xtr.shape[1:]),
                              keras.layers.LSTM(48), keras.layers.Dropout(0.2),
                              keras.layers.Dense(24, activation="relu"), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="mse")
    model.fit(Xtr, ytr, validation_split=0.1, epochs=LSTM_EPOCHS, batch_size=512, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    LSTM_EPOCHS_RUN.append(len(model.history.history["loss"]))
    pred = pd.Series(np.expm1(model.predict(Xte, verbose=0).ravel()), index=ite)
    return pred


LSTM_EPOCHS_RUN = []

# naive baselines
data["pred_persistence"] = data["lag1"]
clim_tab = data.groupby("doy")["pm25"].mean()                    # seasonal climatology (mean per day-of-year)

years = sorted(data.year.unique())
fold_rows, fold_preds = [], []
t_xgb = t_lstm = 0.0
for k, hold in enumerate(years[1:], start=1):                     # four rolling folds, each holds out one full year
    train = data[data.year < hold]
    test = data[data.year == hold].copy()
    # climatology computed from the training years only
    clim_k = train.groupby("doy")["pm25"].mean()
    test["pred_climatology"] = test.doy.map(clim_k).fillna(train.pm25.mean())
    t0 = time.time(); m = fit_xgb(train, FEATS_TEMPORAL); test["pred_xgb"] = predict_xgb(m, test, FEATS_TEMPORAL); t_xgb += time.time() - t0
    if not ARGS.no_lstm:
        t0 = time.time(); test["pred_lstm"] = fit_predict_lstm(train, test, FEATS_TEMPORAL); t_lstm += time.time() - t0
    for model_name in ["xgb", "lstm", "persistence", "climatology"]:
        if f"pred_{model_name}" not in test:
            continue
        sub = test.dropna(subset=[f"pred_{model_name}"])
        fold_rows.append(dict(fold=k, holdout_year=hold, model=model_name, n=len(sub), **season_metrics(sub, f"pred_{model_name}")))
    fold_preds.append(test)
    log(f"  temporal fold {k} (hold-out {hold}) done")
folds = pd.DataFrame(fold_rows)
preds_all = pd.concat(fold_preds)
MODELS = [m for m in ["xgb", "lstm", "persistence", "climatology"] if f"pred_{m}" in preds_all]
n_before_common = len(preds_all)
preds_all = preds_all.dropna(subset=[f"pred_{m}" for m in MODELS])
assert len(preds_all) == n_before_common, "models are not scored on the same test rows"
fold_sizes = [dict(fold=k, holdout_year=int(hold), train_station_days=int((data.year < hold).sum()),
                   test_station_days=int((data.year == hold).sum())) for k, hold in enumerate(years[1:], start=1)]
pd.DataFrame(fold_sizes).to_csv(f"{OUT}/table_fold_sizes.csv", index=False)
STAGE_TIMES["stage1_temporal_folds_s"] = round(time.time() - T_START - sum(STAGE_TIMES.values()), 1)

# leave-one-station-out on the reference stations: the honest figure for the gridded surface
loso_rows = []
for st in stations.station[stations.kind == "reference"]:
    train, test = data[data.station != st], data[data.station == st].copy()
    m = fit_xgb(train, FEATS_SURFACE, params=SURFACE_PARAMS)
    test["pred_xgb_loso"] = predict_xgb(m, test, FEATS_SURFACE)
    loso_rows.append(dict(station=st, **season_metrics(test, "pred_xgb_loso")))
    log(f"  LOSO {st} done")
loso = pd.DataFrame(loso_rows).merge(stations[["station", "dist_chp_min", "elevation", "district"]], on="station")

# Table "models"
cols = ["heat_rmse", "heat_mae", "heat_r2", "nonheat_rmse", "nonheat_r2", "exc_f1", "exc_precision", "exc_recall"]
tab_models = folds.groupby("model")[cols].mean()
tab_models.loc["xgb_loso"] = loso[cols].mean()
order = [m for m in ["xgb", "lstm", "persistence", "climatology", "xgb_loso"] if m in tab_models.index]
tab_models = tab_models.loc[order].rename(index={"xgb": "XGBoost", "lstm": "LSTM", "persistence": "Persistence",
                                                 "climatology": "Climatology", "xgb_loso": "XGBoost, LOSO"})
tab_models["fold_sd_heat_rmse"] = folds.groupby("model")["heat_rmse"].std().rename(
    index={"xgb": "XGBoost", "lstm": "LSTM", "persistence": "Persistence", "climatology": "Climatology"})
tab_models.round(3).to_csv(f"{OUT}/table_models.csv")
folds.round(3).to_csv(f"{OUT}/table_models_by_fold.csv", index=False)
loso.round(3).to_csv(f"{OUT}/table_loso_by_station.csv", index=False)
print("\nTable: held-out error by season (mean of temporal folds)\n", tab_models.round(2).to_string(), "\n")
win_gain = 1 - tab_models.loc["XGBoost", "heat_rmse"] / tab_models.loc["Climatology", "heat_rmse"]
log(f"criterion 1 (beat climatology on winter RMSE): XGBoost gain = {win_gain:.0%} -> {'MET' if win_gain > 0 else 'NOT MET'}")

# ---- uncertainty: moving-block bootstrap over test days (7-day blocks, pooled over the four folds) -----
log(f"Stage 1: block-bootstrap CIs ({N_CI_BOOT} resamples) and Diebold-Mariano tests")
for m in MODELS:
    preds_all[f"se_{m}"] = (preds_all.pm25 - preds_all[f"pred_{m}"]) ** 2
    preds_all[f"hit_{m}"] = (preds_all[f"pred_{m}"] > MPC).astype(int)
preds_all["exceed"] = (preds_all.pm25 > MPC).astype(int)
day_agg = {}
for grp_name, mask in [("heat", preds_all.heating == 1), ("nonheat", preds_all.heating == 0)]:
    d = preds_all[mask].groupby("date")
    agg = d[[f"se_{m}" for m in MODELS]].sum()
    agg["n"] = d.size()
    day_agg[grp_name] = agg.sort_index()
# exceedance counts per day for F1
ex = preds_all.copy()
for m in MODELS:
    ex[f"tp_{m}"] = ex[f"hit_{m}"] * ex.exceed
    ex[f"fp_{m}"] = ex[f"hit_{m}"] * (1 - ex.exceed)
    ex[f"fn_{m}"] = (1 - ex[f"hit_{m}"]) * ex.exceed
ex_day = ex.groupby("date")[[f"{k}_{m}" for m in MODELS for k in ("tp", "fp", "fn")]].sum().sort_index()


def block_indices(n, rs):
    """Moving-block bootstrap index vector of length n with blocks of BLOCK_DAYS consecutive days."""
    starts = rs.integers(0, n - BLOCK_DAYS + 1, size=int(np.ceil(n / BLOCK_DAYS)))
    return (starts[:, None] + np.arange(BLOCK_DAYS)[None, :]).ravel()[:n]


rs_ci = np.random.default_rng(SEED + 7)
boot = {f"{g}_rmse_{m}": [] for g in day_agg for m in MODELS}
boot.update({f"f1_{m}": [] for m in MODELS})
boot.update({f"{g}_drmse_{m}": [] for g in day_agg for m in MODELS if m != "xgb"})
for b in range(N_CI_BOOT):
    for g, agg in day_agg.items():
        idx = block_indices(len(agg), rs_ci)
        sub = agg.iloc[idx].sum()
        r = {m: np.sqrt(sub[f"se_{m}"] / sub["n"]) for m in MODELS}
        for m in MODELS:
            boot[f"{g}_rmse_{m}"].append(r[m])
            if m != "xgb":
                boot[f"{g}_drmse_{m}"].append(r[m] - r["xgb"])
    idx = block_indices(len(ex_day), rs_ci)
    sub = ex_day.iloc[idx].sum()
    for m in MODELS:
        boot[f"f1_{m}"].append(2 * sub[f"tp_{m}"] / max(2 * sub[f"tp_{m}"] + sub[f"fp_{m}"] + sub[f"fn_{m}"], 1))
ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in boot.items()}

# pooled point estimates on the common rows (these are the numbers reported with the CIs)
pooled = {}
for m in MODELS:
    row = {}
    for g, mask in [("heat", preds_all.heating == 1), ("nonheat", preds_all.heating == 0)]:
        y, p_ = preds_all.loc[mask, "pm25"], preds_all.loc[mask, f"pred_{m}"]
        row[f"{g}_rmse"] = np.sqrt(mean_squared_error(y, p_)); row[f"{g}_mae"] = mean_absolute_error(y, p_)
        row[f"{g}_r2"] = r2_score(y, p_)
        row[f"{g}_rmse_lo"], row[f"{g}_rmse_hi"] = ci[f"{g}_rmse_{m}"]
    pr, rc, f1, _ = precision_recall_fscore_support(preds_all.exceed, preds_all[f"hit_{m}"], average="binary", zero_division=0)
    row.update(exc_precision=pr, exc_recall=rc, exc_f1=f1, exc_f1_lo=ci[f"f1_{m}"][0], exc_f1_hi=ci[f"f1_{m}"][1],
               fold_sd_heat_rmse=folds[folds.model == m].heat_rmse.std(), fold_sd_nonheat_rmse=folds[folds.model == m].nonheat_rmse.std())
    pooled[m] = row
tab_pooled = pd.DataFrame(pooled).T


def dm_test(loss_a, loss_b, h=BLOCK_DAYS):
    """Diebold-Mariano test on a daily loss differential with Newey-West variance and the
    Harvey-Leybourne-Newbold small-sample correction; two-sided p from Student t(n-1)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d); dbar = d.mean(); dc = d - dbar
    gamma = [np.dot(dc[k:], dc[:n - k]) / n for k in range(h + 1)]
    var = gamma[0] + 2 * sum((1 - k / (h + 1)) * gamma[k] for k in range(1, h + 1))
    stat = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat *= hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1)), n


dm_rows = []
for g, agg in day_agg.items():
    for m in MODELS:
        if m == "xgb":
            continue
        la = agg[f"se_{m}"] / agg["n"]; lb = agg["se_xgb"] / agg["n"]      # daily MSE across points
        st_, p_, n_ = dm_test(la, lb)
        lo, hi = ci[f"{g}_drmse_{m}"]
        dm_rows.append(dict(season=g, baseline=m, n_days=n_, delta_rmse=pooled[m][f"{g}_rmse"] - pooled["xgb"][f"{g}_rmse"],
                            delta_lo=lo, delta_hi=hi, dm_stat=st_, p_value=p_))
dm = pd.DataFrame(dm_rows)
# Holm correction over all comparisons
order_p = np.argsort(dm.p_value.values); k_tot = len(dm); adj = np.empty(k_tot); run = 0
for rank_i, j in enumerate(order_p):
    run = max(run, min(1.0, (k_tot - rank_i) * dm.p_value.values[j])); adj[j] = run
dm["p_holm"] = adj
dm.to_csv(f"{OUT}/table_dm_tests.csv", index=False)
tab_pooled.to_csv(f"{OUT}/table_models_pooled_ci.csv")
print("\nPooled metrics with 95 % block-bootstrap CI\n", tab_pooled.round(3).to_string())
print("\nDiebold-Mariano tests (XGBoost vs baseline; positive stat = baseline has larger loss)\n", dm.round(4).to_string(), "\n")

# per-season exceedance metrics and base rates (the 35 ug/m3 label is unbalanced in winter)
exc_rows = []
for s in SEASON_NAMES:
    sub = preds_all[preds_all.season == s]
    for m in MODELS:
        pr, rc, f1, _ = precision_recall_fscore_support(sub.exceed, sub[f"hit_{m}"], average="binary", zero_division=0)
        exc_rows.append(dict(season=SEASON_LABELS[s], model=m, n=len(sub), base_rate=sub.exceed.mean(),
                             precision=pr, recall=rc, f1=f1))
pd.DataFrame(exc_rows).round(3).to_csv(f"{OUT}/table_exceedance_by_season.csv", index=False)

# ---- error analysis ------------------------------------------------------------------------------
cm = pd.crosstab(preds_all.exceed, preds_all.hit_xgb).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
cm.to_csv(f"{OUT}/table_confusion_xgb.csv")
cm_clim = pd.crosstab(preds_all.exceed, preds_all.hit_climatology).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
cm_clim.to_csv(f"{OUT}/table_confusion_climatology.csv")
# day within an inversion episode (0 = no inversion, 1 = onset day, 2, 3, 4+)
met_ep = met[["date", "inversion"]].sort_values("date").copy()
run_len = []; c = 0
for v in met_ep.inversion.values:
    c = c + 1 if v == 1 else 0; run_len.append(c)
met_ep["episode_day"] = np.minimum(run_len, 4)
preds_all = preds_all.drop(columns=["episode_day"], errors="ignore").merge(met_ep[["date", "episode_day"]], on="date", how="left")
ep_rows = []
for ed in range(5):
    sub = preds_all[(preds_all.episode_day == ed) & (preds_all.heating == 1)]
    row = dict(episode_day=ed, n=len(sub))
    for m in MODELS:
        row[f"mae_{m}"] = (sub.pm25 - sub[f"pred_{m}"]).abs().mean()
        row[f"bias_{m}"] = (sub[f"pred_{m}"] - sub.pm25).mean()
    ex_sub = sub[sub.exceed == 1]
    row["recall_xgb"] = ex_sub.hit_xgb.mean() if len(ex_sub) else np.nan
    ep_rows.append(row)
ep_tab = pd.DataFrame(ep_rows)
ep_tab.round(3).to_csv(f"{OUT}/table_error_by_episode_day.csv", index=False)
preds_all["abs_err_xgb"] = (preds_all.pm25 - preds_all.pred_xgb).abs()
thr = preds_all.abs_err_xgb.quantile(0.99)
worst = preds_all[preds_all.abs_err_xgb >= thr]
worst_summary = dict(n=int(len(worst)), threshold_ugm3=float(thr),
                     share_heating=float((worst.heating == 1).mean()),
                     share_deep_winter=float((worst.season == "deep_winter").mean()),
                     share_inversion_day=float((worst.episode_day > 0).mean()),
                     share_onset_day=float((worst.episode_day == 1).mean()),
                     share_underprediction=float((worst.pred_xgb < worst.pm25).mean()),
                     share_sensor=float((worst.kind == "sensor").mean()),
                     share_sensor_all=float((preds_all.kind == "sensor").mean()),
                     share_inversion_all_heating=float((preds_all[preds_all.heating == 1].episode_day > 0).mean()))
worst.sort_values("abs_err_xgb", ascending=False).head(20)[["station", "kind", "date", "season", "episode_day", "pm25", "pred_xgb"]].to_csv(f"{OUT}/table_worst_station_days.csv", index=False)
print("Worst 1 % station-days:", json.dumps(worst_summary, indent=1))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1, 1.35]))
cmv = cm.values; cmn = cmv / cmv.sum(axis=1, keepdims=True)
axes[0].imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f"{cmv[i, j]:,}\n({cmn[i, j]:.0%})", ha="center", va="center", fontsize=8,
                     color="white" if cmn[i, j] > 0.5 else "black")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels([f"<{MPC:.0f}", f">{MPC:.0f}"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels([f"<{MPC:.0f}", f">{MPC:.0f}"])
axes[0].set_xlabel("Predicted, $\\mu$g/m$^3$"); axes[0].set_ylabel("Observed, $\\mu$g/m$^3$")
axes[0].set_title("(a) XGBoost exceedance", fontsize=9)
labels_m = {"xgb": "XGBoost", "lstm": "LSTM", "persistence": "Persistence", "climatology": "Climatology"}
styles = {"xgb": ("tab:blue", "o"), "lstm": ("tab:orange", "s"), "persistence": ("tab:green", "^"), "climatology": ("grey", "D")}
for m in MODELS:
    axes[1].plot(ep_tab.episode_day, ep_tab[f"mae_{m}"], marker=styles[m][1], color=styles[m][0], lw=1.2, ms=4, label=labels_m[m])
axes[1].set_xticks(range(5)); axes[1].set_xticklabels(["none", "1 (onset)", "2", "3", "4+"])
axes[1].set_xlabel("Day of inversion episode (heating season)"); axes[1].set_ylabel("MAE, $\\mu$g/m$^3$")
axes[1].set_title("(b) Error by episode day", fontsize=9); axes[1].legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_errors.pdf"); fig.savefig(f"{OUT}/fig_errors.png", dpi=160); plt.close(fig)

# ---- ablation of feature groups (XGBoost, same temporal folds) -------------------------------------
log("Stage 1: feature-group ablation")
ABLATIONS = {"Full model": FEATS_TEMPORAL,
             "No lag features": [f for f in FEATS_TEMPORAL if f not in LAG_FEATS],
             "No satellite features": [f for f in FEATS_TEMPORAL if f not in SAT_FEATS],
             "No calendar / heating features": [f for f in FEATS_TEMPORAL if f not in CAL_FEATS],
             "No spatial features (incl. NDVI)": [f for f in FEATS_TEMPORAL if f not in SPAT_FEATS]}
abl_rows = []
for name, feats in ABLATIONS.items():
    rr = []
    for k, hold in enumerate(years[1:], start=1):
        train, test = data[data.year < hold], data[data.year == hold].copy()
        if name == "Full model":
            test["p"] = preds_all.loc[preds_all.year == hold].set_index(["station", "date"]).loc[
                list(zip(test.station, test.date)), "pred_xgb"].values
        else:
            test["p"] = predict_xgb(fit_xgb(train, feats), test, feats)
        rr.append(season_metrics(test, "p"))
    rr = pd.DataFrame(rr)
    abl_rows.append(dict(variant=name, n_features=len(feats), heat_rmse=rr.heat_rmse.mean(), heat_rmse_sd=rr.heat_rmse.std(),
                         nonheat_rmse=rr.nonheat_rmse.mean(), nonheat_rmse_sd=rr.nonheat_rmse.std(), exc_recall=rr.exc_recall.mean()))
    log(f"  ablation '{name}' done")
abl = pd.DataFrame(abl_rows)
abl.round(3).to_csv(f"{OUT}/table_ablation.csv", index=False)
print("\nAblation\n", abl.round(2).to_string(), "\n")
STAGE_TIMES["stage1_total_s"] = round(time.time() - T_START - STAGE_TIMES["stage0_simulation_s"], 1)

# Figure: held-out year at a central station
central = "REF1"
last_year = years[-1]
ts = preds_all[(preds_all.station == central) & (preds_all.year == last_year)].sort_values("date")
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(ts.doy, ts.pm25, color="black", lw=1.0, label="Observed")
ax.plot(ts.doy, ts.pred_xgb, color="tab:blue", lw=1.0, label="XGBoost")
ax.plot(ts.doy, ts.pred_climatology, color="grey", lw=1.5, ls="--", label="Seasonal climatology")
ax.axhline(MPC, color="tab:red", lw=0.9, ls=":", label=f"MPC {MPC:.0f} $\\mu$g/m$^3$")
ax.axvspan(1, 105, color="grey", alpha=0.12); ax.axvspan(288, 366, color="grey", alpha=0.12)
ax.text(8, ax.get_ylim()[1] * 0.93 if ax.get_ylim()[1] > 0 else 200, "heating season", color="dimgrey", fontsize=9)
ax.set_xlabel("Day of year (held-out year)"); ax.set_ylabel("PM$_{2.5}$, $\\mu$g/m$^3$"); ax.set_xlim(1, 366)
ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_timeseries.pdf"); fig.savefig(f"{OUT}/fig_timeseries.png", dpi=160); plt.close(fig)


# =========================================================================== STAGE 2
log("Stage 2: spatial generalisation to the 500 m grid")
surface_model = fit_xgb(data, FEATS_SURFACE, params=SURFACE_PARAMS)


def grid_day_table(cells, day_rows):
    """Cartesian product of grid cells x selected days with the surface features."""
    c = cells[["cell_id", "elevation", "dist_chp2", "dist_chp1", "dist_chp_min", "built_up", "low_rise",
               "road_density"] + [f"ndvi_{s}" for s in SEASON_NAMES]]
    d = day_rows[["date", "season"] + MET_FEATS + ["doy", "heating"]].copy()
    d["doy_sin"], d["doy_cos"] = np.sin(2 * np.pi * d.doy / 365.25), np.cos(2 * np.pi * d.doy / 365.25)
    d["dow"] = d.date.dt.dayofweek
    X = c.merge(d, how="cross")
    X["ndvi"] = np.select([X.season == s for s in SEASON_NAMES], [X[f"ndvi_{s}"] for s in SEASON_NAMES])
    X = X.drop(columns=[f"ndvi_{s}" for s in SEASON_NAMES])
    # satellite proxies for grid cells: the daily city-wide value plus the cell's road term (cloud -> missing)
    X = X.merge(met[["date", "no2_col", "aod"]], on="date", how="left")
    X["no2_col"] = X.no2_col + 0.03 * X.road_density
    return X


# seasonal mean surfaces from all days of the target period
def seasonal_surface(model, cells, days, chunk=40):
    """Mean predicted PM2.5 per cell and season, computed in chunks of days to bound memory."""
    acc = None
    for i in range(0, len(days), chunk):
        X = grid_day_table(cells, days.iloc[i:i + chunk])
        X["pred"] = predict_xgb(model, X, FEATS_SURFACE)
        part = X.groupby(["season", "cell_id"])["pred"].agg(["sum", "count"])
        acc = part if acc is None else acc.add(part, fill_value=0)
    return (acc["sum"] / acc["count"]).unstack("season")[SEASON_NAMES]


days_all = met[met.year.isin(years)]
surf = seasonal_surface(surface_model, grid, days_all)             # (cells x 4 seasons)
surf.to_csv(f"{OUT}/seasonal_surfaces.csv")

# spatial weights: queen contiguity on the masked grid via the full lattice
W_full = lat2W(GRID_NY, GRID_NX, rook=False)
full_ids = (grid.row * GRID_NX + grid.col).values
pos = {fid: i for i, fid in enumerate(full_ids)}
neigh = {i: [pos[j] for j in W_full.neighbors[fid] if j in pos] for i, fid in enumerate(full_ids)}
from libpysal.weights import W as Wclass
W = Wclass(neigh, silence_warnings=True); W.transform = "B"   # binary contiguity, as in Getis & Ord (1992)


def hotspot_flags(values, z_crit=1.96):
    """Gi* with 95 % confidence (z > 1.96) by default; returns boolean high-value cluster flags and z-scores."""
    gl = G_Local(np.asarray(values, dtype=float), W, transform="B", star=True, permutations=0)
    return (gl.Zs > z_crit), gl.Zs


hot_rows, flags = [], {}
for s in SEASON_NAMES:
    mi = Moran(surf[s].values, W, transformation="r", permutations=199)
    flag, z = hotspot_flags(surf[s].values)
    flags[s] = flag
    flag99, _ = hotspot_flags(surf[s].values, z_crit=2.576)
    hot_rows.append(dict(season=SEASON_LABELS[s], mean_pm25=surf[s].mean(), sd_pm25=surf[s].std(),
                         morans_I=mi.I, morans_p=mi.p_sim, gi_cells=int(flag.sum()), gi_cells_99=int(flag99.sum())))
tab_hot = pd.DataFrame(hot_rows).set_index("season")

# bootstrap stability: retrain on resampled training days, flag again, keep cells flagged in >= 70 % of resamples
log(f"Stage 2: bootstrap stability ({N_BOOT} resamples)")
sub_days = pd.concat([days_all[days_all.season == s].sample(24, random_state=SEED) for s in SEASON_NAMES])
boot_counts = {s: np.zeros(len(grid)) for s in SEASON_NAMES}
uniq_days = data.date.unique()
for b in range(N_BOOT):
    rs = np.random.default_rng(SEED + 100 + b)
    bdays = rs.choice(uniq_days, size=len(uniq_days), replace=True)
    btrain = data.set_index("date").loc[bdays].reset_index()
    bm = fit_xgb(btrain, FEATS_SURFACE, n_estimators=250, params=SURFACE_PARAMS)
    bsurf = seasonal_surface(bm, grid, sub_days)
    for s in SEASON_NAMES:
        f, _ = hotspot_flags(bsurf[s].values)
        boot_counts[s] += f
    if (b + 1) % 10 == 0:
        log(f"  bootstrap {b+1}/{N_BOOT}")
stable = {s: (boot_counts[s] / N_BOOT) >= STABILITY_THRESHOLD for s in SEASON_NAMES}
tab_hot["stable_share"] = [((flags[s] & stable[s]).sum() / max(flags[s].sum(), 1)) for s in SEASON_NAMES]
tab_hot.round(3).to_csv(f"{OUT}/table_hotspots.csv")
print("\nTable: seasonal hotspot statistics\n", tab_hot.round(3).to_string(), "\n")
log(f"criterion 2 (significant positive autocorrelation): min Moran's I = {tab_hot.morans_I.min():.2f}, "
    f"max p = {tab_hot.morans_p.max():.3f} -> {'MET' if (tab_hot.morans_p.max() < 0.05 and tab_hot.morans_I.min() > 0) else 'NOT MET'}")

# persistent (>= 2 of 4 seasons, stable) vs winter-only (deep winter alone)
for s in SEASON_NAMES:
    grid[f"hot_{s}"] = flags[s] & stable[s]
n_flag = grid[[f"hot_{s}" for s in SEASON_NAMES]].sum(axis=1)
grid["persistent"] = n_flag >= 2
grid["winter_only"] = (n_flag == 1) & grid["hot_deep_winter"]
grid["hotspot_class"] = np.select([grid.persistent, grid.winter_only], ["persistent", "winter_only"], "none")
grid["persistence"] = n_flag / 4.0
cls = grid.hotspot_class.value_counts()
# sensitivity of the classification to the Gi* threshold (99 %, no bootstrap filter: upper bound on the change)
f99 = {s: hotspot_flags(surf[s].values, z_crit=2.576)[0] & stable[s] for s in SEASON_NAMES}
n99 = sum(f99[s].astype(int) for s in SEASON_NAMES)
class99 = dict(persistent=int((n99 >= 2).sum()), winter_only=int(((n99 == 1) & f99["deep_winter"]).sum()))
class_by_district = pd.crosstab(grid.district, grid.hotspot_class)
class_by_district.to_csv(f"{OUT}/table_hotspot_class_by_district.csv")
log(f"persistent cells: {cls.get('persistent', 0)} ({cls.get('persistent', 0)/len(grid):.1%}), "
    f"winter-only: {cls.get('winter_only', 0)} ({cls.get('winter_only', 0)/len(grid):.1%})")
print(pd.crosstab(grid.district, grid.hotspot_class), "\n")

# ---- RQ1: SHAP dependence of predicted PM2.5 on NDVI by season and built-up class ------------------
log("Stage 2: SHAP attribution")
X_shap = grid_day_table(grid, sub_days).sample(130000, random_state=SEED).reset_index(drop=True)
X_shap["pred"] = predict_xgb(surface_model, X_shap, FEATS_SURFACE)
explainer = shap.TreeExplainer(surface_model)
sv = explainer.shap_values(X_shap[FEATS_SURFACE])
sv = pd.DataFrame(sv, columns=FEATS_SURFACE)
# log-space attributions converted to ug/m3 at the prediction: d(pm) ~= pm * d(log pm)
sv_ndvi = sv["ndvi"].values * X_shap["pred"].values
X_shap["shap_ndvi"] = sv_ndvi
X_shap["dense"] = X_shap.built_up > grid.built_up.median()
X_shap["heat"] = X_shap.heating == 1
cell_shap = X_shap.groupby(["cell_id", "heat", "dense"]).agg(ndvi=("ndvi", "mean"), shap_ndvi=("shap_ndvi", "mean")).reset_index()

cell_shap4 = X_shap.groupby(["cell_id", "season", "dense"]).agg(ndvi=("ndvi", "mean"), shap_ndvi=("shap_ndvi", "mean"),
                                                            pred=("pred", "mean")).reset_index()


def ndvi_effect_est(d):
    """Change in predicted PM2.5 for +0.3 NDVI from a quadratic fit of the SHAP dependence, evaluated
    across the 5-95 % NDVI range of that season and class (ug/m3), and as % of the class mean prediction."""
    c2 = np.polyfit(d.ndvi, d.shap_ndvi, 2)
    lo, hi = d.ndvi.quantile([0.05, 0.95])
    eff = (np.polyval(c2, hi) - np.polyval(c2, lo)) * 0.3 / (hi - lo)
    return eff, 100 * eff / d.pred.mean(), lo, hi


rs_h1 = np.random.default_rng(SEED + 11)
N_H1_BOOT = 200 if ARGS.quick else 1000
h1_rows, boot_rel = [], {}
for s in SEASON_NAMES:
    for dense, label in [(False, "Low built-up"), (True, "Dense built-up")]:
        d = cell_shap4[(cell_shap4.season == s) & (cell_shap4.dense == dense)].reset_index(drop=True)
        eff, rel, lo, hi = ndvi_effect_est(d)
        bs_abs, bs_rel = [], []
        for b in range(N_H1_BOOT):
            db = d.iloc[rs_h1.integers(0, len(d), len(d))]
            e_, r_, _, _ = ndvi_effect_est(db)
            bs_abs.append(e_); bs_rel.append(r_)
        boot_rel[(s, label)] = np.array(bs_rel)
        # known latent effect of the simulation for the same cells (+0.3 NDVI, % of the cell's level):
        # available only because the data are synthetic; used to check whether the attribution recovers it
        gc = grid.set_index("cell_id").loc[d.cell_id]
        heat_flag = 0 if s == "non_heating" else 1
        strength = 0.08 if heat_flag else 0.26
        damp = np.clip(1.4 - 1.6 * gc.built_up.values, 0.05, 1.0)
        base_f = 1 - strength * damp * (gc[f"ndvi_{s}"].values - 0.3) / 0.3
        true_pct = float(np.mean(-100 * strength * damp / base_f))
        h1_rows.append(dict(season=SEASON_LABELS[s], built_up=label, true_effect_pct=true_pct, n_cells=len(d), ndvi_p05=lo, ndvi_p95=hi,
                            mean_pred=d.pred.mean(), effect_ugm3=eff, effect_lo=np.percentile(bs_abs, 2.5),
                            effect_hi=np.percentile(bs_abs, 97.5), effect_pct=rel,
                            pct_lo=np.percentile(bs_rel, 2.5), pct_hi=np.percentile(bs_rel, 97.5)))
h1 = pd.DataFrame(h1_rows)
# H1 contrast: relative effect in each heating sub-season minus the non-heating effect (same class);
# positive = weaker (less negative) association than outside the heating season
h1_contrast = []
for s in ["early_heating", "deep_winter", "late_heating"]:
    for label in ["Low built-up", "Dense built-up"]:
        diff = boot_rel[(s, label)] - boot_rel[("non_heating", label)]
        point = float(h1[(h1.season == SEASON_LABELS[s]) & (h1.built_up == label)].effect_pct.iloc[0]
                      - h1[(h1.season == "Non-heating") & (h1.built_up == label)].effect_pct.iloc[0])
        h1_contrast.append(dict(season=SEASON_LABELS[s], built_up=label, diff_pct_points=point,
                                lo=np.percentile(diff, 2.5), hi=np.percentile(diff, 97.5), p_boot_two_sided=float(min(1, 2 * min((diff <= 0).mean(), (diff >= 0).mean())))))
h1_contrast = pd.DataFrame(h1_contrast)
h1.round(3).to_csv(f"{OUT}/table_ndvi_effect.csv", index=False)
h1_contrast.round(4).to_csv(f"{OUT}/table_h1_contrast.csv", index=False)
print("NDVI effect (+0.3) by season and built-up class\n", h1.round(2).to_string(), "\n", h1_contrast.round(3).to_string(), "\n")
# NDVI and sub-season: how strongly NDVI tracks the heating sub-period in the attribution sample
ndvi_season_corr = float(np.corrcoef(X_shap.ndvi[X_shap.heat], X_shap.doy_cos[X_shap.heat])[0, 1])

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9), sharey=True)
for ax, s in zip(axes, ["non_heating", "early_heating", "deep_winter", "late_heating"]):
    for dense, color, label in [(False, "tab:green", "Low built-up"), (True, "tab:brown", "Dense built-up")]:
        d = cell_shap4[(cell_shap4.season == s) & (cell_shap4.dense == dense)]
        bins = pd.cut(d.ndvi, 25)
        bd = d.groupby(bins, observed=True).agg(ndvi=("ndvi", "mean"), shap=("shap_ndvi", "mean")).dropna()
        ax.scatter(bd.ndvi, bd.shap, s=8, color=color, alpha=0.55)
        c2 = np.polyfit(d.ndvi, d.shap_ndvi, 2)
        lo, hi = d.ndvi.quantile([0.05, 0.95]); xs = np.linspace(lo, hi, 60)
        ax.plot(xs, np.polyval(c2, xs), color=color, lw=1.8, label=label)
    ax.axhline(0, color="black", lw=0.6); ax.set_title(SEASON_LABELS[s], fontsize=8.5)
    ax.set_xlabel("NDVI", fontsize=8); ax.tick_params(labelsize=7)
axes[0].set_ylabel("SHAP of NDVI, $\\mu$g/m$^3$", fontsize=8); axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_shap_ndvi.pdf"); fig.savefig(f"{OUT}/fig_shap_ndvi.png", dpi=160); plt.close(fig)
slopes = {(r.season, r.built_up): float(r.effect_ugm3) for r in h1.itertuples()}
shap_rank = sv.abs().mean().sort_values(ascending=False)
shap_rank.round(4).to_csv(f"{OUT}/shap_importance_surface_model.csv")
print("Mean |SHAP| ranking (surface model):\n", shap_rank.head(8).round(4).to_string(), "\n")

# temporal model attribution ranking (lags, BLH, temperature, heating flag ...)
m_temp = fit_xgb(data, FEATS_TEMPORAL)
sv_t = pd.DataFrame(shap.TreeExplainer(m_temp).shap_values(data[FEATS_TEMPORAL].sample(6000, random_state=SEED)), columns=FEATS_TEMPORAL)
sv_t.abs().mean().sort_values(ascending=False).round(4).to_csv(f"{OUT}/shap_importance_temporal_model.csv")

STAGE_TIMES["stage2_s"] = round(time.time() - T_START - STAGE_TIMES["stage0_simulation_s"] - STAGE_TIMES["stage1_total_s"], 1)
# =========================================================================== STAGE 3
log("Stage 3: AHP prioritisation")
CRITERIA = ["persistence", "population", "ndvi_deficit", "land_available"]
# AHP pairwise comparison matrix (row i vs column j, Saaty scale): persistence > exposure > NDVI deficit > land
A = np.array([[1, 2, 3, 4],
              [1 / 2, 1, 2, 2],
              [1 / 3, 1 / 2, 1, 2],
              [1 / 4, 1 / 2, 1 / 2, 1]], dtype=float)
eigvals, eigvecs = np.linalg.eig(A)
imax = eigvals.real.argmax()
w = np.abs(eigvecs[:, imax].real); w = w / w.sum()
lam_max = eigvals.real[imax]
CI = (lam_max - 4) / 3; RI = 0.90; CR = CI / RI
weights = dict(zip(CRITERIA, w))
log(f"AHP weights: " + ", ".join(f"{k}={v:.2f}" for k, v in weights.items()) + f"; CR = {CR:.3f} ({'accepted' if CR < 0.10 else 'REJECTED'})")

grid["ndvi_deficit"] = 1 - grid.ndvi_non_heating
crit = grid[CRITERIA].copy()
crit = (crit - crit.min()) / (crit.max() - crit.min())             # min-max scaling to [0, 1]
crit = crit[grid.land_available > 0.05]                             # cells with no plantable land are excluded


def priority(wd):
    return sum(wd[k] * crit[k] for k in CRITERIA)


grid["priority"] = priority(weights)
grid.loc[~grid.index.isin(crit.index), "priority"] = np.nan
rank = grid.priority.rank(ascending=False)
grid["rank"] = rank
top100 = grid[rank <= 100]
dist_tab = pd.crosstab(top100.district, top100.hotspot_class).reindex(columns=["persistent", "winter_only", "none"], fill_value=0)
dist_tab["total"] = dist_tab.sum(axis=1)
dist_tab = dist_tab.sort_values("total", ascending=False)
dist_tab.to_csv(f"{OUT}/table_top100_by_district.csv")
print("Top-100 cells by district and hotspot class\n", dist_tab.to_string(), "\n")

# sensitivity: perturb each weight by +/-20 % (renormalised), compare rankings with Spearman's rho
base_rank = grid.loc[crit.index, "priority"].rank(ascending=False)
base_top20 = set(base_rank[base_rank <= 20].index)
sens_rows = []
for k in CRITERIA:
    for f in (0.8, 1.2):
        wd = dict(weights); wd[k] *= f
        tot = sum(wd.values()); wd = {kk: v / tot for kk, v in wd.items()}
        r = priority(wd).rank(ascending=False)
        rho = stats.spearmanr(base_rank, r).correlation
        overlap = len(base_top20 & set(r[r <= 20].index))
        sens_rows.append(dict(criterion=k, factor=f, spearman_rho=rho, top20_overlap=overlap))
# random joint perturbations
for i in range(50):
    rs = np.random.default_rng(SEED + 500 + i)
    wd = {k: v * rs.uniform(0.8, 1.2) for k, v in weights.items()}
    tot = sum(wd.values()); wd = {kk: v / tot for kk, v in wd.items()}
    r = priority(wd).rank(ascending=False)
    sens_rows.append(dict(criterion="joint_random", factor=np.nan, spearman_rho=stats.spearmanr(base_rank, r).correlation,
                          top20_overlap=len(base_top20 & set(r[r <= 20].index))))
sens = pd.DataFrame(sens_rows)
sens.round(4).to_csv(f"{OUT}/table_sensitivity.csv", index=False)
rho_med, rho_min, ov_min = sens.spearman_rho.median(), sens.spearman_rho.min(), int(sens.top20_overlap.min())
log(f"criterion 3 / H2: Spearman rho median {rho_med:.3f}, min {rho_min:.3f}; top-20 overlap min {ov_min} -> "
    f"{'MET' if rho_min > 0.8 else 'NOT MET (report as tiers)'}")

# tiers at cell level (strict order is reported at district level)
grid["tier"] = pd.cut(grid["rank"], [0, 100, 300, len(grid)], labels=["Tier 1", "Tier 2", "Tier 3"]).astype(object)
grid.loc[grid.priority.isna(), "tier"] = "Excluded (no plantable land)"
n_excluded = int(grid.priority.isna().sum())
sens_by_crit = sens[sens.criterion != "joint_random"].copy()
sens_by_crit.round(4).to_csv(f"{OUT}/table_sensitivity_by_criterion.csv", index=False)
# how strongly the criteria agree with each other among scored cells
crit_corr = crit.corr(method="spearman")
crit_corr.round(3).to_csv(f"{OUT}/table_criteria_correlation.csv")
top100_classes = top100.hotspot_class.value_counts().to_dict()
grid.to_csv(f"{OUT}/grid_results.csv", index=False)

# figure: top-100 cells by district, split by hotspot class
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(dist_tab))
ax.bar(x, dist_tab.persistent, color="tab:red", alpha=0.85, label="Persistent hotspot cells")
ax.bar(x, dist_tab.winter_only, bottom=dist_tab.persistent, color="tab:blue", alpha=0.75, label="Winter-only hotspot cells")
ax.bar(x, dist_tab.none, bottom=dist_tab.persistent + dist_tab.winter_only, color="lightgrey", label="Other")
for i, t in enumerate(dist_tab.total):
    ax.text(i, t + 0.5, str(int(t)), ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(dist_tab.index, rotation=30, ha="right")
ax.set_ylabel("Cells in top-100"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_districts.pdf"); fig.savefig(f"{OUT}/fig_districts.png", dpi=160); plt.close(fig)

# map of hotspot classes and priority score (vector PDF)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
cmap_cls = {"none": "#e6e6e6", "winter_only": "#4c72b0", "persistent": "#c44e52"}
for k, lab in [("none", "No hotspot"), ("winter_only", "Winter-only"), ("persistent", "Persistent")]:
    g_ = grid[grid.hotspot_class == k]
    axes[0].scatter(g_.lon, g_.lat, c=cmap_cls[k], s=7.5, marker="s", lw=0, label=lab)
for n, (la, lo) in CHP.items():
    axes[0].plot(lo, la, "k^", ms=6); axes[0].text(lo + 0.004, la, n, fontsize=7)
axes[0].set_title("(a) Hotspot class", fontsize=9); axes[0].legend(frameon=True, framealpha=1, edgecolor="none", fontsize=7, loc="lower left", markerscale=1.5)
sc = axes[1].scatter(grid.lon, grid.lat, c=grid.priority, s=7.5, marker="s", lw=0, cmap="YlOrRd")
axes[1].scatter(top100.lon, top100.lat, facecolors="none", edgecolors="black", s=7, marker="s", lw=0.4)
axes[1].set_title("(b) Priority score $P_j$, top-100 outlined", fontsize=9)
cb = plt.colorbar(sc, ax=axes[1], fraction=0.04); cb.set_label("$P_j$ (unitless)", fontsize=8)
for ax in axes:
    ax.set_aspect(1 / np.cos(np.radians(43.25))); ax.set_xlabel("Longitude, °E", fontsize=8); ax.set_ylabel("Latitude, °N", fontsize=8)
    ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_maps.pdf"); fig.savefig(f"{OUT}/fig_maps.png", dpi=160); plt.close(fig)
STAGE_TIMES["stage3_s"] = round(time.time() - T_START - sum(v for k, v in STAGE_TIMES.items() if k != "stage1_temporal_folds_s"), 1)

# =========================================================================== SUMMARY
top3 = dist_tab.total.head(3)
summary = {
    "data": {"station_days": int(len(data)), "points": int(data.station.nunique()), "years": [int(y) for y in years],
             "grid_cells": int(len(grid)), "note": "SIMULATED dataset tuned to published Almaty statistics"},
    "stage1": {"winter_rmse_gain_vs_climatology": round(float(win_gain), 3),
               "xgb_vs_persistence_winter": round(float(1 - tab_models.loc['XGBoost', 'heat_rmse'] / tab_models.loc['Persistence', 'heat_rmse']), 3),
               "exceedance_recall": round(float(tab_models.loc["XGBoost", "exc_recall"]), 3),
               "exceedance_precision": round(float(tab_models.loc["XGBoost", "exc_precision"]), 3),
               "lstm_minus_xgb_winter_rmse": (round(float(tab_models.loc["LSTM", "heat_rmse"] - tab_models.loc["XGBoost", "heat_rmse"]), 2) if "LSTM" in tab_models.index else None),
               "xgb_time_s": round(t_xgb, 1), "lstm_time_s": round(t_lstm, 1),
               "table": tab_models.round(3).to_dict()},
    "stage2": {"table": tab_hot.round(3).to_dict(), "persistent_cells": int(cls.get("persistent", 0)),
               "winter_only_cells": int(cls.get("winter_only", 0)),
               "ndvi_plus0.3_effect_ugm3": {f"{k[0]} / {k[1]}": round(v, 2) for k, v in slopes.items()},
               "top_shap_features_surface": shap_rank.head(6).round(4).to_dict()},
    "stage3": {"ahp_weights": {k: round(float(v), 3) for k, v in weights.items()}, "consistency_ratio": round(float(CR), 3),
               "top100_by_district": dist_tab.total.to_dict(), "top3_districts_share": int(top3.sum()),
               "spearman_median": round(float(rho_med), 3), "spearman_min": round(float(rho_min), 3), "top20_overlap_min": ov_min},
}
summary["data"].update(raw_station_days=int(N_RAW), removed_completeness=int(N_RAW - N_BEFORE_LAGS),
                       removed_lag_warmup=int(N_BEFORE_LAGS - len(data)), fold_sizes=fold_sizes,
                       satellite_missing_share=sat_missing, n_sensor_calibrations=int(len(calib)),
                       calib_slope_range=[round(float(calib.slope.min()), 3), round(float(calib.slope.max()), 3)])
summary["stage1"].update(lstm_epochs_run=LSTM_EPOCHS_RUN, pooled=tab_pooled.round(3).to_dict(orient="index"),
                         dm_tests=dm.round(5).to_dict(orient="records"), worst_1pct=worst_summary,
                         ablation=abl.round(3).to_dict(orient="records"), loso_by_station=loso.round(3).to_dict(orient="records"))
summary["stage2"].update(class_99=class99, ndvi_effect=h1.round(3).to_dict(orient="records"),
                         h1_contrast=h1_contrast.round(4).to_dict(orient="records"),
                         ndvi_vs_doycos_corr_heating=round(ndvi_season_corr, 3), n_boot_stability=N_BOOT)
summary["stage3"].update(lambda_max=round(float(lam_max), 4), n_excluded_cells=n_excluded, top100_classes=top100_classes,
                         criteria_spearman=crit_corr.round(3).to_dict())
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)
import importlib
vers = {}
for lib in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "shap", "libpysal", "esda", "matplotlib", "tensorflow"]:
    try:
        vers[lib] = importlib.import_module(lib).__version__
    except Exception:
        vers[lib] = "not installed"
repro = dict(python=sys.version.split()[0], platform=platform.platform(), processor=platform.processor() or platform.machine(),
             cpu_count=os.cpu_count(), seed=SEED, quick=bool(ARGS.quick), libraries=vers,
             stage_times_s=STAGE_TIMES, total_runtime_min=round((time.time() - T_START) / 60, 1),
             xgb_time_s=round(t_xgb, 1), lstm_time_s=round(t_lstm, 1))
try:
    with open("/proc/cpuinfo") as fh:
        repro["cpu_model"] = next(l.split(":")[1].strip() for l in fh if l.startswith("model name"))
    with open("/proc/meminfo") as fh:
        repro["ram_gb"] = round(int(next(l.split()[1] for l in fh if l.startswith("MemTotal"))) / 1024 ** 2, 1)
except Exception:
    pass
with open(f"{OUT}/reproducibility.json", "w") as f:
    json.dump(repro, f, indent=2)
log(f"done in {repro['total_runtime_min']} min. Outputs in ./{OUT}/")
