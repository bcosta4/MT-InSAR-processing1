"""
09_ddt_timeseries.py  (v2.5 — ocean-cell IDW applied to DDT and GDDT)

Replaces the calendar date axis with cumulative Degree Days of Thaw (DDT)
and re-expresses displacement rates as cm per DDT unit.

v2.5 COASTAL FIX (final):
  ERA5-Land stl1 has exactly two cell types in the Tuktoyaktuk grid:
    - 0    valid days : pure ocean cells  -> flagged, infilled via IDW
    - 1104 valid days : land cells (stl1 is NaN during frozen season by design)
  Previous versions used a valid_fraction threshold which incorrectly flagged
  ALL cells (land and ocean alike) because land cells also have ~50% NaN days.
  v2.5 uses valid_count == 0 as the ocean flag — unambiguous, no threshold needed.

Sign convention: negative = subsidence, positive = uplift.
Units: displacement in cm, velocity in cm per degC*day.
"""

import os
import csv
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime, timedelta

_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"] = str(_proj_data)
    os.environ["PROJ_LIB"] = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"


# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR       = Path(r"C:\bcosta\DS-InSAR\tc\shifted")
ERA5_DIR       = Path(r"C:\bcosta\DS-InSAR\era5")
ERA5_CSV       = ERA5_DIR / "tuk_t2m_daily.csv"
ERA5_T2M_GRID  = ERA5_DIR / "era5land_t2m_daily_grid.nc"
ERA5_STL1_GRID = ERA5_DIR / "era5land_stl1_daily_grid.nc"

# ── Configuration ──────────────────────────────────────────────────────────────
DATE_FMT        = "%Y%m%d"
OUTPUT_NODATA   = -9999.0
DDT_THRESHOLD   = 0.0
VELOCITY_METHOD = "linear"
INFILL_RADIUS   = 5     # search ±5 cells (~0.5°, ~45 km) — wider to guarantee
                        # land neighbours are found across the coastal zone
INFILL_MIN_CELLS = 1


# ══════════════════════════════════════════════════════════════════════════════
# ERA5-Land loader
# ══════════════════════════════════════════════════════════════════════════════
def load_era5land_grid(nc_path, var_name):
    npz_path = nc_path.with_suffix(".npz")
    if not nc_path.exists() and npz_path.exists():
        print(f"  Loading {npz_path.name} (.npz fallback)...")
        d = np.load(npz_path, allow_pickle=False)
        dates = [datetime.strptime(s, "%Y%m%d") for s in d["dates"]]
        return dates, d["data"].astype(np.float32), d["lats"], d["lons"]

    if not nc_path.exists():
        raise FileNotFoundError(nc_path)

    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray required: conda install xarray")
    import pandas as pd

    print(f"  Loading {nc_path.name}...")
    ds = xr.open_dataset(nc_path)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"

    t_var = None
    for candidate in [var_name, var_name.upper()]:
        if candidate in ds:
            t_var = candidate
            break
    if t_var is None:
        raise KeyError(f"'{var_name}' not in {nc_path.name}. Available: {list(ds.data_vars)}")

    lats  = ds[lat_name].values.astype(np.float64)
    lons  = ds[lon_name].values.astype(np.float64)
    lons  = np.where(lons > 180, lons - 360, lons)
    dates = [pd.Timestamp(t).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
             for t in ds["time"].values]
    data  = ds[t_var].values.astype(np.float32)
    ds.close()

    print(f"  Shape: {data.shape}  lat:{lats.min():.2f}-{lats.max():.2f}  "
          f"T:{np.nanmin(data):.1f}-{np.nanmax(data):.1f} degC")
    return dates, data, lats, lons


# ══════════════════════════════════════════════════════════════════════════════
# Spatial reference helpers
# ══════════════════════════════════════════════════════════════════════════════
def resolve_ref_tif(stem):
    for ext in ("hdr", "img"):
        candidate = WORK_DIR / f"{stem}.data" / f"localIncidenceAngle.{ext}"
        if candidate.exists():
            print(f"  [ref TIF] Using SNAP native: {candidate.relative_to(WORK_DIR)}")
            return candidate
    tc = WORK_DIR / "localIncidenceAngle_TC.tif"
    if tc.exists():
        print(f"  [ref TIF] WARNING: SNAP native not found. Falling back to {tc.name}.")
        return tc
    print(f"  [ref TIF] WARNING: No ref TIF found for {stem}.")
    return None


def pixel_coords_to_latlon(pixel_coords, ref_tif_path):
    from rasterio.transform import xy as rio_xy
    print(f"  [lat/lon] Reading transform from: {ref_tif_path.name}")
    with rasterio.open(ref_tif_path) as src:
        transform = src.transform
        src_crs   = src.crs

    rows = pixel_coords[:, 0].astype(int)
    cols = pixel_coords[:, 1].astype(int)
    xs, ys = rio_xy(transform, rows, cols)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    try:
        if src_crs and not src_crs.is_geographic:
            from pyproj import Transformer
            tr = Transformer.from_crs(src_crs.to_epsg() or src_crs.to_wkt(),
                                      "EPSG:4326", always_xy=True)
            lons_px, lats_px = tr.transform(xs, ys)
        else:
            lons_px, lats_px = xs, ys
    except Exception as e:
        print(f"  [WARN] CRS reprojection failed ({e}). Trying EPSG:3155.")
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:3155", "EPSG:4326", always_xy=True)
            lons_px, lats_px = tr.transform(xs, ys)
        except Exception:
            lons_px = (xs - 500000) / 111320 * (1 / np.cos(np.radians(69.44))) + (-135.0)
            lats_px = ys / 110540

    print(f"  [lat/lon] DS pixel range: lat {lats_px.min():.4f}-{lats_px.max():.4f}  "
          f"lon {lons_px.min():.4f}-{lons_px.max():.4f}")
    return lats_px.astype(np.float64), lons_px.astype(np.float64)


def find_nearest_era5_cells(lats_px, lons_px, lats_grid, lons_grid):
    i_lats = np.argmin(np.abs(lats_grid[:, None] - lats_px[None, :]), axis=0)
    i_lons = np.argmin(np.abs(lons_grid[:, None] - lons_px[None, :]), axis=0)
    return i_lats.astype(np.int32), i_lons.astype(np.int32)


def build_date_to_idx(dates):
    return {d.replace(hour=0, minute=0, second=0, microsecond=0): i
            for i, d in enumerate(dates)}


# ══════════════════════════════════════════════════════════════════════════════
# v2.5 — ocean mask from valid_count == 0
# ══════════════════════════════════════════════════════════════════════════════
def build_ocean_mask(raw_data_3d):
    """
    Flag cells with zero finite stl1 days as ocean.
    Land cells have ~1104 valid days (stl1 is NaN during frozen season by design).
    Ocean cells have 0 valid days (ERA5-Land land-sea mask excludes them fully).

    Returns
    -------
    ocean_mask : (n_lat, n_lon) bool  — True = ocean/invalid, False = land/valid
    valid_count : (n_lat, n_lon) int
    """
    valid_count = np.sum(np.isfinite(raw_data_3d), axis=0)
    ocean_mask  = (valid_count == 0)

    unique, counts = np.unique(valid_count, return_counts=True)
    print(f"  [ocean mask] valid_count distribution:")
    for u, c in zip(unique, counts):
        tag = " <- OCEAN (will be infilled)" if u == 0 else " <- land (stl1 seasonal)"
        print(f"               {u:5d} valid days : {c:4d} cells{tag}")
    print(f"  [ocean mask] Ocean cells total : {int(np.sum(ocean_mask)):,}")
    print(f"  [ocean mask] Land  cells total : {int(np.sum(~ocean_mask)):,}")
    return ocean_mask, valid_count


def infill_ocean_cells_idw(ddt_at_acq, i_lats_used, i_lons_used, ocean_mask,
                           radius=INFILL_RADIUS, min_cells=INFILL_MIN_CELLS):
    """
    For each ERA5 cell used by DS pixels where ocean_mask=True,
    replace its accumulated DDT time series with IDW mean of valid land neighbours.
    """
    n_acq, n_lat, n_lon = ddt_at_acq.shape
    unique_cells = np.unique(np.column_stack([i_lats_used, i_lons_used]), axis=0)

    flagged = [(il, ic) for (il, ic) in unique_cells if ocean_mask[il, ic]]
    if not flagged:
        print(f"  [coastal infill] No ocean ERA5 cells used by DS pixels — no infill needed.")
        return 0, []

    print(f"  [coastal infill] {len(flagged)} ocean ERA5 cell(s) used by DS pixels. Infilling...")
    infilled = []

    for (il, ic) in flagged:
        neighbours  = []
        dist_sq     = []
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = il + di, ic + dj
                if ni < 0 or ni >= n_lat or nj < 0 or nj >= n_lon:
                    continue
                if ocean_mask[ni, nj]:          # skip other ocean cells
                    continue
                nb_col = ddt_at_acq[:, ni, nj]
                if np.any(np.isfinite(nb_col)):
                    neighbours.append((ni, nj))
                    dist_sq.append(float(di**2 + dj**2))

        if len(neighbours) < min_cells:
            print(f"  [coastal infill] Cell ({il},{ic}): no valid land neighbours within "
                  f"{radius} cells — pixel will be dropped.")
            continue

        weights  = np.array([1.0 / max(d, 0.25) for d in dist_sq], dtype=np.float64)
        weights /= weights.sum()

        infilled_col = np.zeros(n_acq, dtype=np.float64)
        for w, (ni, nj) in zip(weights, neighbours):
            nb_col = ddt_at_acq[:, ni, nj].astype(np.float64)
            nb_col = np.where(np.isfinite(nb_col), nb_col, 0.0)
            infilled_col += w * nb_col

        ddt_at_acq[:, il, ic] = infilled_col.astype(np.float32)
        infilled.append((il, ic))

    print(f"  [coastal infill] Done: {len(infilled)} / {len(flagged)} ocean cell(s) infilled via IDW.")
    return len(infilled), infilled


# ══════════════════════════════════════════════════════════════════════════════
# DDT accumulation
# ══════════════════════════════════════════════════════════════════════════════
def compute_spatial_ddt(data_3d, lats_grid, lons_grid, era5_dates,
                        acquisition_dates, lats_px, lons_px,
                        threshold=DDT_THRESHOLD,
                        apply_coastal_infill=False,
                        ocean_mask=None):
    n_pixels = len(lats_px)
    n_acq    = len(acquisition_dates)

    i_lats, i_lons = find_nearest_era5_cells(lats_px, lons_px, lats_grid, lons_grid)
    unique_cells = len(np.unique(np.column_stack([i_lats, i_lons]), axis=0))
    print(f"  DS pixels: {n_pixels:,} -> {unique_cells} unique ERA5-Land cells")

    era5_date_idx = build_date_to_idx(era5_dates)
    acq_start     = datetime(acquisition_dates[0].year, 1, 1)
    acq_end       = acquisition_dates[-1]
    n_lat         = len(lats_grid)
    n_lon         = len(lons_grid)

    ddt_running  = np.zeros((n_lat, n_lon), dtype=np.float64)
    acq_date_set = {d.replace(hour=0, minute=0, second=0, microsecond=0): i
                    for i, d in enumerate(acquisition_dates)}
    ddt_at_acq   = np.full((n_acq, n_lat, n_lon), np.nan, dtype=np.float32)

    current = acq_start
    while current <= acq_end:
        norm     = current.replace(hour=0, minute=0, second=0, microsecond=0)
        era5_idx = era5_date_idx.get(norm)
        if era5_idx is not None:
            day_slice = data_3d[era5_idx].astype(np.float64)
            pos_mask  = (day_slice > threshold) & np.isfinite(day_slice)
            ddt_running[pos_mask] += (day_slice[pos_mask] - threshold)
        if norm in acq_date_set:
            ddt_at_acq[acq_date_set[norm]] = ddt_running.astype(np.float32)
        current += timedelta(days=1)

    n_infilled = 0
    if apply_coastal_infill:
        if ocean_mask is None:
            raise ValueError("ocean_mask required when apply_coastal_infill=True")
        n_infilled, _ = infill_ocean_cells_idw(ddt_at_acq, i_lats, i_lons, ocean_mask)

    ddt_pixel_acq = ddt_at_acq[:, i_lats, i_lons]
    ddt_mean_acq  = np.nanmean(ddt_at_acq, axis=(1, 2)).astype(np.float32)

    n_nan = int(np.sum(~np.isfinite(ddt_pixel_acq)))
    if n_nan > 0:
        print(f"  [WARN] {n_nan} NaN DDT entries remain after infill.")

    return ddt_pixel_acq.astype(np.float32), ddt_mean_acq, i_lats, i_lons, n_infilled


# ══════════════════════════════════════════════════════════════════════════════
# Velocity fitting
# ══════════════════════════════════════════════════════════════════════════════
def compute_spatial_ddt_velocity(timeseries_cm, ddt_pixel_acq, method="linear"):
    n_dates, n_pixels = timeseries_cm.shape
    velocity_ddt = np.full(n_pixels, np.nan, dtype=np.float32)
    for px in range(n_pixels):
        ts_px  = timeseries_cm[:, px].astype(np.float64)
        ddt_px = ddt_pixel_acq[:, px].astype(np.float64)
        valid  = np.isfinite(ts_px) & np.isfinite(ddt_px)
        if np.sum(valid) < 3 or np.std(ddt_px[valid]) < 1e-6:
            continue
        G = np.column_stack([np.ones(valid.sum()), ddt_px[valid]])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(G, ts_px[valid], rcond=None)
            velocity_ddt[px] = float(coeffs[1])
        except np.linalg.LinAlgError:
            continue
    return velocity_ddt


def compute_uniform_ddt_velocity(timeseries_cm, ddt_values, method="linear"):
    n_dates, n_pixels = timeseries_cm.shape
    velocity_ddt = np.full(n_pixels, np.nan, dtype=np.float32)
    valid = np.isfinite(ddt_values)
    if np.sum(valid) < 3:
        return velocity_ddt
    G      = np.column_stack([np.ones(valid.sum()),
                               ddt_values[valid].astype(np.float64)])
    coeffs = np.linalg.solve(G.T @ G,
                              G.T @ timeseries_cm[valid].astype(np.float64))
    return coeffs[1, :].astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CSV fallback
# ══════════════════════════════════════════════════════════════════════════════
def load_era5_temperature_csv(csv_path):
    temp_dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                temp_dict[datetime.strptime(row["date"].strip(), DATE_FMT)] = \
                    float(row["t2m_mean_c"])
            except (ValueError, KeyError):
                continue
    return temp_dict


def compute_uniform_ddt(acquisition_dates, temp_dict, threshold=DDT_THRESHOLD):
    start      = datetime(acquisition_dates[0].year, 1, 1)
    ddt_series = {}
    cumulative = 0.0
    current    = start
    while current <= acquisition_dates[-1]:
        t = temp_dict.get(current)
        if t is not None and t > threshold:
            cumulative += (t - threshold)
        ddt_series[current] = cumulative
        current += timedelta(days=1)
    return np.array([ddt_series.get(d, np.nan) for d in acquisition_dates],
                    dtype=np.float32)


def synthetic_ddt_fallback(acquisition_dates, threshold=DDT_THRESHOLD):
    print("  WARNING: No ERA5 data — using SYNTHETIC DDT approximation.")
    start      = datetime(acquisition_dates[0].year, 1, 1)
    ddt_series = {}
    cumulative = 0.0
    current    = start
    while current <= acquisition_dates[-1]:
        doy = current.timetuple().tm_yday
        t   = -25.0 + 30.0 * np.sin((doy - 60) / 365.0 * 2 * np.pi)
        if t > threshold:
            cumulative += (t - threshold)
        ddt_series[current] = cumulative
        current += timedelta(days=1)
    return (np.array([ddt_series.get(d, np.nan) for d in acquisition_dates],
                     dtype=np.float32), True)


# ══════════════════════════════════════════════════════════════════════════════
# GeoTIFF writer — tiled for robustness with large rasters
# ══════════════════════════════════════════════════════════════════════════════
def write_geotiff_from_vector(values_1d, pixel_coords, ref_tif_path,
                              out_path, nodata, description=""):
    print(f"  [write_tif] {out_path.name}")
    with rasterio.open(ref_tif_path) as src:
        nrows     = src.height
        ncols     = src.width
        transform = src.transform
        crs       = src.crs

    out_2d = np.full((nrows, ncols), nodata, dtype=np.float32)
    out_2d[pixel_coords[:, 0], pixel_coords[:, 1]] = values_1d

    with rasterio.open(
        out_path, "w",
        driver    = "GTiff",
        height    = nrows,
        width     = ncols,
        count     = 1,
        dtype     = np.float32,
        crs       = crs,
        transform = transform,
        nodata    = nodata,
        compress  = "deflate",
        tiled     = True,
        blockxsize = 256,
        blockysize = 256,
    ) as dst:
        dst.write(out_2d, 1)
        if description:
            dst.update_tags(1, description=description)


# ══════════════════════════════════════════════════════════════════════════════
# Per-network processing
# ══════════════════════════════════════════════════════════════════════════════
def process_network(stem, spatial_grids, temp_dict_csv, synthetic):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    ts_path     = WORK_DIR / f"{stem}_timeseries_referenced.npy"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    dates_path  = WORK_DIR / f"{stem}_date_list.csv"

    for p in [ts_path, coords_path, dates_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} — run 06_reference_point.py first.")
            return

    ref_tif      = resolve_ref_tif(stem)
    timeseries   = np.load(ts_path).astype(np.float32) / 10.0
    pixel_coords = np.load(coords_path)

    acquisition_dates = []
    days_list         = []
    with open(dates_path, newline="") as f:
        for row in csv.DictReader(f):
            acquisition_dates.append(datetime.strptime(row["date"], DATE_FMT))
            days_list.append(int(row["days_since_first"]))

    n_dates, n_pixels = timeseries.shape
    print(f"  Time series : {timeseries.shape} [cm]")
    print(f"  Dates       : {n_dates}  "
          f"({acquisition_dates[0]:%Y-%m-%d} -> {acquisition_dates[-1]:%Y-%m-%d})")

    use_spatial      = spatial_grids is not None and len(spatial_grids) > 0
    has_stl1         = use_spatial and "stl1" in spatial_grids
    n_infilled_ddt   = 0
    n_infilled_gddt  = 0

    if synthetic:
        ddt_uniform, _   = synthetic_ddt_fallback(acquisition_dates)
        ddt_pixel_t2m    = ddt_uniform[:, None] * np.ones(n_pixels, dtype=np.float32)
        ddt_mean_t2m     = ddt_uniform
        ddt_pixel_stl1   = None
        ddt_mean_stl1    = None

    elif not use_spatial:
        ddt_uniform      = compute_uniform_ddt(acquisition_dates, temp_dict_csv)
        ddt_pixel_t2m    = ddt_uniform[:, None] * np.ones(n_pixels, dtype=np.float32)
        ddt_mean_t2m     = ddt_uniform
        ddt_pixel_stl1   = None
        ddt_mean_stl1    = None

    else:
        if ref_tif is None:
            print("  ERROR: no ref TIF — cannot compute spatial DDT.")
            return

        lats_px, lons_px = pixel_coords_to_latlon(pixel_coords, ref_tif)

        # Build common ocean mask from stl1 valid_count == 0, then apply same coastal
        # IDW infill to BOTH DDT (t2m) and GDDT (stl1) for spatial consistency.
        stl1_dates, stl1_data, stl1_lats, stl1_lons = spatial_grids["stl1"]
        ocean_mask, _ = build_ocean_mask(stl1_data)

        print("  Computing per-pixel DDT from ERA5-Land t2m (air temp) + ocean-cell IDW...")
        t2m_dates, t2m_data, t2m_lats, t2m_lons = spatial_grids["t2m"]
        ddt_pixel_t2m, ddt_mean_t2m, _, _, n_infilled_ddt = compute_spatial_ddt(
            t2m_data, t2m_lats, t2m_lons, t2m_dates,
            acquisition_dates, lats_px, lons_px,
            apply_coastal_infill=True, ocean_mask=ocean_mask)

        if has_stl1:
            print("  Computing per-pixel GDDT from ERA5-Land stl1 (soil 0-7 cm)...")
            ddt_pixel_stl1, ddt_mean_stl1, _, _, n_infilled_gddt = compute_spatial_ddt(
                stl1_data, stl1_lats, stl1_lons, stl1_dates,
                acquisition_dates, lats_px, lons_px,
                apply_coastal_infill=True, ocean_mask=ocean_mask)
        else:
            ddt_pixel_stl1 = None
            ddt_mean_stl1  = None
            n_infilled_gddt = 0

    ddt_range = float(np.nanmax(ddt_mean_t2m) - np.nanmin(ddt_mean_t2m))
    span_yrs  = (acquisition_dates[-1] - acquisition_dates[0]).days / 365.25
    print(f"  DDT range : {ddt_range:.1f} degC*day | "
          f"mean annual: {ddt_range / max(span_yrs, 0.1):.1f} degC*day/yr")

    print(f"\n  Fitting displacement vs DDT (air, {VELOCITY_METHOD})...")
    velocity_ddt = (compute_spatial_ddt_velocity(timeseries, ddt_pixel_t2m, VELOCITY_METHOD)
                    if use_spatial else
                    compute_uniform_ddt_velocity(timeseries, ddt_mean_t2m, VELOCITY_METHOD))
    valid_v = velocity_ddt[np.isfinite(velocity_ddt)]
    if len(valid_v) > 0:
        print(f"  alpha_DDT  : mean={np.mean(valid_v):.5f}  "
              f"std={np.std(valid_v):.5f} cm/DDT  n={len(valid_v):,}  "
              f"(+{n_infilled_ddt} ocean cells infilled via IDW v2.5)")

    velocity_gddt = None
    if ddt_pixel_stl1 is not None:
        print(f"  Fitting displacement vs GDDT (soil, {VELOCITY_METHOD})...")
        velocity_gddt = (compute_spatial_ddt_velocity(timeseries, ddt_pixel_stl1, VELOCITY_METHOD)
                         if use_spatial else
                         compute_uniform_ddt_velocity(timeseries, ddt_mean_stl1, VELOCITY_METHOD))
        valid_g = velocity_gddt[np.isfinite(velocity_gddt)]
        if len(valid_g) > 0:
            print(f"  alpha_GDDT : mean={np.mean(valid_g):.5f}  "
                  f"std={np.std(valid_g):.5f} cm/GDDT  n={len(valid_g):,}  "
                  f"(+{n_infilled_gddt} ocean cells infilled via IDW v2.5)")

    source_str = ("spatial_ERA5-Land" if use_spatial
                  else ("point_CSV" if temp_dict_csv else "SYNTHETIC"))
    infill_str = (f"ocean_IDW_v2.5:{n_infilled_gddt}_cells"
                  if n_infilled_gddt > 0 else "no_infill_needed")

    out_ddt_csv  = WORK_DIR / f"{stem}_ddt_axis.csv"
    out_ts_ddt   = WORK_DIR / f"{stem}_timeseries_ddt.npy"
    out_vel_ddt  = WORK_DIR / f"{stem}_velocity_ddt.npy"
    out_vel_gddt = WORK_DIR / f"{stem}_velocity_gddt.npy"
    out_tif_ddt  = WORK_DIR / f"{stem}_velocity_ddt_map.tif"
    out_tif_gddt = WORK_DIR / f"{stem}_velocity_gddt_map.tif"
    out_tif_diff = WORK_DIR / f"{stem}_velocity_ddt_gddt_diff.tif"
    out_rep      = WORK_DIR / f"{stem}_ddt_report.txt"

    with open(out_ddt_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["index", "date", "days_since_first", "ddt_t2m_mean", "ddt_source"]
        if ddt_mean_stl1 is not None:
            header.insert(4, "gddt_stl1_mean")
        writer.writerow(header)
        for i, d in enumerate(acquisition_dates):
            row = [i, d.strftime(DATE_FMT), days_list[i],
                   round(float(ddt_mean_t2m[i]), 3)]
            if ddt_mean_stl1 is not None:
                row.insert(3, round(float(ddt_mean_stl1[i]), 3))
            row.append(source_str)
            writer.writerow(row)

    np.save(out_ts_ddt, timeseries)
    np.save(out_vel_ddt, velocity_ddt)
    if velocity_gddt is not None:
        np.save(out_vel_gddt, velocity_gddt)

    if ref_tif is not None and ref_tif.exists():
        write_geotiff_from_vector(
            velocity_ddt, pixel_coords, ref_tif, out_tif_ddt, OUTPUT_NODATA,
            f"cm/DDT (air t2m). Negative=subsidence. {source_str}.")

        if velocity_gddt is not None:
            write_geotiff_from_vector(
                velocity_gddt, pixel_coords, ref_tif, out_tif_gddt, OUTPUT_NODATA,
                f"cm/GDDT (soil stl1, {infill_str}). Negative=subsidence.")

            diff_vel = velocity_ddt - velocity_gddt
            diff_vel[~np.isfinite(diff_vel)] = OUTPUT_NODATA
            write_geotiff_from_vector(
                diff_vel, pixel_coords, ref_tif, out_tif_diff, OUTPUT_NODATA,
                "alpha_DDT - alpha_GDDT. Positive=more subsidence per air DDT "
                "than per soil DDT (snow insulation). Negative=soil warms faster than air.")
    else:
        print("  Skipping GeoTIFFs — ref TIF not available.")

    valid_v  = velocity_ddt[np.isfinite(velocity_ddt)]
    ts_now   = datetime.now().isoformat(timespec="seconds")
    report   = [
        f"DDT report       : {stem}\n",
        f"Timestamp        : {ts_now}\n",
        f"Script version   : v2.5 (ocean mask from valid_count == 0)\n",
        f"WORK_DIR         : {WORK_DIR}\n",
        f"Ref TIF          : {ref_tif.name if ref_tif else 'NOT FOUND'}\n",
        f"DDT source       : {source_str}\n",
        f"DDT threshold    : {DDT_THRESHOLD} degC\n",
        f"Velocity method  : {VELOCITY_METHOD}\n",
        f"GDDT infill v2.5 : {infill_str}\n",
        f"Total DDT range  : {ddt_range:.2f} degC*day (air, spatial mean)\n",
        (f"alpha_DDT mean   : {np.mean(valid_v):.6f} cm/degC*day  "
         f"(n={len(valid_v):,})\n") if len(valid_v) else "",
    ]
    if velocity_gddt is not None:
        valid_g = velocity_gddt[np.isfinite(velocity_gddt)]
        if len(valid_g) > 0:
            report += [
                f"alpha_GDDT mean  : {np.mean(valid_g):.6f} cm/degC*day  (n={len(valid_g):,})\n",
                f"alpha_GDDT std   : {np.std(valid_g):.6f} cm/degC*day\n",
            ]
    report.append("\nDDT values at acquisition dates (spatial mean):\n")
    for i, d in enumerate(acquisition_dates):
        gddt_i = float(ddt_mean_stl1[i]) if ddt_mean_stl1 is not None else np.nan
        report.append(f"  [{i:02d}] {d:%Y-%m-%d}  "
                      f"DDT={float(ddt_mean_t2m[i]):.2f}  "
                      f"GDDT={gddt_i:.2f}  degC*day\n")
    Path(out_rep).write_text("".join(report), encoding="utf-8")
    print(f"  Written: {out_ddt_csv.name}  {out_ts_ddt.name}  "
          f"{out_vel_ddt.name}  {out_rep.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ts_files = sorted(WORK_DIR.glob("*_timeseries_referenced.npy"))
    if not ts_files:
        print(f"No *_timeseries_referenced.npy in {WORK_DIR}.")
        return

    stems = [f.stem.replace("_timeseries_referenced", "") for f in ts_files]
    print("09_ddt_timeseries.py  (v2.5 — ocean-cell IDW applied to DDT and GDDT)")
    print(f"WORK_DIR : {WORK_DIR}")
    print(f"Found {len(stems)} network(s): {stems}")

    spatial_grids = {}
    synthetic     = False
    temp_dict_csv = None

    for var_name, nc_path, label in [
        ("t2m",  ERA5_T2M_GRID,  "air temperature 2m"),
        ("stl1", ERA5_STL1_GRID, "soil temperature layer 1"),
    ]:
        try:
            era5_dates, data_3d, lats, lons = load_era5land_grid(nc_path, var_name)
            spatial_grids[var_name] = (era5_dates, data_3d, lats, lons)
            print(f"  Loaded: {nc_path.name} ({label}, {len(era5_dates)} days)")
        except FileNotFoundError:
            print(f"  Not found: {nc_path.name}")
        except Exception as e:
            print(f"  ERROR loading {nc_path.name}: {e}")

    if not spatial_grids:
        if ERA5_CSV.exists():
            print(f"Falling back to single-point CSV: {ERA5_CSV.name}")
            temp_dict_csv = load_era5_temperature_csv(ERA5_CSV)
        else:
            print("No ERA5 data — using SYNTHETIC DDT.")
            synthetic = True

    for stem in stems:
        process_network(stem, spatial_grids if spatial_grids else None,
                        temp_dict_csv, synthetic)

    print("\n" + "=" * 60)
    print("DDT time series complete.")
    if synthetic:
        print("IMPORTANT: Synthetic DDT used. Download ERA5-Land for real results.")


if __name__ == "__main__":
    main()
