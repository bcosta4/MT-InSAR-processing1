"""
05b_inversion.py
Coherence-weighted SBAS least-squares inversion.
Solves A*v = delta_phi per pixel using the design matrix from 04 and the
phase/coherence matrices from 03. Produces a mean LOS velocity map and a
full displacement time series for each network independently.

Phase input auto-detection:
If _phase_matrix_atmo.npy exists (produced by 05a_atmospheric_correction.py),
it is used automatically. Otherwise falls back to _phase_matrix.npy.
Run 05a first to apply atmospheric correction before inversion.

Sign convention (applied in this script):
Input phase_matrix is in SNAP radar convention:
positive phase = increased slant range = subsidence
This script negates the time series and velocity on output so that
all downstream files follow the geophysical convention:
NEGATIVE = subsidence (moving away from satellite)
POSITIVE = uplift (moving toward satellite)

Unit conversion (applied in this script):
After inversion and sign correction, the script loads the terrain-corrected
local incidence angle raster (localIncidenceAngle_TC.tif) and converts
per-pixel LOS phase [rad] to vertical displacement [cm]:

d_vert [cm] = (phi [rad] * lambda [m]) / (4 * pi * cos(theta)) * 100

where theta is the spatially-varying local incidence angle, clipped to the
valid Sentinel-1 IW range [29, 46] degrees before use. Pixels outside this
range are set to NaN and filled from the near/far linear ramp.

Geotransform (v1.1 FIX):
export_geotiffs() now reads CRS and transform from localIncidenceAngle_TC.tif
(same master TIF used by script 02 and 07) instead of from mean_coherence.tif.
This fixes the ~95 km SE offset on _velocity_cmyr.tif and
_velocity_rad_per_day.tif that was inherited from the incorrectly located
mean_coherence.tif produced by the old script 02.

MIN_VALID_IFGS policy (v1.2 change):
MIN_VALID_IFGS is no longer a fixed global constant. It is set dynamically
inside process_network() to n_ifgs (the total number of interferograms in the
network), requiring every IFG to have valid phase and coherence at a pixel
before that pixel is accepted for inversion. This is the strictest possible
filter and ensures no pixel is inverted with a partially broken network.

v1.3 diagnostics (NEW):
  - East-West asymmetry diagnostic: splits pixel grid into west and east halves
    and reports mean/std/median velocity in each half, plus their difference.
    Also reports correlation between pixel column index and velocity (Pearson r).
    A large E-W difference (>0.3 cm/yr) or r > 0.3 suggests incidence angle
    projection effects or ionospheric gradients — check individual IFGs and
    compare with slope map.
  - Exports _incidence_angle_ds_pixels.tif: per-DS-pixel incidence angle
    remapped back onto the full image grid. Use in QGIS to overlay with
    velocity map to check whether E-W asymmetry tracks incidence angle gradient.

Outputs (per network, written to WORK_DIR):
_velocity.npy               float32 (n_valid_pixels,) [cm/year]
_timeseries.npy             float32 (n_dates, n_valid_pixels) [cm]
_residuals.npy              float32 (n_ifgs, n_valid_pixels) [rad]
_inversion_report.txt       processing parameters and statistics
_velocity_cmyr.tif          float32 GeoTIFF [cm/year] -- QGIS preview
_velocity_rad_per_day.tif   float32 GeoTIFF [rad/day] -- QC
_incidence_angle_ds_pixels.tif  float32 GeoTIFF [degrees] -- NEW: E-W diag
"""

import os
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.ndimage import zoom

_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"] = str(_proj_data)
    os.environ["PROJ_LIB"]  = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

import rasterio

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR = Path(r"C:\bcosta\DS-InSAR\tc")

_EPSG3155_WKT = (
    'PROJCS["NAD83(CSRS) / UTM zone 8N",'
    'GEOGCS["NAD83(CSRS)",'
    'DATUM["NAD83_Canadian_Spatial_Reference_System",'
    'SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],'
    'AUTHORITY["EPSG","6140"]],'
    'PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
    'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
    'AUTHORITY["EPSG","4617"]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",-135],'
    'PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",0],'
    'UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AUTHORITY["EPSG","3155"]]'
)

# ── Configuration ──────────────────────────────────────────────────────────────
MIN_COH_WEIGHT     = 0.10
CHUNK_SIZE         = 50000
WAVELENGTH_M       = 0.05546
NODATA             = -9999.0
DATE_FMT           = "%Y%m%d"
INCIDENCE_NEAR_DEG = 41.07575
INCIDENCE_FAR_DEG  = 45.52002
INC_MIN_VALID      = 29.0
INC_MAX_VALID      = 46.0
EW_DIFF_WARN_CMYR  = 0.30


# ── Resolve incidence / master TIF per stack stem ─────────────────────────────
def resolve_incidence_tif(stem):
    hdr = WORK_DIR / f"{stem}.data" / "localIncidenceAngle.hdr"
    img = WORK_DIR / f"{stem}.data" / "localIncidenceAngle.img"
    if hdr.exists() and img.exists():
        print(f"    [incidence] Using SNAP native: {img.relative_to(WORK_DIR)}")
        return img
    if hdr.exists() and not img.exists():
        print(f"    [incidence] WARNING: .hdr found but .img missing in {stem}.data/ — falling back.")
    tc = WORK_DIR / "localIncidenceAngle_TC.tif"
    if tc.exists():
        print(f"    [incidence] Falling back to {tc.name} (TC export).")
        return tc
    print(f"    [incidence] WARNING: No incidence raster found for {stem}. Using linear ramp.")
    return None


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_date_list(csv_path):
    dates, days_list = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            dates.append(datetime.strptime(row["date"], DATE_FMT))
            days_list.append(int(row["days_since_first"]))
    return dates, np.array(days_list, dtype=np.float32)


def load_incidence_angle(coh_tif_path, incidence_tif_path):
    with rasterio.open(coh_tif_path) as coh_src:
        coh_nrows = coh_src.height
        coh_ncols = coh_src.width

    if incidence_tif_path is None:
        print(f"   [incidence] Linear ramp fallback ({INCIDENCE_NEAR_DEG:.3f}-{INCIDENCE_FAR_DEG:.3f} deg)")
        col_idx = np.arange(coh_ncols, dtype=np.float32)
        ramp = (INCIDENCE_NEAR_DEG + (INCIDENCE_FAR_DEG - INCIDENCE_NEAR_DEG) *
                col_idx / max(coh_ncols - 1, 1))
        return np.tile(ramp, (coh_nrows, 1)).astype(np.float32)

    with rasterio.open(incidence_tif_path) as inc_src:
        inc_arr   = inc_src.read(1).astype(np.float32)
        inc_nrows = inc_src.height
        inc_ncols = inc_src.width

    print(f"   [incidence] Opened: {incidence_tif_path.name} "
          f"shape=({inc_nrows},{inc_ncols}) "
          f"range {np.nanmin(inc_arr):.3f}-{np.nanmax(inc_arr):.3f} deg")

    inc_arr[inc_arr <= 0] = np.nan
    inc_arr[(inc_arr < INC_MIN_VALID) | (inc_arr > INC_MAX_VALID)] = np.nan

    if inc_nrows == coh_nrows and inc_ncols == coh_ncols:
        print("   [incidence] Exact grid match — no resampling needed")
        return inc_arr

    zoom_r = coh_nrows / inc_nrows
    zoom_c = coh_ncols / inc_ncols
    print(f"   [incidence] Resampling ({inc_nrows},{inc_ncols}) -> ({coh_nrows},{coh_ncols})")
    inc_mean   = float(np.nanmean(inc_arr))
    nan_mask   = np.isnan(inc_arr)
    inc_filled = inc_arr.copy()
    inc_filled[nan_mask] = inc_mean
    inc_zoomed = zoom(inc_filled, zoom=(zoom_r, zoom_c), order=1).astype(np.float32)
    nan_zoomed = zoom(nan_mask.astype(np.float32), zoom=(zoom_r, zoom_c), order=1)
    inc_zoomed[nan_zoomed > 0.5] = np.nan
    return inc_zoomed


def phase_to_displacement_cm(phase_data, wavelength_m, incidence_deg):
    inc_rad  = np.radians(incidence_deg)
    d_los_m  = (phase_data * wavelength_m) / (4.0 * np.pi)
    d_vert_m = d_los_m / np.cos(inc_rad)
    return (d_vert_m * 100.0).astype(np.float32)


def weighted_lstsq_chunk(A, phase_chunk, coh_chunk, min_coh, min_valid):
    n_ifgs, n_pixels = phase_chunk.shape
    n_intervals      = A.shape[1]
    v_chunk          = np.full((n_intervals, n_pixels), np.nan, dtype=np.float32)
    res_chunk        = np.full((n_ifgs, n_pixels),      np.nan, dtype=np.float32)
    valid_mask       = np.zeros(n_pixels, dtype=bool)
    W_all            = np.clip(coh_chunk, min_coh, 1.0).astype(np.float32)
    for px in range(n_pixels):
        phi   = phase_chunk[:, px]
        w     = W_all[:, px]
        valid = np.isfinite(phi) & np.isfinite(w)
        if np.sum(valid) < min_valid:
            continue
        phi_v = phi[valid]; w_v = w[valid]; A_v = A[valid, :]
        W_mat = np.diag(w_v)
        AtWA  = A_v.T @ W_mat @ A_v + 1e-6 * np.eye(n_intervals)
        AtWphi = A_v.T @ W_mat @ phi_v
        try:
            v, _, _, _ = np.linalg.lstsq(AtWA, AtWphi, rcond=None)
        except np.linalg.LinAlgError:
            continue
        v_chunk[:, px]   = v.astype(np.float32)
        valid_mask[px]   = True
        res_all          = np.full(n_ifgs, np.nan, dtype=np.float32)
        res_all[valid]   = (A_v @ v - phi_v).astype(np.float32)
        res_chunk[:, px] = res_all
    return v_chunk, res_chunk, valid_mask


def intervals_to_timeseries(v_matrix, days_array):
    n_dates  = len(days_array)
    n_pixels = v_matrix.shape[1]
    ts       = np.zeros((n_dates, n_pixels), dtype=np.float32)
    for i in range(1, n_dates):
        ts[i, :] = ts[i - 1, :] + v_matrix[i - 1, :]
    return -ts  # sign flip: negative = subsidence


def compute_mean_velocity(ts, days_array):
    t      = days_array.astype(np.float64)
    G      = np.column_stack([np.ones(len(t)), t]).astype(np.float64)
    coeffs = np.linalg.solve(G.T @ G, G.T @ ts.astype(np.float64))
    return coeffs[1, :].astype(np.float32)


def make_2d(values_1d, pixel_coords, nrows, ncols, nodata=NODATA):
    grid = np.full((nrows, ncols), nodata, dtype=np.float32)
    grid[pixel_coords[:, 0], pixel_coords[:, 1]] = values_1d
    return grid


def write_tif(array_2d, out_path, meta, nodata=NODATA):
    out_meta = meta.copy()
    out_meta.update(dtype="float32", count=1, nodata=nodata)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(array_2d.astype(np.float32), 1)


# ── E-W Asymmetry Diagnostic ──────────────────────────────────────────────────
def ew_asymmetry_diagnostic(velocity_cm_yr, pixel_coords, ncols, incidence_px):
    valid_mask = np.isfinite(velocity_cm_yr)
    col_idx    = pixel_coords[:, 1].astype(np.float32)
    half_col   = ncols // 2
    west_mask  = valid_mask & (col_idx < half_col)
    east_mask  = valid_mask & (col_idx >= half_col)
    vel_west   = velocity_cm_yr[west_mask]
    vel_east   = velocity_cm_yr[east_mask]
    mean_w = float(np.nanmean(vel_west))   if len(vel_west) > 0 else np.nan
    mean_e = float(np.nanmean(vel_east))   if len(vel_east) > 0 else np.nan
    std_w  = float(np.nanstd(vel_west))    if len(vel_west) > 0 else np.nan
    std_e  = float(np.nanstd(vel_east))    if len(vel_east) > 0 else np.nan
    med_w  = float(np.nanmedian(vel_west)) if len(vel_west) > 0 else np.nan
    med_e  = float(np.nanmedian(vel_east)) if len(vel_east) > 0 else np.nan
    ew_diff = mean_e - mean_w
    r_col = np.nan
    if np.sum(valid_mask) > 2:
        col_v = col_idx[valid_mask]; vel_v = velocity_cm_yr[valid_mask]
        if np.std(col_v) > 0 and np.std(vel_v) > 0:
            r_col = float(np.corrcoef(col_v, vel_v)[0, 1])
    r_inc = np.nan
    if incidence_px is not None:
        valid_inc = valid_mask & np.isfinite(incidence_px)
        if np.sum(valid_inc) > 2:
            inc_v = incidence_px[valid_inc]; vel_vi = velocity_cm_yr[valid_inc]
            if np.std(inc_v) > 0 and np.std(vel_vi) > 0:
                r_inc = float(np.corrcoef(inc_v, vel_vi)[0, 1])
    print("\n  [E-W ASYMMETRY DIAGNOSTIC]")
    print(f"   West (col<{half_col}): mean={mean_w:+.3f} std={std_w:.3f} med={med_w:+.3f} cm/yr")
    print(f"   East (col>={half_col}): mean={mean_e:+.3f} std={std_e:.3f} med={med_e:+.3f} cm/yr")
    print(f"   E-W diff: {ew_diff:+.3f} cm/yr | r(col,vel)={r_col:.4f} | r(inc,vel)={r_inc:.4f}")
    if abs(ew_diff) > EW_DIFF_WARN_CMYR:
        print(f"   *** WARNING |E-W diff| > {EW_DIFF_WARN_CMYR} cm/yr ***")
        print(f"   Check: LOS projection on slopes / ionospheric gradient / residual atmosphere")
    else:
        print(f"   E-W difference below {EW_DIFF_WARN_CMYR} cm/yr threshold — OK")
    return {"mean_west": mean_w, "mean_east": mean_e, "std_west": std_w, "std_east": std_e,
            "median_west": med_w, "median_east": med_e, "ew_diff": ew_diff,
            "r_col_vel": r_col, "r_inc_vel": r_inc}


# ── GeoTIFF export ─────────────────────────────────────────────────────────────
def export_geotiffs(stem, velocity_cm_yr, velocity_rad,
                    incidence_px, pixel_coords, coh_tif_path,
                    master_tif_path,
                    phase_source, inc_min, inc_max, inc_mean_val,
                    n_nan_filled, min_valid_ifgs, ew_diag):
    from rasterio.crs import CRS
    if master_tif_path is None or not master_tif_path.exists():
        print(f"   [GeoTIFF] WARN: master TIF not found — export skipped.")
        return
    print(f"   [GeoTIFF] Spatial ref from: {master_tif_path.name}")
    with rasterio.open(master_tif_path) as master_src:
        master_meta  = master_src.meta.copy()
        master_nrows = master_src.height
        master_ncols = master_src.width
        try:
            wkt = master_src.crs.to_wkt()
        except Exception:
            wkt = str(master_src.crs)
        if wkt.startswith("LOCAL_CS"):
            master_meta["crs"] = CRS.from_wkt(_EPSG3155_WKT)
            print(f"   [GeoTIFF] LOCAL_CS detected — overriding with EPSG:3155")
    nrows = master_nrows
    ncols = master_ncols
    vel_2d_cmyr    = make_2d(np.where(np.isfinite(velocity_cm_yr), velocity_cm_yr, NODATA),
                             pixel_coords, nrows, ncols)
    vel_2d_rad_day = make_2d(np.where(np.isfinite(velocity_rad), velocity_rad, NODATA),
                             pixel_coords, nrows, ncols)
    out_cmyr    = WORK_DIR / f"{stem}_velocity_cmyr.tif"
    out_rad_day = WORK_DIR / f"{stem}_velocity_rad_per_day.tif"
    write_tif(vel_2d_cmyr,    out_cmyr,    master_meta)
    write_tif(vel_2d_rad_day, out_rad_day, master_meta)
    out_inc_tif = WORK_DIR / f"{stem}_incidence_angle_ds_pixels.tif"
    inc_2d = make_2d(np.where(np.isfinite(incidence_px), incidence_px, NODATA),
                     pixel_coords, nrows, ncols)
    write_tif(inc_2d, out_inc_tif, master_meta)
    valid   = np.isfinite(velocity_cm_yr)
    n_total = int(np.sum(valid))
    n_sub   = int(np.sum(velocity_cm_yr[valid] < 0))
    n_upl   = int(np.sum(velocity_cm_yr[valid] > 0))
    print(f"   velocity range: {velocity_cm_yr[valid].min():.3f} – {velocity_cm_yr[valid].max():.3f} cm/yr")
    print(f"   Subsiding: {n_sub:,} ({n_sub/max(n_total,1)*100:.1f}%) | "
          f"Uplifting: {n_upl:,} ({n_upl/max(n_total,1)*100:.1f}%)")
    print(f"   Written: {out_cmyr.name}")
    print(f"   Written: {out_rad_day.name}")
    print(f"   Written: {out_inc_tif.name}")


# ── Per-network processing ─────────────────────────────────────────────────────
def process_network(stem):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    phase_path_atmo = WORK_DIR / f"{stem}_phase_matrix_atmo.npy"
    phase_path_raw  = WORK_DIR / f"{stem}_phase_matrix.npy"
    if phase_path_atmo.exists():
        phase_path   = phase_path_atmo
        phase_source = "atmospherically corrected (05a)"
    else:
        phase_path   = phase_path_raw
        phase_source = "raw (05a not run)"
    print(f"  Phase input  : {phase_path.name} [{phase_source}]")

    coh_path    = WORK_DIR / f"{stem}_coh_matrix.npy"
    design_path = WORK_DIR / f"{stem}_design_matrix.npy"
    dates_path  = WORK_DIR / f"{stem}_date_list.csv"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    coh_tif     = WORK_DIR / f"{stem}_mean_coherence.tif"

    for p in [phase_path, coh_path, design_path, dates_path, coords_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} — run scripts 03 and 04 first.")
            return

    # v1.4 FIX: resolve per-network, not from a global constant
    incidence_tif = resolve_incidence_tif(stem)
    master_tif    = incidence_tif

    phase_matrix = np.load(phase_path)
    coh_matrix   = np.load(coh_path)
    A            = np.load(design_path)
    pixel_coords = np.load(coords_path)
    dates, days  = load_date_list(dates_path)

    n_ifgs, n_pixels = phase_matrix.shape
    n_dates          = len(dates)
    min_valid_ifgs   = n_ifgs

    print(f"  Phase matrix   : {phase_matrix.shape}")
    print(f"  Design matrix  : {A.shape}")
    print(f"  Min valid IFGs : {min_valid_ifgs} (= n_ifgs)")
    print(f"  Dates          : {n_dates} ({dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d})")
    print(f"  Processing {n_pixels:,} pixels in chunks of {CHUNK_SIZE:,}...")

    with rasterio.open(coh_tif) as src:
        ncols_coh = src.width

    incidence_2d = load_incidence_angle(coh_tif, incidence_tif)
    incidence_px = incidence_2d[pixel_coords[:, 0], pixel_coords[:, 1]]

    nan_mask_inc = np.isnan(incidence_px)
    n_nan_inc    = int(np.sum(nan_mask_inc))
    if n_nan_inc > 0:
        col_indices = pixel_coords[nan_mask_inc, 1].astype(np.float32)
        fill_vals   = (INCIDENCE_NEAR_DEG + (INCIDENCE_FAR_DEG - INCIDENCE_NEAR_DEG) *
                       col_indices / max(ncols_coh - 1, 1))
        incidence_px[nan_mask_inc] = fill_vals
        print(f"  [WARN] {n_nan_inc} NaN incidence values filled from near/far ramp")

    inc_min_val  = float(np.nanmin(incidence_px))
    inc_max_val  = float(np.nanmax(incidence_px))
    inc_mean_val = float(np.nanmean(incidence_px))
    print(f"  Incidence (DS) : min={inc_min_val:.3f} max={inc_max_val:.3f} mean={inc_mean_val:.3f} deg")

    v_matrix   = np.full((A.shape[1], n_pixels), np.nan, dtype=np.float32)
    res_matrix = np.full((n_ifgs, n_pixels),     np.nan, dtype=np.float32)
    valid_all  = np.zeros(n_pixels, dtype=bool)
    n_chunks   = int(np.ceil(n_pixels / CHUNK_SIZE))

    for chunk_idx in range(n_chunks):
        px_start = chunk_idx * CHUNK_SIZE
        px_end   = min(px_start + CHUNK_SIZE, n_pixels)
        v_c, res_c, valid_c = weighted_lstsq_chunk(
            A, phase_matrix[:, px_start:px_end],
            coh_matrix[:, px_start:px_end], MIN_COH_WEIGHT, min_valid_ifgs)
        v_matrix[:,   px_start:px_end] = v_c
        res_matrix[:, px_start:px_end] = res_c
        valid_all[px_start:px_end]     = valid_c
        print(f"  Chunk {chunk_idx+1:3d}/{n_chunks} "
              f"px {px_start:>7,}-{px_end:>7,} "
              f"valid: {valid_c.sum()/max(px_end-px_start,1)*100:.1f}% "
              f"[{(chunk_idx+1)/n_chunks*100:.0f}%]")

    print("\n  Building time series...")
    timeseries_rad = intervals_to_timeseries(v_matrix, days)
    timeseries_rad[:, ~valid_all] = np.nan

    print("  Fitting linear velocity...")
    velocity_rad = compute_mean_velocity(timeseries_rad, days)
    velocity_rad[~valid_all] = np.nan

    print("\n  Converting to cm using per-pixel incidence angle...")
    timeseries_cm  = phase_to_displacement_cm(timeseries_rad, WAVELENGTH_M, incidence_px)
    velocity_cm_yr = phase_to_displacement_cm(velocity_rad * 365.25, WAVELENGTH_M, incidence_px)
    timeseries_cm[:, ~valid_all]  = np.nan
    velocity_cm_yr[~valid_all]    = np.nan

    n_valid  = int(np.sum(valid_all))
    vel_mean = float(np.nanmean(velocity_cm_yr))
    vel_std  = float(np.nanstd(velocity_cm_yr))
    vel_min  = float(np.nanmin(velocity_cm_yr))
    vel_max  = float(np.nanmax(velocity_cm_yr))
    res_rmse = float(np.sqrt(np.nanmean(res_matrix ** 2)))
    n_sub    = int(np.sum(velocity_cm_yr[valid_all] < 0))
    n_upl    = int(np.sum(velocity_cm_yr[valid_all] > 0))

    print(f"\n  Pixels inverted  : {n_valid:,} / {n_pixels:,} ({n_valid/n_pixels*100:.1f}%)")
    print(f"  Velocity [cm/yr] : mean={vel_mean:.3f} std={vel_std:.3f} range={vel_min:.3f}-{vel_max:.3f}")
    print(f"  Subsiding        : {n_sub:,} ({n_sub/max(n_valid,1)*100:.1f}%)")
    print(f"  Uplifting        : {n_upl:,} ({n_upl/max(n_valid,1)*100:.1f}%)")
    print(f"  Inversion RMSE   : {res_rmse:.4f} rad")

    ew_diag = ew_asymmetry_diagnostic(velocity_cm_yr, pixel_coords, ncols_coh, incidence_px)

    out_vel = WORK_DIR / f"{stem}_velocity.npy"
    out_ts  = WORK_DIR / f"{stem}_timeseries.npy"
    out_res = WORK_DIR / f"{stem}_residuals.npy"
    out_rep = WORK_DIR / f"{stem}_inversion_report.txt"
    np.save(out_vel, velocity_cm_yr)
    np.save(out_ts,  timeseries_cm)
    np.save(out_res, res_matrix)

    inc_src_label = incidence_tif.name if incidence_tif else "linear ramp"
    report = [
        f"Inversion report : {stem}\n",
        f"Timestamp        : {datetime.now().isoformat(timespec='seconds')}\n",
        f"\n[v1.4 fix]\n",
        f"  Incidence TIF  : {inc_src_label} (resolved per-network from .data/)\n",
        f"  GeoTIFF ref    : same as above\n",
        f"\n[Sign convention]\n",
        f"  Input  : SNAP radar (positive=subsidence)\n",
        f"  Output : geophysical (negative=subsidence, positive=uplift)\n",
        f"\n[Results]\n",
        f"  Pixels inverted: {n_valid:,} / {n_pixels:,}\n",
        f"  Velocity mean  : {vel_mean:.4f} cm/year\n",
        f"  Velocity std   : {vel_std:.4f} cm/year\n",
        f"  Velocity min   : {vel_min:.4f} cm/year\n",
        f"  Velocity max   : {vel_max:.4f} cm/year\n",
        f"  Subsiding      : {n_sub:,} ({n_sub/max(n_valid,1)*100:.1f}%)\n",
        f"  Uplifting      : {n_upl:,} ({n_upl/max(n_valid,1)*100:.1f}%)\n",
        f"  Inversion RMSE : {res_rmse:.4f} rad\n",
        f"\n[E-W Asymmetry]\n",
        f"  Mean West  : {ew_diag['mean_west']:+.4f} cm/yr\n",
        f"  Mean East  : {ew_diag['mean_east']:+.4f} cm/yr\n",
        f"  E-W diff   : {ew_diag['ew_diff']:+.4f} cm/yr\n",
        f"  r(col,vel) : {ew_diag['r_col_vel']:.4f}\n",
        f"  r(inc,vel) : {ew_diag['r_inc_vel']:.4f}\n",
    ]
    Path(out_rep).write_text("".join(report), encoding="utf-8")
    print(f"\n  Written: {out_vel.name}  Written: {out_ts.name}")
    print(f"  Written: {out_res.name}  Written: {out_rep.name}")

    print("\n  Exporting GeoTIFFs...")
    export_geotiffs(stem, velocity_cm_yr, velocity_rad,
                    incidence_px, pixel_coords, coh_tif, master_tif,
                    phase_source, inc_min_val, inc_max_val, inc_mean_val,
                    n_nan_inc, min_valid_ifgs, ew_diag)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    design_files = sorted(WORK_DIR.glob("*_design_matrix.npy"))
    if not design_files:
        print(f"No *_design_matrix.npy in {WORK_DIR}. Run 04 first.")
        return
    stems = [f.stem.replace("_design_matrix", "") for f in design_files]
    print("05b_inversion.py (v1.4 — per-network SNAP native incidence TIF)")
    print(f"Found {len(stems)} network(s): {stems}")
    for stem in stems:
        process_network(stem)
    print("\n" + "=" * 60)
    print("Inversion complete. Run 06_reference_point.py next.")

if __name__ == "__main__":
    main()