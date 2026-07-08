"""
06_reference_point.py
Reference point selection and subtraction.

Inputs from 05b_inversion.py are already in physical displacement units
and already atmospherically corrected at the IFG level by 05a:
  _velocity.npy    [cm/year]
  _timeseries.npy  [cm]

This script performs ONE operation only:
  Reference point subtraction in [cm/year] and [cm]

Atmospheric correction is NOT applied here. It was applied per-IFG
in 05a_atmospheric_correction.py before inversion. Applying it again
to the time series would double-correct and remove real deformation signal.

Reference-point logic:
  - A manual anchor pixel (REF_COL_ANCHOR, REF_ROW_ANCHOR) defines
    the stable area based on field knowledge
  - All DS-valid pixels within +/- REF_WINDOW_PX (Chebyshev) are candidates
  - The candidate with the highest mean coherence is selected
  - That pixel's velocity and time series are subtracted from all others

Sign convention (unchanged from 05b):
  NEGATIVE = subsidence (moving away from satellite)
  POSITIVE = uplift     (moving toward satellite)

Outputs (per network, written to WORK_DIR):
  _velocity_referenced.npy    float32 (n_valid_pixels,)         [cm/year]
  _timeseries_referenced.npy  float32 (n_dates, n_valid_pixels) [cm]
  _reference_point.csv        reference pixel metadata
  _reference_report.txt       full processing log

Changes from v1.0:
  - Added GNSS_VEL_CMYR: externally supplied vertical GNSS velocity at the
    reference point location, covering the InSAR observation period (2019-2024).
  - After relative reference subtraction, a scalar shift is applied so that
    the output velocities are expressed in an absolute geodetic frame:
        v_abs = v_relative + GNSS_VEL_CMYR
  - The time series receives the same shift as a linear ramp:
        ts_abs[t] = ts_relative[t] + GNSS_VEL_CMYR * days[t] / 365.25
  - All outputs, logs, and CSV metadata updated to record the GNSS value,
    its source, and the period it covers.
  - If GNSS_VEL_CMYR is set to None the script behaves exactly as v1.0
    (relative referencing only) so it is safe to run before GNSS data
    is available.
"""

import os
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"]    = str(_proj_data)
    os.environ["PROJ_LIB"]     = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR = Path(r"C:\bcosta\DS-InSAR\tc")

# ── Configuration ─────────────────────────────────────────────────────────────
REF_COL_ANCHOR = 2065   # Image-X in SNAP Pixel Info
REF_ROW_ANCHOR = 3210   # Image-Y in SNAP Pixel Info
REF_WINDOW_PX  = 10     # Chebyshev search radius [pixels]
NODATA         = -9999.0

# ── GNSS velocity adjustment ──────────────────────────────────────────────────
# Vertical velocity of the reference point from GNSS, in cm/year.
# Period: must match the InSAR observation window (2019-2024).
# Sign convention: NEGATIVE = subsidence, POSITIVE = uplift (same as InSAR).
GNSS_VEL_CMYR   = -0.12942    # <-- fill this in with your value
GNSS_SOURCE     = "SONEL TUKT ULR20, vertical WLS fit" 
GNSS_PERIOD     = "2019-06-01 to 2024-09-30" # for logging only


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_tif(path):
    import rasterio
    with rasterio.open(path) as src:
        arr  = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        tf   = src.transform
    return arr, meta, tf


def apply_gnss_adjustment(vel_rel, ts_rel, days_array, gnss_vel_cmyr):
    """
    Shift a relative velocity field and time series into an absolute
    geodetic frame using a GNSS-derived velocity at the reference point.

    Parameters
    ----------
    vel_rel       : (n_pixels,)       float32   relative velocity [cm/yr]
    ts_rel        : (n_dates, n_pixels) float32  relative time series [cm]
    days_array    : (n_dates,)        float      days since first acquisition
    gnss_vel_cmyr : float             GNSS vertical velocity [cm/yr]
                    NEGATIVE = subsidence, POSITIVE = uplift

    Returns
    -------
    vel_abs : (n_pixels,)           absolute velocity [cm/yr]
    ts_abs  : (n_dates, n_pixels)   absolute time series [cm]

    Method
    ------
    The GNSS velocity is the known absolute motion of the reference pixel.
    After relative subtraction, that pixel sits at zero. Adding the GNSS
    velocity shifts the entire field so the reference pixel now reads
    gnss_vel_cmyr, and all other pixels are offset accordingly:

        v_abs(p) = v_rel(p) + gnss_vel_cmyr

    The time series gets the equivalent linear ramp:

        ts_abs(t, p) = ts_rel(t, p) + gnss_vel_cmyr * (days[t] / 365.25)
    """
    vel_abs = vel_rel + np.float32(gnss_vel_cmyr)

    t_years  = (days_array / 365.25).astype(np.float32)          # (n_dates,)
    ramp     = (gnss_vel_cmyr * t_years).astype(np.float32)       # (n_dates,)
    ts_abs   = ts_rel + ramp[:, np.newaxis]                       # broadcast

    return vel_abs, ts_abs


def evaluate_reference_candidates(coh_2d, pixel_coords, velocity, timeseries,
                                   anchor_row, anchor_col, window_px):
    """
    Find the best reference pixel within a Chebyshev window of the anchor.
    Selection criterion: highest mean coherence among DS-valid candidates
    with finite velocity.
    """
    rows_arr = pixel_coords[:, 0]
    cols_arr = pixel_coords[:, 1]
    coh_vals = coh_2d[rows_arr, cols_arr]

    in_window = (
        (np.abs(rows_arr - anchor_row) <= window_px) &
        (np.abs(cols_arr - anchor_col) <= window_px)
    )
    valid = (in_window &
             np.isfinite(coh_vals) &
             (coh_vals > 0) &
             np.isfinite(velocity))

    idx = np.where(valid)[0]
    if idx.size == 0:
        raise RuntimeError(
            f"No valid DS pixels found within {window_px} px of "
            f"anchor (row={anchor_row}, col={anchor_col}).\n"
            f"Increase REF_WINDOW_PX or verify pixel_coords alignment."
        )

    cand_coh    = coh_vals[idx]
    cand_vel    = velocity[idx]
    cand_ts     = timeseries[:, idx]
    cand_ts_std = np.nanstd(cand_ts, axis=0)

    best_local = int(np.nanargmax(cand_coh))
    best_idx   = int(idx[best_local])

    return {
        "n_candidates" : int(idx.size),
        "best_idx"     : best_idx,
        "best_row"     : int(rows_arr[best_idx]),
        "best_col"     : int(cols_arr[best_idx]),
        "best_coh"     : float(coh_vals[best_idx]),
        "best_vel"     : float(velocity[best_idx]),
        "best_ts_std"  : float(np.nanstd(timeseries[:, best_idx])),
        "best_dist"    : int(max(abs(rows_arr[best_idx] - anchor_row),
                                 abs(cols_arr[best_idx] - anchor_col))),
        "coh_min"      : float(np.nanmin(cand_coh)),
        "coh_med"      : float(np.nanmedian(cand_coh)),
        "coh_max"      : float(np.nanmax(cand_coh)),
        "vel_min"      : float(np.nanmin(cand_vel)),
        "vel_med"      : float(np.nanmedian(cand_vel)),
        "vel_max"      : float(np.nanmax(cand_vel)),
        "tsstd_min"    : float(np.nanmin(cand_ts_std)),
        "tsstd_med"    : float(np.nanmedian(cand_ts_std)),
        "tsstd_max"    : float(np.nanmax(cand_ts_std)),
    }


# ── Per-network processing ────────────────────────────────────────────────────
def process_network(stem):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    vel_path    = WORK_DIR / f"{stem}_velocity.npy"
    ts_path     = WORK_DIR / f"{stem}_timeseries.npy"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    coh_tif     = WORK_DIR / f"{stem}_mean_coherence.tif"
    dates_path  = WORK_DIR / f"{stem}_date_list.csv"

    for p in [vel_path, ts_path, coords_path, coh_tif, dates_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} -- run scripts 03-05 first.")
            return

    # -- Load inputs -----------------------------------------------------------
    velocity     = np.load(vel_path).astype(np.float32)
    timeseries   = np.load(ts_path).astype(np.float32)
    pixel_coords = np.load(coords_path)
    coh_2d, meta, transform = read_tif(coh_tif)
    coh_2d[coh_2d == NODATA] = np.nan

    dates_list = []
    days_list  = []
    with open(dates_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates_list.append(row["date"])
            days_list.append(int(row["days_since_first"]))
    days_array = np.array(days_list, dtype=np.float32)

    n_dates, n_pixels = timeseries.shape

    print(f"  Velocity array  : {velocity.shape}  [cm/year]")
    print(f"  Time series     : {timeseries.shape}  [cm]")
    print(f"  Coherence image : {coh_2d.shape}")
    print(f"  Dates           : {n_dates}  ({dates_list[0]} -> {dates_list[-1]})")
    print(f"  Atmospheric correction : already applied per-IFG in 05a")

    # -- Reference point selection ---------------------------------------------
    print(f"\n  Reference point selection...")
    print(f"  Anchor (col, row) : ({REF_COL_ANCHOR}, {REF_ROW_ANCHOR})")
    print(f"  Search window     : +/- {REF_WINDOW_PX} px (Chebyshev)")

    qc = evaluate_reference_candidates(
        coh_2d, pixel_coords,
        velocity, timeseries,
        anchor_row = REF_ROW_ANCHOR,
        anchor_col = REF_COL_ANCHOR,
        window_px  = REF_WINDOW_PX,
    )

    ref_idx = qc["best_idx"]
    ref_r   = qc["best_row"]
    ref_c   = qc["best_col"]
    ref_coh = qc["best_coh"]
    ref_vel = qc["best_vel"]

    print(f"  Candidates found        : {qc['n_candidates']}")
    print(f"  Candidate coherence     : "
          f"min={qc['coh_min']:.4f}  med={qc['coh_med']:.4f}  max={qc['coh_max']:.4f}")
    print(f"  Candidate velocity      : "
          f"min={qc['vel_min']:.3f}  med={qc['vel_med']:.3f}  max={qc['vel_max']:.3f} cm/year")
    print(f"  Candidate ts std        : "
          f"min={qc['tsstd_min']:.3f}  med={qc['tsstd_med']:.3f}  max={qc['tsstd_max']:.3f} cm")
    print(f"  Selected (col, row)     : ({ref_c}, {ref_r})")
    print(f"  Offset from anchor      : dc={ref_c - REF_COL_ANCHOR}  dr={ref_r - REF_ROW_ANCHOR}")
    print(f"  Reference coherence     : {ref_coh:.4f}")
    print(f"  Reference ts std        : {qc['best_ts_std']:.3f} cm")
    print(f"  Reference velocity (InSAR) : {ref_vel:.4f} cm/year")

    # -- Step 1: relative subtraction (same as v1.0) ---------------------------
    ref_ts  = timeseries[:, ref_idx].copy()
    ts_rel  = timeseries - ref_ts[:, np.newaxis]
    vel_rel = velocity   - ref_vel

    # -- Step 2: GNSS adjustment (new in v2.0) ---------------------------------
    if GNSS_VEL_CMYR is not None:
        print(f"\n  GNSS adjustment...")
        print(f"  GNSS velocity           : {GNSS_VEL_CMYR:+.4f} cm/year")
        print(f"  Source                  : {GNSS_SOURCE}")
        print(f"  Period                  : {GNSS_PERIOD}")

        vel_out, ts_out = apply_gnss_adjustment(
            vel_rel, ts_rel, days_array, GNSS_VEL_CMYR
        )

        print(f"  InSAR ref pixel (pre-GNSS)  : {ref_vel:+.4f} cm/year")
        print(f"  InSAR ref pixel (post-GNSS) : "
              f"{float(vel_out[ref_idx]):+.4f} cm/year  "
              f"[should equal GNSS = {GNSS_VEL_CMYR:+.4f}]")

        gnss_mode = "absolute (GNSS-adjusted)"
    else:
        print(f"\n  GNSS_VEL_CMYR = None — skipping GNSS adjustment (relative only).")
        vel_out  = vel_rel
        ts_out   = ts_rel
        gnss_mode = "relative only (no GNSS)"

    # -- Post-adjustment stats -------------------------------------------------
    vel_mean    = float(np.nanmean(vel_out))
    vel_std     = float(np.nanstd(vel_out))
    vel_min     = float(np.nanmin(vel_out))
    vel_max     = float(np.nanmax(vel_out))
    n_finite    = int(np.sum(np.isfinite(vel_out)))
    n_subsiding = int(np.sum(vel_out < 0))
    n_uplifting = int(np.sum(vel_out > 0))
    pct_sub     = n_subsiding / n_finite * 100 if n_finite > 0 else 0.0

    print(f"\n  Velocity [cm/year] : mean={vel_mean:.3f}  std={vel_std:.3f}")
    print(f"  Velocity range     : {vel_min:.3f} - {vel_max:.3f} cm/year")
    print(f"  Subsiding pixels   : {n_subsiding:,} ({pct_sub:.1f}%)")
    print(f"  Uplifting pixels   : {n_uplifting:,} ({100 - pct_sub:.1f}%)")

    # -- Save outputs ----------------------------------------------------------
    out_vel = WORK_DIR / f"{stem}_velocity_referenced.npy"
    out_ts  = WORK_DIR / f"{stem}_timeseries_referenced.npy"
    out_ref = WORK_DIR / f"{stem}_reference_point.csv"
    out_rep = WORK_DIR / f"{stem}_reference_report.txt"

    np.save(out_vel, vel_out)
    np.save(out_ts,  ts_out)

    with open(out_ref, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stem", "anchor_col", "anchor_row", "window_px",
            "ref_col", "ref_row", "ref_idx", "ref_coh", "n_candidates",
            "ref_vel_insar_cmyr", "ref_ts_std_cm",
            "gnss_vel_cmyr", "gnss_source", "gnss_period", "gnss_mode",
            "cand_coh_min", "cand_coh_med", "cand_coh_max",
            "cand_vel_min_cmyr", "cand_vel_med_cmyr", "cand_vel_max_cmyr",
            "cand_tsstd_min_cm", "cand_tsstd_med_cm", "cand_tsstd_max_cm",
        ])
        writer.writerow([
            stem, REF_COL_ANCHOR, REF_ROW_ANCHOR, REF_WINDOW_PX,
            ref_c, ref_r, ref_idx, round(ref_coh, 4), qc["n_candidates"],
            round(ref_vel, 4), round(qc["best_ts_std"], 4),
            GNSS_VEL_CMYR if GNSS_VEL_CMYR is not None else "N/A",
            GNSS_SOURCE, GNSS_PERIOD, gnss_mode,
            round(qc["coh_min"], 4), round(qc["coh_med"], 4), round(qc["coh_max"], 4),
            round(qc["vel_min"], 4), round(qc["vel_med"], 4), round(qc["vel_max"], 4),
            round(qc["tsstd_min"], 4), round(qc["tsstd_med"], 4), round(qc["tsstd_max"], 4),
        ])

    report_lines = [
        f"Reference report  : {stem}\n",
        f"Timestamp         : {datetime.now().isoformat(timespec='seconds')}\n",
        f"Script version    : v2.0 (GNSS velocity adjustment)\n",
        f"\n[Pipeline context]\n",
        f"  Atmospheric correction : applied per-IFG in 05a (NOT repeated here)\n",
        f"  Unit conversion        : applied in 05b (outputs already in cm/year, cm)\n",
        f"\n[Reference point logic]\n",
        f"  Mode              : manual anchor + local coherence maximisation\n",
        f"  Anchor (col, row) : ({REF_COL_ANCHOR}, {REF_ROW_ANCHOR})\n",
        f"  Search window     : +/- {REF_WINDOW_PX} px (Chebyshev)\n",
        f"  Candidates        : {qc['n_candidates']}\n",
        f"  Candidate coherence  : "
        f"{qc['coh_min']:.4f} / {qc['coh_med']:.4f} / {qc['coh_max']:.4f} (min/med/max)\n",
        f"  Candidate velocity   : "
        f"{qc['vel_min']:.4f} / {qc['vel_med']:.4f} / {qc['vel_max']:.4f} cm/year\n",
        f"  Selected (col, row)  : ({ref_c}, {ref_r})\n",
        f"  Offset from anchor   : dc={ref_c - REF_COL_ANCHOR}  dr={ref_r - REF_ROW_ANCHOR}\n",
        f"  Reference coherence  : {ref_coh:.4f}\n",
        f"  Reference velocity (InSAR, pre-GNSS) : {ref_vel:.4f} cm/year\n",
        f"\n[GNSS adjustment]\n",
        f"  Mode              : {gnss_mode}\n",
        f"  GNSS velocity     : "
        f"{GNSS_VEL_CMYR if GNSS_VEL_CMYR is not None else 'N/A'} cm/year\n",
        f"  Source            : {GNSS_SOURCE if GNSS_SOURCE else 'not specified'}\n",
        f"  Period            : {GNSS_PERIOD}\n",
        f"  Method            : scalar shift  v_abs = v_rel + GNSS_VEL\n",
        f"                      ts_abs(t) = ts_rel(t) + GNSS_VEL * (days[t] / 365.25)\n",
        f"\n[Results after referencing]\n",
        f"  Velocity mean    : {vel_mean:.4f} cm/year\n",
        f"  Velocity std     : {vel_std:.4f} cm/year\n",
        f"  Velocity min     : {vel_min:.4f} cm/year\n",
        f"  Velocity max     : {vel_max:.4f} cm/year\n",
        f"  Subsiding pixels : {n_subsiding:,} ({pct_sub:.1f}%)\n",
        f"  Uplifting pixels : {n_uplifting:,} ({100 - pct_sub:.1f}%)\n",
        f"\n[Sign convention]\n",
        f"  negative velocity = subsidence (moving away from satellite)\n",
        f"  positive velocity = uplift     (moving toward satellite)\n",
    ]
    Path(out_rep).write_text("".join(report_lines), encoding="utf-8")

    print(f"\n  Written: {out_vel.name}  ({vel_out.nbytes / 1e6:.1f} MB)  [cm/year]")
    print(f"  Written: {out_ts.name}   ({ts_out.nbytes / 1e6:.1f} MB)  [cm]")
    print(f"  Written: {out_ref.name}")
    print(f"  Written: {out_rep.name}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    vel_files = sorted(WORK_DIR.glob("*_velocity.npy"))
    if not vel_files:
        print(f"No *_velocity.npy files found in {WORK_DIR}")
        print("Run 05b_inversion.py first.")
        return

    stems = [f.stem.replace("_velocity", "") for f in vel_files]
    stems = [s for s in stems if "_referenced" not in s]
    print(f"Found {len(stems)} network(s): {stems}")

    for stem in stems:
        process_network(stem)

    print("\n" + "=" * 60)
    print("Reference point correction complete.")
    print("Outputs in cm/year and cm -- run 07_export_geotiff.py next.")


if __name__ == "__main__":
    main()