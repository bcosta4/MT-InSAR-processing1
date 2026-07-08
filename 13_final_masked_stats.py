"""
13_final_masked_stats.py
Statistics on all masked TIFs. Two sections:
  1. Per-TIF detailed stats for every *_masked.tif in MASKED_DIR
     (writes {stem}_stats.txt + 00_masked_summary.txt)
  2. Slope-masking delta table comparing baseline vs slope-filtered
     velocity rasters (writes slope_mask_stats.csv + slope_mask_delta_stats.csv)

No masking is performed here -- statistics only.
"""

import os
from pathlib import Path

_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data    = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"]    = str(_proj_data)
    os.environ["PROJ_LIB"]     = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

import numpy as np
import rasterio
import pandas as pd
from datetime import datetime


# ==============================================================================
# Configuration
# ==============================================================================

# FIX: MASKED_DIR must match 12.OUT_DIR  (inputs/shifted, NOT results/).
# Script 12 writes *_masked.tif to SHIFTED_DIR = inputs/shifted.
# Script 13 Part 1 reads *_masked.tif from MASKED_DIR -- they must agree.
WORK_DIR    = Path(r"E:\Processing\InSAR\inputs")
MASKED_DIR  = WORK_DIR / "shifted"          # quality-masked TIFs from script 12
RESULTS_DIR = Path(r"E:/Processing/InSAR/results")  # slope-masked TIFs from script 12
NODATA      = -9999.0

UNITS_MAP = {
    "_velocity_cmyr_masked.tif":             "cm/yr",
    "_velocity_map_masked.tif":              "cm/yr",
    "_velocity_rad_per_day_masked.tif":      "rad/day",
    "_mean_coherence_masked.tif":            "dimensionless",
    "_coherence_mask_masked.tif":            "binary",
    "_residuals_map_masked.tif":             "rad",
    "_velocity_ddt_map_masked.tif":          "cm/DDT",
    "_velocity_gddt_map_masked.tif":         "cm/GDDT",
    "_velocity_ddt_gddt_diff_masked.tif":    "cm/DDT",
    "_incidence_angle_ds_pixels_masked.tif": "degrees",
    "_timeseries_stack_masked.tif":          "cm",
}

CLIP_RANGES = {
    "_velocity_cmyr_masked.tif":             (-30.0,  10.0),
    "_velocity_map_masked.tif":              (-30.0,  10.0),
    "_velocity_rad_per_day_masked.tif":      (-1.0,   1.0),
    "_velocity_ddt_map_masked.tif":          (-0.1,   0.05),
    "_velocity_gddt_map_masked.tif":         (-0.1,   0.05),
    "_velocity_ddt_gddt_diff_masked.tif":    (-0.05,  0.05),
    "_mean_coherence_masked.tif":            (0.3,    1.0),
    "_coherence_mask_masked.tif":            (0.0,    1.0),
    "_residuals_map_masked.tif":             (-10.0,  10.0),
    "_incidence_angle_ds_pixels_masked.tif": (15.0,   50.0),
    "_timeseries_stack_masked.tif":          (-50.0,  20.0),
}

SIGN_CONVENTION_TIFS = {
    "_velocity_cmyr_masked.tif",
    "_velocity_map_masked.tif",
    "_velocity_ddt_map_masked.tif",
    "_velocity_gddt_map_masked.tif",
    "_velocity_ddt_gddt_diff_masked.tif",
}

PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

INSAR_STEMS = [
    "Network2_ifg_unw_Stack_TC1_velocity_map_masked",
    "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_gddt_diff_masked",
]

SHORT_NAMES = {
    "Network2_ifg_unw_Stack_TC1_velocity_map_masked":           "velocity",
    "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked":      "velocity_gddt",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked":       "velocity_ddt",
    "Network2_ifg_unw_Stack_TC1_velocity_ddt_gddt_diff_masked": "ddt_gddt_diff",
}

THRESHOLDS = [2, 4, 5]


# ==============================================================================
# Part 1 helpers -- detailed per-TIF stats
# ==============================================================================

def get_units(filename):
    for suffix, units in UNITS_MAP.items():
        if filename.endswith(suffix):
            return units
    return "unknown"


def get_clip_range(filename):
    for suffix, rng in CLIP_RANGES.items():
        if filename.endswith(suffix):
            return rng
    return None


def read_valid(tif_path):
    bands = {}
    with rasterio.open(tif_path) as src:
        n_bands     = src.count
        file_nodata = src.nodata
        meta = {
            "crs":       src.crs.to_string() if src.crs else "unknown",
            "epsg":      src.crs.to_epsg()   if src.crs else None,
            "transform": src.transform,
            "shape":     (src.height, src.width),
            "n_bands":   n_bands,
            "dtype":     src.dtypes[0],
            "nodata":    file_nodata,
        }
        for b in range(1, n_bands + 1):
            data = src.read(b).astype(np.float64)
            mask = np.isfinite(data)
            if file_nodata is not None:
                tol   = max(abs(file_nodata) * 1e-5, 1.0)
                mask &= np.abs(data - file_nodata) > tol
            mask &= (data != NODATA)
            F32_MAX = 3.4e+38
            mask &= (data > -F32_MAX) & (data < F32_MAX)
            bands[b] = data[mask].astype(np.float32)
    return bands, meta


def band_stats(valid, units, sign_convention=False):
    n = len(valid)
    if n == 0:
        return None
    pcts = np.percentile(valid, PERCENTILES)
    s = {
        "n_valid":     n,
        "units":       units,
        "mean":        float(np.mean(valid)),
        "median":      float(np.median(valid)),
        "std":         float(np.std(valid)),
        "min":         float(np.min(valid)),
        "max":         float(np.max(valid)),
        "percentiles": dict(zip(PERCENTILES, pcts.tolist())),
        "iqr":         float(pcts[PERCENTILES.index(75)] - pcts[PERCENTILES.index(25)]),
    }
    if sign_convention:
        s["n_negative"]    = int(np.sum(valid < 0))
        s["n_positive"]    = int(np.sum(valid > 0))
        s["n_zero"]        = int(np.sum(valid == 0))
        s["pct_subsiding"] = s["n_negative"] / max(n, 1) * 100
        s["pct_uplifting"] = s["n_positive"] / max(n, 1) * 100
    return s


def format_report(tif_path, bands, raw_counts, meta, units, sign_convention, clip, timestamp):
    sep  = "=" * 65
    sep2 = "-" * 65
    clip_str = f"{clip[0]} to {clip[1]} {units}" if clip else "none"
    lines = [
        f"{sep}\n",
        f"FILE     : {tif_path.name}\n",
        f"Timestamp: {timestamp}\n",
        f"{sep}\n",
        f"CRS      : {meta['crs']}\n",
        f"EPSG     : {meta['epsg']}\n",
        f"Shape    : {meta['shape'][0]} rows x {meta['shape'][1]} cols\n",
        f"Bands    : {meta['n_bands']}\n",
        f"Dtype    : {meta['dtype']}\n",
        f"NoData   : {meta['nodata']}\n",
        f"Units    : {units}\n",
        f"Clip     : {clip_str}\n",
        f"{sep2}\n",
    ]
    for b, valid in bands.items():
        s         = band_stats(valid, units, sign_convention)
        n_raw     = raw_counts.get(b, 0)
        n_clipped = n_raw - len(valid)
        bheader   = f"Band {b}" if meta["n_bands"] > 1 else "Statistics"
        lines.append(f"\n{bheader}\n{sep2}\n")
        if s is None:
            lines += [f"  Raw read    : {n_raw:,}\n",
                      f"  Clipped out : {n_clipped:,}\n",
                      "  No valid pixels after filtering.\n"]
            continue
        lines += [
            f"  Raw read    : {n_raw:,} pixels\n",
            f"  Clipped out : {n_clipped:,}  (fill/artifact values outside plausible range)\n",
            f"  Valid pixels: {s['n_valid']:,}\n\n",
            f"  Mean        : {s['mean']:+.6f} {units}\n",
            f"  Median      : {s['median']:+.6f} {units}\n",
            f"  Std dev     : {s['std']:.6f} {units}\n",
            f"  Min         : {s['min']:+.6f} {units}\n",
            f"  Max         : {s['max']:+.6f} {units}\n",
            f"  IQR (25-75%): {s['iqr']:.6f} {units}\n",
            f"\n  Percentiles:\n",
        ]
        for pct, val in s["percentiles"].items():
            lines.append(f"    p{pct:02d} : {val:+.6f} {units}\n")
        if sign_convention:
            lines += [
                f"\n  Sign convention (negative = subsidence, positive = uplift):\n",
                f"  Subsiding pixels : {s['n_negative']:,}  ({s['pct_subsiding']:.1f}%)\n",
                f"  Uplifting pixels : {s['n_positive']:,}  ({s['pct_uplifting']:.1f}%)\n",
                f"  Zero             : {s['n_zero']:,}\n",
            ]
    lines.append(f"\n{sep}\n")
    return lines


def format_summary_row(tif_name, b1_stats, n_clipped):
    if b1_stats is None:
        return f"  {tif_name}\n    NO VALID PIXELS  (clipped: {n_clipped:,})\n\n"
    sc = ""
    if "pct_subsiding" in b1_stats:
        sc = f"  sub={b1_stats['pct_subsiding']:.1f}%  upl={b1_stats['pct_uplifting']:.1f}%"
    return (
        f"  {tif_name}\n"
        f"    n={b1_stats['n_valid']:,}  "
        f"mean={b1_stats['mean']:+.6f}  "
        f"std={b1_stats['std']:.6f}  "
        f"[{b1_stats['min']:+.6f}, {b1_stats['max']:+.6f}]"
        f"  {b1_stats['units']}{sc}"
        f"  clipped={n_clipped:,}\n\n"
    )


# ==============================================================================
# Part 2 helpers -- slope delta stats
# ==============================================================================

def raster_stats(path, fallback_nodata=-9999.0):
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else fallback_nodata
    valid = data[(data != nodata) & ~np.isnan(data)]
    if len(valid) == 0:
        return None
    return {
        "n_valid_pixels":  int(len(valid)),
        "n_masked_pixels": int(data.size - len(valid)),
        "pct_masked":      round(100 * (data.size - len(valid)) / data.size, 2),
        "mean":            round(float(np.mean(valid)),  4),
        "median":          round(float(np.median(valid)),4),
        "std":             round(float(np.std(valid)),   4),
        "min":             round(float(np.min(valid)),   4),
        "max":             round(float(np.max(valid)),   4),
        "p5":              round(float(np.percentile(valid,  5)), 4),
        "p25":             round(float(np.percentile(valid, 25)), 4),
        "p75":             round(float(np.percentile(valid, 75)), 4),
        "p95":             round(float(np.percentile(valid, 95)), 4),
        "pct_negative":    round(100 * np.sum(valid < 0) / len(valid), 2),
        "pct_positive":    round(100 * np.sum(valid > 0) / len(valid), 2),
    }


# ==============================================================================
# Main
# ==============================================================================

def main():

    timestamp = datetime.now().isoformat(timespec="seconds")
    print("13_final_masked_stats.py")
    print("=" * 65)

    # ── Part 1: Detailed per-TIF stats ───────────────────────────────────────
    print("PART 1 -- Detailed stats on *_masked.tif files")
    print(f"  Input dir : {MASKED_DIR}")

    tifs = sorted(MASKED_DIR.glob("*_masked.tif"))
    if not tifs:
        print(f"  WARNING: No *_masked.tif files found in {MASKED_DIR}")
        print("  Run 12_apply_quality_slope_mask.py first.")
    else:
        print(f"  Found     : {len(tifs)} masked TIF(s)")
        print("=" * 65)

        summary_lines = [
            "MASKED TIF STATISTICS SUMMARY\n",
            f"Timestamp : {timestamp}\n",
            f"Directory : {MASKED_DIR}\n",
            f"TIFs      : {len(tifs)}\n",
            "=" * 65 + "\n\n",
        ]

        for tif in tifs:
            units       = get_units(tif.name)
            sign_conv   = any(tif.name.endswith(s) for s in SIGN_CONVENTION_TIFS)
            clip        = get_clip_range(tif.name)
            bands, meta = read_valid(tif)
            raw_counts  = {b: len(v) for b, v in bands.items()}
            if clip is not None:
                lo, hi = clip
                bands  = {b: v[(v >= lo) & (v <= hi)] for b, v in bands.items()}
            clipped_counts = {b: raw_counts[b] - len(bands[b]) for b in bands}

            report_lines = format_report(
                tif, bands, raw_counts, meta, units, sign_conv, clip, timestamp
            )
            (MASKED_DIR / (tif.stem + "_stats.txt")).write_text(
                "".join(report_lines), encoding="utf-8"
            )

            b1         = bands.get(1, np.array([]))
            b1_stats   = band_stats(b1, units, sign_conv)
            n_clipped1 = clipped_counts.get(1, 0)

            if b1_stats:
                sc_str = ""
                if sign_conv:
                    sc_str = (f"  sub={b1_stats['pct_subsiding']:.1f}%"
                              f"  upl={b1_stats['pct_uplifting']:.1f}%")
                print(f"  {tif.name}")
                print(f"    n={b1_stats['n_valid']:,}  mean={b1_stats['mean']:+.6f}  "
                      f"std={b1_stats['std']:.6f}  "
                      f"[{b1_stats['min']:+.6f}, {b1_stats['max']:+.6f}] "
                      f"{units}{sc_str}  clipped={n_clipped1:,}")
            else:
                print(f"  {tif.name}  -- NO VALID PIXELS (clipped={n_clipped1:,})")

            summary_lines.append(format_summary_row(tif.name, b1_stats, n_clipped1))

        out_summary = MASKED_DIR / "00_masked_summary.txt"
        out_summary.write_text("".join(summary_lines), encoding="utf-8")
        print(f"\n  Per-TIF stats -> {MASKED_DIR}")
        print(f"  Summary       -> {out_summary.name}")

    # ── Part 2: Slope masking delta table ────────────────────────────────────
    print()
    print("=" * 65)
    print("PART 2 -- Slope masking delta stats")
    print(f"  Results dir : {RESULTS_DIR}")
    print("=" * 65)

    rows = []
    for stem in INSAR_STEMS:
        short     = SHORT_NAMES[stem]
        base_path = RESULTS_DIR / f"{stem}.tif"
        if base_path.exists():
            s = raster_stats(base_path)
            if s:
                rows.append({"raster": short, "slope_threshold": "baseline", **s})
        else:
            print(f"  WARNING -- baseline not found: {base_path.name}")

        for thresh in THRESHOLDS:
            p = RESULTS_DIR / f"{stem}_slope{thresh}deg.tif"
            if not p.exists():
                print(f"  Skipping (not found): {p.name}")
                continue
            s = raster_stats(p)
            if s:
                rows.append({"raster": short, "slope_threshold": f"{thresh}deg", **s})

    if rows:
        df       = pd.DataFrame(rows)
        out_full = RESULTS_DIR / "slope_mask_stats.csv"
        df.to_csv(out_full, index=False)
        print(f"  Full stats -> {out_full}")

        cols = ["raster", "slope_threshold", "n_valid_pixels", "pct_masked",
                "mean", "median", "std", "p5", "p95", "pct_negative"]
        print("\n" + df[cols].to_string(index=False))

        delta_rows = []
        for stem in INSAR_STEMS:
            short = SHORT_NAMES[stem]
            base  = df[(df["raster"] == short) & (df["slope_threshold"] == "baseline")]
            if base.empty:
                continue
            b = base.iloc[0]
            for thresh in THRESHOLDS:
                t_row = df[(df["raster"] == short) & (df["slope_threshold"] == f"{thresh}deg")]
                if t_row.empty:
                    continue
                t = t_row.iloc[0]
                delta_rows.append({
                    "raster":          short,
                    "slope_threshold": f"{thresh}deg",
                    "pixels_removed":  int(b["n_valid_pixels"] - t["n_valid_pixels"]),
                    "pct_masked":      t["pct_masked"],
                    "delta_mean":      round(t["mean"]         - b["mean"],         4),
                    "delta_median":    round(t["median"]       - b["median"],       4),
                    "delta_std":       round(t["std"]          - b["std"],          4),
                    "delta_p5":        round(t["p5"]           - b["p5"],           4),
                    "delta_p95":       round(t["p95"]          - b["p95"],          4),
                    "delta_pct_neg":   round(t["pct_negative"] - b["pct_negative"], 2),
                })

        delta_df  = pd.DataFrame(delta_rows)
        out_delta = RESULTS_DIR / "slope_mask_delta_stats.csv"
        delta_df.to_csv(out_delta, index=False)
        print("\n" + delta_df.to_string(index=False))
        print(f"\n  Delta stats -> {out_delta}")

    print("\n" + "=" * 65)
    print("Done.")


if __name__ == "__main__":
    main()
