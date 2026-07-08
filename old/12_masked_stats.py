"""
12_masked_stats.py
Standalone statistics script. Reads all *_masked.tif files from the masked/
folder and writes one .txt report per TIF, plus a combined summary report.

No masking is performed here — just statistics on already-masked TIFs.

Inputs  : C:\bcosta\DS-InSAR\tc\shifted\masked\  (*_masked.tif)
Outputs : same folder
            {stem}_stats.txt   — per-TIF detailed report
            00_masked_summary.txt — combined summary of all TIFs
"""

import os
from pathlib import Path

# ── PROJ path fix ──────────────────────────────────────────────────────────────
_conda_prefix = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj_data    = _conda_prefix / "Library" / "share" / "proj"
if _proj_data.exists():
    os.environ["PROJ_DATA"]    = str(_proj_data)
    os.environ["PROJ_LIB"]     = str(_proj_data)
    os.environ["PROJ_NETWORK"] = "OFF"

import numpy as np
import rasterio
from datetime import datetime

# ── Configuration ──────────────────────────────────────────────────────────────
MASKED_DIR = Path(r"C:\bcosta\DS-InSAR\results")
NODATA     = -9999.0

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

# Physically plausible value ranges — anything outside is a fill/artifact
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

# ── Helpers ────────────────────────────────────────────────────────────────────
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
    """
    Read all bands. Masks out:
      - the file's own nodata value (with float tolerance)
      - the config NODATA sentinel (-9999)
      - NaN and ±inf
      - float32 fill value (~3.4e+38)
    Returns float32 valid arrays per band.
    """
    bands = {}
    with rasterio.open(tif_path) as src:
        n_bands    = src.count
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

            # Build mask: start with finite values only
            mask = np.isfinite(data)

            # Mask file nodata (with tolerance for float imprecision)
            if file_nodata is not None:
                tol = max(abs(file_nodata) * 1e-5, 1.0)
                mask &= np.abs(data - file_nodata) > tol

            # Mask config nodata
            mask &= (data != NODATA)

            # Mask float32 max fill value (any value beyond float32 range)
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
        "iqr":         float(pcts[PERCENTILES.index(75)] -
                             pcts[PERCENTILES.index(25)]),
    }
    if sign_convention:
        s["n_negative"]    = int(np.sum(valid < 0))
        s["n_positive"]    = int(np.sum(valid > 0))
        s["n_zero"]        = int(np.sum(valid == 0))
        s["pct_subsiding"] = s["n_negative"] / max(n, 1) * 100
        s["pct_uplifting"] = s["n_positive"] / max(n, 1) * 100
    return s


def format_report(tif_path, bands, raw_counts, meta, units,
                  sign_convention, clip, timestamp):
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
        s = band_stats(valid, units, sign_convention)
        n_raw     = raw_counts.get(b, 0)
        n_clipped = n_raw - len(valid)
        bheader   = f"Band {b}" if meta["n_bands"] > 1 else "Statistics"
        lines.append(f"\n{bheader}\n")
        lines.append(f"{sep2}\n")

        if s is None:
            lines.append(f"  Raw read     : {n_raw:,} pixels\n")
            lines.append(f"  Clipped out  : {n_clipped:,}  "
                         f"(fill/artifact values outside plausible range)\n")
            lines.append("  No valid pixels after filtering.\n")
            continue

        lines += [
            f"  Raw read     : {n_raw:,} pixels\n",
            f"  Clipped out  : {n_clipped:,}  "
            f"(fill/artifact values outside plausible range)\n",
            f"  Valid pixels : {s['n_valid']:,}\n",
            f"\n",
            f"  Mean         : {s['mean']:+.6f} {units}\n",
            f"  Median       : {s['median']:+.6f} {units}\n",
            f"  Std dev      : {s['std']:.6f} {units}\n",
            f"  Min          : {s['min']:+.6f} {units}\n",
            f"  Max          : {s['max']:+.6f} {units}\n",
            f"  IQR (25-75%) : {s['iqr']:.6f} {units}\n",
            f"\n  Percentiles:\n",
        ]
        for pct, val in s["percentiles"].items():
            lines.append(f"    p{pct:02d}  : {val:+.6f} {units}\n")

        if sign_convention:
            lines += [
                f"\n  Sign convention  (negative = subsidence, positive = uplift):\n",
                f"  Subsiding pixels : {s['n_negative']:,}  "
                f"({s['pct_subsiding']:.1f}%)\n",
                f"  Uplifting pixels : {s['n_positive']:,}  "
                f"({s['pct_uplifting']:.1f}%)\n",
                f"  Zero             : {s['n_zero']:,}\n",
            ]
    lines.append(f"\n{sep}\n")
    return lines


def format_summary_row(tif_name, b1_valid, b1_stats, n_clipped):
    if b1_stats is None:
        return f"  {tif_name}\n    NO VALID PIXELS  (clipped: {n_clipped:,})\n\n"
    sc = ""
    if "pct_subsiding" in b1_stats:
        sc = (f"  sub={b1_stats['pct_subsiding']:.1f}%"
              f"  upl={b1_stats['pct_uplifting']:.1f}%")
    return (
        f"  {tif_name}\n"
        f"    n={b1_stats['n_valid']:,}  "
        f"mean={b1_stats['mean']:+.6f}  "
        f"std={b1_stats['std']:.6f}  "
        f"[{b1_stats['min']:+.6f}, {b1_stats['max']:+.6f}]"
        f"  {b1_stats['units']}"
        f"{sc}"
        f"  clipped={n_clipped:,}\n\n"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    tifs = sorted(MASKED_DIR.glob("*_masked.tif"))
    if not tifs:
        print(f"No *_masked.tif files found in {MASKED_DIR}")
        return

    timestamp = datetime.now().isoformat(timespec="seconds")
    print("12_masked_stats.py")
    print(f"Input dir : {MASKED_DIR}")
    print(f"Found     : {len(tifs)} masked TIF(s)")
    print("=" * 65)

    summary_lines = [
        f"MASKED TIF STATISTICS SUMMARY\n",
        f"Timestamp : {timestamp}\n",
        f"Directory : {MASKED_DIR}\n",
        f"TIFs      : {len(tifs)}\n",
        "=" * 65 + "\n\n",
    ]

    for tif in tifs:
        units     = get_units(tif.name)
        sign_conv = any(tif.name.endswith(s) for s in SIGN_CONVENTION_TIFS)
        clip      = get_clip_range(tif.name)

        bands, meta = read_valid(tif)

        # Track raw counts before clip
        raw_counts = {b: len(v) for b, v in bands.items()}

        # Apply physical clip range
        if clip is not None:
            lo, hi = clip
            bands = {b: v[(v >= lo) & (v <= hi)] for b, v in bands.items()}

        clipped_counts = {b: raw_counts[b] - len(bands[b]) for b in bands}

        # Per-TIF report
        report_lines = format_report(
            tif, bands, raw_counts, meta, units, sign_conv, clip, timestamp
        )
        out_txt = MASKED_DIR / (tif.stem + "_stats.txt")
        out_txt.write_text("".join(report_lines), encoding="utf-8")

        # Console + summary
        b1          = bands.get(1, np.array([]))
        b1_stats    = band_stats(b1, units, sign_conv)
        n_clipped_b1 = clipped_counts.get(1, 0)

        if b1_stats:
            sc_str = ""
            if sign_conv:
                sc_str = (f"  sub={b1_stats['pct_subsiding']:.1f}%"
                          f"  upl={b1_stats['pct_uplifting']:.1f}%")
            print(f"  {tif.name}")
            print(f"    n={b1_stats['n_valid']:,}  "
                  f"mean={b1_stats['mean']:+.6f}  "
                  f"std={b1_stats['std']:.6f}  "
                  f"[{b1_stats['min']:+.6f}, {b1_stats['max']:+.6f}] "
                  f"{units}{sc_str}  clipped={n_clipped_b1:,}")
            print(f"    -> {out_txt.name}")
        else:
            print(f"  {tif.name}  — NO VALID PIXELS  "
                  f"(clipped={n_clipped_b1:,})")

        summary_lines.append(
            format_summary_row(tif.name, b1, b1_stats, n_clipped_b1)
        )

    out_summary = MASKED_DIR / "00_masked_summary.txt"
    out_summary.write_text("".join(summary_lines), encoding="utf-8")

    print()
    print("=" * 65)
    print(f"Done. Per-TIF stats written to: {MASKED_DIR}")
    print(f"Summary : {out_summary.name}")


if __name__ == "__main__":
    main()
