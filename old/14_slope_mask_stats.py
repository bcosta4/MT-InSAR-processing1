
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

FOLDER = Path(r"E:/Processing/InSAR/results")

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


def raster_stats(path, fallback_nodata=-9999.0):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else fallback_nodata
    valid = data[(data != nodata) & ~np.isnan(data)]
    if len(valid) == 0:
        return None
    return {
        "n_valid_pixels":  int(len(valid)),
        "n_masked_pixels": int(data.size - len(valid)),
        "pct_masked":      round(100 * (data.size - len(valid)) / data.size, 2),
        "mean":            round(float(np.mean(valid)), 4),
        "median":          round(float(np.median(valid)), 4),
        "std":             round(float(np.std(valid)), 4),
        "min":             round(float(np.min(valid)), 4),
        "max":             round(float(np.max(valid)), 4),
        "p5":              round(float(np.percentile(valid,  5)), 4),
        "p25":             round(float(np.percentile(valid, 25)), 4),
        "p75":             round(float(np.percentile(valid, 75)), 4),
        "p95":             round(float(np.percentile(valid, 95)), 4),
        "pct_negative":    round(100 * np.sum(valid < 0) / len(valid), 2),
        "pct_positive":    round(100 * np.sum(valid > 0) / len(valid), 2),
    }


# ── Collect stats ────────────────────────────────────────────────────────────
rows = []

for stem in INSAR_STEMS:
    short = SHORT_NAMES[stem]

    # Baseline (original masked raster, no slope filter)
    base_path = FOLDER / f"{stem}.tif"
    if base_path.exists():
        s = raster_stats(base_path)
        if s:
            rows.append({"raster": short, "slope_threshold": "baseline", **s})
    else:
        print(f"WARNING – baseline not found: {base_path.name}")

    # Slope-masked versions
    for thresh in THRESHOLDS:
        p = FOLDER / f"{stem}_slope{thresh}deg.tif"
        if not p.exists():
            print(f"  Skipping (not found): {p.name}")
            continue
        s = raster_stats(p)
        if s:
            rows.append({"raster": short, "slope_threshold": f"{thresh}deg", **s})

df = pd.DataFrame(rows)

# ── Full stats CSV ────────────────────────────────────────────────────────────
out_full = FOLDER / "slope_mask_stats.csv"
df.to_csv(out_full, index=False)
print(f"Full stats saved → {out_full}")

# ── Summary print ─────────────────────────────────────────────────────────────
print("\n" + "="*90)
print("SLOPE MASKING — SUMMARY TABLE")
print("="*90)
cols = ["raster","slope_threshold","n_valid_pixels","pct_masked",
        "mean","median","std","p5","p95","pct_negative"]
print(df[cols].to_string(index=False))

# ── Delta table vs baseline ───────────────────────────────────────────────────
delta_rows = []
for stem in INSAR_STEMS:
    short = SHORT_NAMES[stem]
    base = df[(df["raster"] == short) & (df["slope_threshold"] == "baseline")]
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
            "delta_mean":      round(t["mean"]        - b["mean"],        4),
            "delta_median":    round(t["median"]      - b["median"],      4),
            "delta_std":       round(t["std"]         - b["std"],         4),
            "delta_p5":        round(t["p5"]          - b["p5"],          4),
            "delta_p95":       round(t["p95"]         - b["p95"],         4),
            "delta_pct_neg":   round(t["pct_negative"]- b["pct_negative"],2),
        })

delta_df = pd.DataFrame(delta_rows)
out_delta = FOLDER / "slope_mask_delta_stats.csv"
delta_df.to_csv(out_delta, index=False)

print("\n" + "="*90)
print("CHANGE RELATIVE TO BASELINE (slope-masked minus baseline)")
print("="*90)
print(delta_df.to_string(index=False))
print(f"\nDelta stats saved → {out_delta}")
print("Done.")
