"""
14_landcover_classification.py

Land cover stratification of DS-InSAR displacement rates.

For each combination of:
  - 3 slope thresholds : 2deg, 4deg, 5deg
  - 3 velocity products : velocity_map, velocity_ddt_map, velocity_gddt_map

The script:
  1. Reprojects + clips Canada 2020 Land Cover (EPSG:3979, 30m) to each
     velocity raster grid (UTM Zone 8N ~27.6m) using nearest-neighbour.
  2. Extracts valid (non-nodata) pixel pairs.
  3. Computes per-class statistics: n, mean, median, std, p5/p25/p75/p95,
     % subsiding, % uplifting, 95% bootstrap CI of mean.
  4. Produces figures:
       Fig A — Violin + strip plot        (one per product × slope)
       Fig B — Horizontal median ± IQR bar chart (one per product × slope)
       Fig C — Heatmap: mean rate by class × slope threshold (one per product)
       Fig D — % subsiding grouped bar   (one per product)
       Fig E — Sample-size bubble plot   (all products × all slopes)
  5. Writes CSV summaries and a plain-text report.
  6. Exports GeoPackage polygon layers with per-class statistics joined as
     attributes (one layer per product × slope).

Pipeline position: after 12_apply_quality_slope_mask.py

Inputs:
  RESULTS_DIR : velocity rasters (*_masked_slope{N}deg.tif)
  LC_PATH     : landcover-2020-classification.tif

Outputs (written to RESULTS_DIR/figures/landcover/):
  14_lc_stats_{product}_{slope}deg.csv
  14_lc_report.txt
  fig14A_violin_{product}_{slope}deg.png      (9 files)
  fig14B_bar_{product}_{slope}deg.png         (9 files)
  fig14C_heatmap_{product}.png                (3 files)
  fig14D_pct_subsiding_{product}.png          (3 files)
  fig14E_samplesize.png                       (1 file)

GeoPackage output:
  E:/Processing/InSAR/landcover_analysis/landcover_stats.gpkg
    layers: {product_key}_slope{N}deg  (9 layers, polygon, UTM Zone 8N)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── PROJ path fix (before rasterio/fiona) ─────────────────────────────────────
_conda = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
_proj  = _conda / "Library" / "share" / "proj"
if _proj.exists():
    os.environ["PROJ_DATA"]    = str(_proj)
    os.environ["PROJ_LIB"]     = str(_proj)
    os.environ["PROJ_NETWORK"] = "OFF"

from osgeo import gdal, osr, ogr
import geopandas as gpd
from shapely.wkb import loads as wkb_loads

gdal.UseExceptions()

# ==============================================================================
# Configuration
# ==============================================================================

RESULTS_DIR = Path(r"E:\Processing\InSAR\results")
LC_PATH     = Path(r"E:\GIS\land_cover\landcover-2020-classification.tif")
OUT_DIR     = RESULTS_DIR / "figures" / "landcover"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GPKG_DIR  = Path(r"E:\Processing\InSAR\landcover_analysis")
GPKG_DIR.mkdir(parents=True, exist_ok=True)
GPKG_PATH = GPKG_DIR / "landcover_stats.gpkg"

NODATA     = -9999.0
MIN_PIXELS = 50  # minimum valid pixels to include a class

# ── Velocity products ─────────────────────────────────────────────────────────
PRODUCTS = {
    "velocity_map": {
        "label": "Velocity",
        "units": "cm yr\u207b\u00b9",
        "files": {
            2: "Network2_ifg_unw_Stack_TC1_velocity_map_masked_slope2deg.tif",
            4: "Network2_ifg_unw_Stack_TC1_velocity_map_masked_slope4deg.tif",
            5: "Network2_ifg_unw_Stack_TC1_velocity_map_masked_slope5deg.tif",
        },
    },
    "velocity_ddt_map": {
        "label": "DDT velocity",
        "units": "cm DDT\u207b\u00b9",
        "files": {
            2: "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked_slope2deg.tif",
            4: "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked_slope4deg.tif",
            5: "Network2_ifg_unw_Stack_TC1_velocity_ddt_map_masked_slope5deg.tif",
        },
    },
    "velocity_gddt_map": {
        "label": "GDDT velocity",
        "units": "cm GDDT\u207b\u00b9",
        "files": {
            2: "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked_slope2deg.tif",
            4: "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked_slope4deg.tif",
            5: "Network2_ifg_unw_Stack_TC1_velocity_gddt_map_masked_slope5deg.tif",
        },
    },
}

SLOPE_THRESHOLDS = [2, 4, 5]

# ── Canada Land Cover 2020 class definitions ──────────────────────────────────
LC_CLASSES = {
    1:  ("Temperate/Sub-polar Needleleaf Forest",        "#003C00"),
    2:  ("Sub-polar Taiga Needleleaf Forest",            "#939B6F"),
    3:  ("Tropical/Sub-tropical Broadleaf Evergreen",   "#006200"),
    4:  ("Tropical/Sub-tropical Broadleaf Deciduous",   "#1DAA04"),
    5:  ("Temperate/Sub-polar Broadleaf Deciduous",     "#138B04"),
    6:  ("Mixed Forest",                                 "#5B742A"),
    7:  ("Tropical/Sub-tropical Shrubland",             "#B29D2A"),
    8:  ("Temperate/Sub-polar Shrubland",               "#B29D32"),
    9:  ("Tropical/Sub-tropical Grassland",             "#E7DB5D"),
    10: ("Temperate/Sub-polar Grassland",               "#E0CE89"),
    11: ("Sub-polar/Polar Shrubland-Lichen-Moss",       "#9B7453"),
    12: ("Sub-polar/Polar Grassland-Lichen-Moss",       "#B9D38E"),
    13: ("Sub-polar/Polar Barren-Lichen-Moss",          "#3F896F"),
    14: ("Wetland",                                      "#6AA289"),
    15: ("Cropland",                                     "#F0B86C"),
    16: ("Barren Land",                                  "#A7AAAD"),
    17: ("Urban",                                        "#DB2025"),
    18: ("Water",                                        "#4B6FA2"),
    19: ("Snow/Ice",                                     "#FEF9FE"),
}

LC_SHORT = {
    1:  "NL Forest",
    2:  "Taiga NL Forest",
    3:  "BL Evgr. Forest",
    4:  "BL Decid. Forest",
    5:  "Temp BL Decid.",
    6:  "Mixed Forest",
    7:  "Trop Shrubland",
    8:  "Temp Shrubland",
    9:  "Trop Grassland",
    10: "Temp Grassland",
    11: "Polar Shrub-Lichen",
    12: "Polar Grass-Lichen",
    13: "Polar Barren",
    14: "Wetland",
    15: "Cropland",
    16: "Barren Land",
    17: "Urban",
    18: "Water",
    19: "Snow/Ice",
}

# ==============================================================================
# Helper: read raster file
# ==============================================================================

def read_raster(path: Path) -> tuple[np.ndarray, tuple, str, float]:
    ds   = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f"Failed to open raster: {path}")
    band = ds.GetRasterBand(1)
    arr  = band.ReadAsArray().astype(np.float32)
    gt   = ds.GetGeoTransform()
    prj  = ds.GetProjection()
    nd   = band.GetNoDataValue()
    ds   = None
    return arr, gt, prj, nd if nd is not None else NODATA

# ==============================================================================
# Helper: reproject LC to velocity grid (nearest-neighbour, categorical)
# ==============================================================================

def reproject_lc_to_velocity(lc_path: Path, vel_path: Path) -> np.ndarray:
    vel_ds        = gdal.Open(str(vel_path))
    dst_crs       = vel_ds.GetProjection()
    dst_transform = vel_ds.GetGeoTransform()
    dst_shape     = (vel_ds.RasterYSize, vel_ds.RasterXSize)

    mem_drv = gdal.GetDriverByName("MEM")
    dst_ds  = mem_drv.Create("", dst_shape[1], dst_shape[0], 1, gdal.GDT_Byte)
    dst_ds.SetGeoTransform(dst_transform)
    dst_ds.SetProjection(dst_crs)

    lc_ds = gdal.Open(str(lc_path))
    gdal.ReprojectImage(lc_ds, dst_ds, lc_ds.GetProjection(), dst_crs,
                        gdal.GRA_NearestNeighbour)

    lc_aligned = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    lc_ds = None
    dst_ds = None
    return lc_aligned

# ==============================================================================
# Helper: per-class statistics with bootstrap CI
# ==============================================================================

def extract_stats(vel_arr: np.ndarray, lc_arr: np.ndarray,
                  units: str) -> tuple[pd.DataFrame, dict]:
    valid_mask = (
        (vel_arr != NODATA) & np.isfinite(vel_arr) &
        (lc_arr > 0) & (lc_arr <= 19)
    )
    v   = vel_arr[valid_mask].astype(np.float32)
    c   = lc_arr[valid_mask]
    rows = []
    pixel_data = {}
    rng = np.random.default_rng(42)

    for cls_id in range(1, 20):
        vals = v[c == cls_id]
        n    = len(vals)
        if n < MIN_PIXELS:
            continue
        pixel_data[cls_id] = vals
        boot = np.array([
            np.mean(rng.choice(vals, size=n, replace=True))
            for _ in range(500)
        ])
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
        rows.append({
            "class_id":    cls_id,
            "class_name":  LC_CLASSES[cls_id][0],
            "class_short": LC_SHORT[cls_id],
            "color":       LC_CLASSES[cls_id][1],
            "units":       units,
            "n":           n,
            "mean":        float(np.mean(vals)),
            "median":      float(np.median(vals)),
            "std":         float(np.std(vals)),
            "p05":         float(np.percentile(vals,  5)),
            "p25":         float(np.percentile(vals, 25)),
            "p75":         float(np.percentile(vals, 75)),
            "p95":         float(np.percentile(vals, 95)),
            "ci_lo":       float(ci_lo),
            "ci_hi":       float(ci_hi),
            "pct_sub":     float(np.sum(vals < 0) / n * 100),
            "pct_upl":     float(np.sum(vals > 0) / n * 100),
        })
    return pd.DataFrame(rows), pixel_data

# ==============================================================================
# Helper: export GeoPackage polygon layer
# ==============================================================================

def export_gpkg_layer(lc_aligned: np.ndarray, stats_df: pd.DataFrame,
                      vel_path: Path, product_key: str, slope: int) -> None:
    """
    Polygonize the aligned LC raster, dissolve by class_id, join statistics,
    and append as a new layer to GPKG_PATH.
    Layer name: {product_key}_slope{slope}deg
    """
    vel_ds = gdal.Open(str(vel_path))
    gt     = vel_ds.GetGeoTransform()
    prj    = vel_ds.GetProjection()
    nrows, ncols = lc_aligned.shape
    vel_ds = None

    # write LC array to an in-memory raster for polygonize
    mem_drv = gdal.GetDriverByName("MEM")
    mem_ds  = mem_drv.Create("", ncols, nrows, 1, gdal.GDT_Int32)
    mem_ds.SetGeoTransform(gt)
    mem_ds.SetProjection(prj)
    band = mem_ds.GetRasterBand(1)
    band.WriteArray(lc_aligned.astype(np.int32))
    band.SetNoDataValue(0)

    # polygonize into an in-memory OGR layer
    ogr_mem_drv = ogr.GetDriverByName("Memory")
    ogr_ds      = ogr_mem_drv.CreateDataSource("memdata")
    srs         = osr.SpatialReference()
    srs.ImportFromWkt(prj)
    ogr_lyr = ogr_ds.CreateLayer("lc_poly", srs=srs,
                                  geom_type=ogr.wkbMultiPolygon)
    ogr_lyr.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))
    gdal.Polygonize(band, None, ogr_lyr, 0, [], callback=None)
    mem_ds = None

    # read OGR features into Python list
    records = []
    ogr_lyr.ResetReading()
    feat = ogr_lyr.GetNextFeature()
    while feat:
        geom = feat.GetGeometryRef()
        if geom is not None:
            records.append({
                "class_id": feat.GetField("class_id"),
                "geometry": wkb_loads(bytes(geom.ExportToWkb())),
            })
        feat = ogr_lyr.GetNextFeature()
    ogr_ds = None

    if not records:
        print(f"   WARNING: no polygons generated for "
              f"{product_key} slope{slope}deg")
        return

    gdf = gpd.GeoDataFrame(records, crs=prj)

    # dissolve so each class_id = one MultiPolygon
    gdf = gdf.dissolve(by="class_id").reset_index()

    # join per-class statistics
    stat_cols = ["class_id", "class_name", "class_short", "color", "units",
                 "n", "mean", "median", "std",
                 "p05", "p25", "p75", "p95",
                 "ci_lo", "ci_hi", "pct_sub", "pct_upl"]
    gdf = gdf.merge(stats_df[stat_cols], on="class_id", how="left")
    gdf["slope_deg"] = slope
    gdf["product"]   = product_key

    layer_name = f"{product_key}_slope{slope}deg"

    # append mode: if GPKG already exists, add layer; otherwise create fresh
    write_mode = "a" if GPKG_PATH.exists() else "w"
    gdf.to_file(str(GPKG_PATH), layer=layer_name, driver="GPKG",
                mode=write_mode)
    print(f"   GPKG layer -> {layer_name}  ({len(gdf)} classes)")

# ==============================================================================
# Figure A — Violin + strip plot
# ==============================================================================

def fig_violin(pixel_data: dict, stats_df: pd.DataFrame,
               slope: int, units: str, label: str, out_path: Path):
    df      = stats_df.sort_values("median")
    classes = df["class_id"].tolist()
    shorts  = [LC_SHORT[c] for c in classes]
    colors  = [LC_CLASSES[c][1] for c in classes]

    fig, ax = plt.subplots(figsize=(max(11, len(classes) * 0.9), 5.5))
    rng = np.random.default_rng(0)

    for i, (cls_id, col) in enumerate(zip(classes, colors)):
        vals = pixel_data.get(cls_id, np.array([]))
        if len(vals) < MIN_PIXELS:
            continue
        parts = ax.violinplot(vals, positions=[i], widths=0.65,
                              showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(col); pc.set_edgecolor("#333")
            pc.set_alpha(0.55);    pc.set_linewidth(0.6)

        row = df[df.class_id == cls_id].iloc[0]
        ax.plot([i, i], [row.p25, row.p75], color="#222", lw=2.2, zorder=4)
        ax.scatter(i, row["median"], color="white", edgecolors="#222",
                   s=22, zorder=5, linewidths=0.8)

        n_strip = min(len(vals), 400)
        sub     = rng.choice(vals, n_strip, replace=False)
        jitter  = rng.uniform(-0.18, 0.18, n_strip)
        ax.scatter(i + jitter, sub, color=col, alpha=0.18, s=2.5,
                   linewidths=0, zorder=2)

    ax.axhline(0, color="#888", lw=0.7, ls="--", alpha=0.6)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(shorts, rotation=42, ha="right", fontsize=7)
    ax.set_ylabel(f"Displacement rate ({units})")
    ax.set_title(f"{label} — land cover stratification | slope \u2264{slope}\u00b0",
                 fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)

    legend_patch = mpatches.Patch(color="#aaa",
        label="\u25cf median | box = IQR | strip = sampled pixels")
    ax.legend(handles=[legend_patch], frameon=False,
              loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# Figure B — Horizontal bar: median ± IQR
# ==============================================================================

def fig_bar_median(stats_df: pd.DataFrame, slope: int,
                   units: str, label: str, out_path: Path):
    df  = stats_df.sort_values("median")
    y   = np.arange(len(df))
    col = [LC_CLASSES[c][1] for c in df.class_id]

    fig, ax = plt.subplots(figsize=(7.5, max(4.5, len(df) * 0.40)))

    ax.barh(y, df["median"], color=col, edgecolor="#333", linewidth=0.5,
            height=0.60, alpha=0.82, zorder=3, label="Median")
    ax.errorbar(df["median"], y,
                xerr=[df["median"] - df["p25"], df["p75"] - df["median"]],
                fmt="none", color="#222", lw=1.1, capsize=3, capthick=0.9,
                zorder=4, label="IQR")
    ax.errorbar(df["mean"], y,
                xerr=[df["mean"] - df["ci_lo"], df["ci_hi"] - df["mean"]],
                fmt="D", color="#c0392b", ms=3.5, lw=0.8, capsize=2,
                zorder=5, label="Mean \u00b1 95% CI")

    ax.axvline(0, color="#888", lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["class_short"], fontsize=7.5)
    ax.set_xlabel(f"Displacement rate ({units})")
    ax.set_title(f"{label} — median \u00b1 IQR by land cover | slope \u2264{slope}\u00b0",
                 fontweight="bold", pad=6)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    xlim_r = ax.get_xlim()[1]
    for i, row in enumerate(df.itertuples()):
        ax.text(xlim_r * 1.01, i, f"n={row.n:,}",
                va="center", fontsize=5.5, color="#555")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# Figure C — Heatmap: mean rate × (class × slope threshold)
# ==============================================================================

def fig_heatmap(all_stats: dict, units: str, label: str, out_path: Path):
    all_classes = sorted({cls for df in all_stats.values()
                          for cls in df.class_id})
    col_labels  = [f"\u2264{s}\u00b0" for s in SLOPE_THRESHOLDS]
    mat = np.full((len(all_classes), len(SLOPE_THRESHOLDS)), np.nan)

    for j, slope in enumerate(SLOPE_THRESHOLDS):
        df = all_stats.get(slope, pd.DataFrame())
        for i, cls_id in enumerate(all_classes):
            row = df[df.class_id == cls_id]
            if not row.empty:
                mat[i, j] = row.iloc[0]["mean"]

    row_labels = [LC_SHORT[c] for c in all_classes]
    vabs = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, ax = plt.subplots(figsize=(5.5, max(5, len(all_classes) * 0.42)))
    im = ax.imshow(mat, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_xlabel("Slope mask threshold")
    ax.set_title(f"{label} — mean rate by class \u00d7 slope threshold",
                 fontweight="bold", pad=6)

    for i in range(len(all_classes)):
        for j in range(len(SLOPE_THRESHOLDS)):
            val = mat[i, j]
            if not np.isnan(val):
                fc = "white" if abs(val) > vabs * 0.6 else "#111"
                ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                        fontsize=6.5, color=fc)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(units, fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# Figure D — % subsiding grouped bar
# ==============================================================================

def fig_pct_subsiding(all_stats: dict, label: str, out_path: Path):
    all_classes  = sorted({cls for df in all_stats.values()
                            for cls in df.class_id})
    shorts       = [LC_SHORT[c] for c in all_classes]
    x            = np.arange(len(all_classes))
    width        = 0.22
    slope_colors = ["#2166ac", "#4dac26", "#d01c8b"]

    fig, ax = plt.subplots(figsize=(max(10, len(all_classes) * 0.9), 4.5))

    for j, (slope, sc) in enumerate(zip(SLOPE_THRESHOLDS, slope_colors)):
        df   = all_stats.get(slope, pd.DataFrame())
        vals = []
        for cls_id in all_classes:
            row = df[df.class_id == cls_id]
            vals.append(row.iloc[0]["pct_sub"] if not row.empty else np.nan)
        offset = (j - len(SLOPE_THRESHOLDS) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9, color=sc, alpha=0.78,
               edgecolor="#333", linewidth=0.4, label=f"\u2264{slope}\u00b0")

    ax.axhline(50, color="#888", lw=0.7, ls=":", alpha=0.7,
               label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(shorts, rotation=42, ha="right", fontsize=7)
    ax.set_ylabel("% subsiding pixels")
    ax.set_ylim(0, 108)
    ax.set_title(f"{label} — % subsiding pixels by land cover class",
                 fontweight="bold", pad=6)
    ax.legend(title="Slope threshold", frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# Figure E — Sample-size bubble plot
# ==============================================================================

def fig_samplesize(all_product_stats: dict, out_path: Path):
    products    = [p for p in PRODUCTS if all_product_stats.get(p)]
    prod_labels = [PRODUCTS[p]["label"] for p in products]

    fig, axes = plt.subplots(1, len(SLOPE_THRESHOLDS),
                             figsize=(5.5 * len(SLOPE_THRESHOLDS), 4.5),
                             sharey=True)
    if len(SLOPE_THRESHOLDS) == 1:
        axes = [axes]

    for ax, slope in zip(axes, SLOPE_THRESHOLDS):
        all_classes = sorted({
            cls
            for p in products
            for cls in all_product_stats[p].get(
                slope, pd.DataFrame()
            ).get("class_id", pd.Series([], dtype=int))
        })
        if not all_classes:
            continue
        shorts = [LC_SHORT[c] for c in all_classes]
        x_map  = {c: i for i, c in enumerate(all_classes)}
        y_map  = {p: i for i, p in enumerate(products)}

        for p in products:
            df = all_product_stats[p].get(slope, pd.DataFrame())
            if df.empty:
                continue
            for _, row in df.iterrows():
                xi = x_map.get(int(row.class_id))
                yi = y_map.get(p)
                if xi is None or yi is None:
                    continue
                size  = max(20, np.log10(max(row.n, 1)) * 130)
                color = LC_CLASSES[int(row.class_id)][1]
                ax.scatter(xi, yi, s=size, color=color, alpha=0.78,
                           edgecolors="#333", linewidths=0.5)
                lbl = f"{int(row.n)//1000}k" if row.n >= 1000 else str(int(row.n))
                ax.text(xi, yi, lbl, ha="center", va="center",
                        fontsize=5, color="#111")

        ax.set_xticks(range(len(all_classes)))
        ax.set_xticklabels(shorts, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(products)))
        ax.set_yticklabels(prod_labels, fontsize=8)
        ax.set_title(f"Slope \u2264{slope}\u00b0", fontweight="bold", pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(-0.7, len(all_classes) - 0.3)

    fig.suptitle(
        "Valid pixel count (bubble size = log\u2081\u2080 n) by class \u00d7 product",
        fontweight="bold", y=1.01, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("14_landcover_classification.py")
    print("=" * 60)
    print(f"LC raster   : {LC_PATH}")
    print(f"Results dir : {RESULTS_DIR}")
    print(f"Output dir  : {OUT_DIR}")
    print(f"GPKG output : {GPKG_PATH}")
    print("=" * 60)

    report = [
        "14_landcover_classification report\n",
        f"Timestamp   : {datetime.now().isoformat(timespec='seconds')}\n",
        f"LC raster   : {LC_PATH}\n",
        f"Results dir : {RESULTS_DIR}\n\n",
    ]

    # remove stale GPKG so layers are not duplicated on re-runs
    if GPKG_PATH.exists():
        GPKG_PATH.unlink()
        print(f"Removed existing GPKG: {GPKG_PATH.name}")

    all_product_stats = {p: {} for p in PRODUCTS}

    for product_key, prod_cfg in PRODUCTS.items():
        label = prod_cfg["label"]
        units = prod_cfg["units"]
        print(f"\n{'='*60}")
        print(f"Product: {label} ({units})")
        print(f"{'='*60}")
        report += [f"\n{'='*60}\n", f"Product: {label} ({units})\n",
                   f"{'='*60}\n"]

        for slope in SLOPE_THRESHOLDS:
            fname    = prod_cfg["files"][slope]
            vel_path = RESULTS_DIR / fname

            if not vel_path.exists():
                print(f"  SKIP (not found): {fname}")
                report.append(f"  SKIP slope {slope}deg: {fname} not found\n")
                continue

            print(f"\n  Slope \u2264{slope}\u00b0 -> {fname}")

            # Load velocity raster
            vel_arr, _gt, _prj, vel_nd = read_raster(vel_path)
            vel_arr[vel_arr == vel_nd]    = NODATA
            vel_arr[~np.isfinite(vel_arr)] = NODATA

            # Reproject LC -> velocity grid
            print(f"  Reprojecting LC to velocity grid ...", end=" ", flush=True)
            lc_aligned = reproject_lc_to_velocity(LC_PATH, vel_path)
            print("done")

            # Statistics
            print(f"  Computing statistics ...", end=" ", flush=True)
            stats_df, pixel_data = extract_stats(vel_arr, lc_aligned, units)
            print(f"done ({len(stats_df)} classes with \u2265{MIN_PIXELS} px)")

            if stats_df.empty:
                print("  WARNING: no classes met minimum pixel threshold.")
                continue

            all_product_stats[product_key][slope] = stats_df

            # Save CSV
            csv_path = OUT_DIR / f"14_lc_stats_{product_key}_slope{slope}deg.csv"
            stats_df.to_csv(csv_path, index=False, float_format="%.5f")
            print(f"  CSV -> {csv_path.name}")

            # Export GeoPackage polygon layer
            print(f"  Exporting GPKG layer ...", end=" ", flush=True)
            export_gpkg_layer(lc_aligned, stats_df, vel_path,
                              product_key, slope)

            # Report block
            report.append(f"\n  Slope \u2264{slope}\u00b0\n")
            for _, row in stats_df.sort_values("median").iterrows():
                report.append(
                    f"  {row.class_short:<22} n={row.n:>8,} "
                    f"mean={row['mean']:+.4f} median={row['median']:+.4f} "
                    f"std={row['std']:.4f} "
                    f"sub={row.pct_sub:.1f}% upl={row.pct_upl:.1f}%\n"
                )

            # Fig A — Violin
            path_a = OUT_DIR / f"fig14A_violin_{product_key}_slope{slope}deg.png"
            print(f"  Fig A (violin) ...", end=" ", flush=True)
            fig_violin(pixel_data, stats_df, slope, units, label, path_a)
            print("saved")

            # Fig B — Bar
            path_b = OUT_DIR / f"fig14B_bar_{product_key}_slope{slope}deg.png"
            print(f"  Fig B (bar) ...", end=" ", flush=True)
            fig_bar_median(stats_df, slope, units, label, path_b)
            print("saved")

        # Figs C and D — per product, across all slopes
        prod_stats = all_product_stats[product_key]
        if prod_stats:
            path_c = OUT_DIR / f"fig14C_heatmap_{product_key}.png"
            print(f"\n  Fig C (heatmap) ...", end=" ", flush=True)
            fig_heatmap(prod_stats, units, label, path_c)
            print("saved")

            path_d = OUT_DIR / f"fig14D_pct_subsiding_{product_key}.png"
            print(f"  Fig D (%subsiding) ...", end=" ", flush=True)
            fig_pct_subsiding(prod_stats, label, path_d)
            print("saved")

    # Fig E — sample size bubble
    if any(all_product_stats[p] for p in PRODUCTS):
        path_e = OUT_DIR / "fig14E_samplesize.png"
        print(f"\nFig E (sample size bubble) ...", end=" ", flush=True)
        fig_samplesize(all_product_stats, path_e)
        print("saved")

    # Write report
    rep_path = OUT_DIR / "14_lc_report.txt"
    rep_path.write_text("".join(report), encoding="utf-8")
    print(f"\nReport -> {rep_path.name}")
    print("=" * 60)
    print(f"Done.")
    print(f"  Figures/CSV : {OUT_DIR}")
    print(f"  GeoPackage  : {GPKG_PATH}")

if __name__ == "__main__":
    main()