"""
08_plot_results.py
Diagnostic figures for DS-InSAR deformation results.

Per-network figures:
  1. Velocity map (cm/year) -- primary deformation product
  2. Time series plot -- displacement vs time for representative pixels
  3. Residuals map -- inversion quality
  4. Coherence vs velocity scatter -- validates coherence weighting

Cross-network comparison (Network1 vs Network2):
  5. Side-by-side velocity maps
  6. Difference map (N1 - N2)
  7. Velocity distribution histograms

Outputs (written to PLOTS_DIR):
  <stem>_velocity_map.png
  <stem>_timeseries_plot.png
  <stem>_residuals_map.png
  <stem>_coh_vs_velocity.png
  comparison_velocity_maps.png
  comparison_difference_map.png
  comparison_histograms.png

Units: velocity in cm/year, displacement in cm.
Reads GeoTIFF outputs from 07_export_geotiff.py.
"""

import os
import csv
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Fix PROJ conflict ──────────────────────────────────────────────────────────
def _fix_proj():
    try:
        from pyproj.datadir import get_data_dir
        proj_dir = str(get_data_dir())
        os.environ["PROJ_DATA"]    = proj_dir
        os.environ["PROJ_LIB"]     = proj_dir
        os.environ["PROJ_NETWORK"] = "OFF"
        return True
    except Exception:
        pass
    _conda = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\anaconda3"))
    _p     = _conda / "Library" / "share" / "proj"
    if _p.exists():
        os.environ["PROJ_DATA"]    = str(_p)
        os.environ["PROJ_LIB"]     = str(_p)
        os.environ["PROJ_NETWORK"] = "OFF"
        return True
    return False

_fix_proj()

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR  = Path(r"C:\bcosta\DS-InSAR\tc")
PLOTS_DIR = Path(r"C:\bcosta\DS-InSAR\tc\plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
VEL_CLIP_PCT = 2
N_TS_PIXELS  = 5
DPI          = 150
NODATA       = -9999.0
DATE_FMT     = "%Y%m%d"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_tif(tif_path, band=1):
    import rasterio
    with rasterio.open(tif_path) as src:
        arr    = src.read(band).astype(np.float32)
        nodata = src.nodata
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, extent


def load_tif_all_bands(tif_path):
    import rasterio
    with rasterio.open(tif_path) as src:
        arr    = src.read().astype(np.float32)
        nodata = src.nodata
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, extent


def load_dates(dates_path):
    dates = []
    days  = []
    with open(dates_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            dates.append(datetime.strptime(row["date"].strip(), DATE_FMT))
            for key in ("days_since_first", "days", "day_offset", "day"):
                if key in row:
                    days.append(int(row[key]))
                    break
            else:
                days.append(i * 12)
    return dates, np.array(days, dtype=int)


def vel_clim(vel_2d, pct=VEL_CLIP_PCT):
    """
    Symmetric color limits clipped at pct/100-pct percentiles.
    Floor of 0.5 cm/year prevents a degenerate axis on near-zero scenes.
    """
    valid = vel_2d[np.isfinite(vel_2d)]
    if len(valid) == 0:
        return -5.0, 5.0
    vlo     = float(np.percentile(valid, pct))
    vhi     = float(np.percentile(valid, 100 - pct))
    abs_max = max(abs(vlo), abs(vhi), 0.5)
    return -abs_max, abs_max


def select_representative_pixels(vel_1d, n=N_TS_PIXELS):
    """Select pixels spanning the velocity range for time series plotting."""
    finite     = np.isfinite(vel_1d)
    idx_finite = np.where(finite)[0]
    vel_finite = vel_1d[finite]

    if len(vel_finite) == 0:
        return []

    sorted_idx = np.argsort(vel_finite)
    n_half     = max(1, n // 2)
    selected   = []
    used       = set()

    for i in range(len(sorted_idx)):
        if len([s for s in selected if "Subsidence" in s[0]]) >= n_half:
            break
        px = int(idx_finite[sorted_idx[i]])
        if px not in used:
            v = float(vel_finite[sorted_idx[i]])
            selected.append((f"Subsidence ({v:.2f} cm/yr)", px))
            used.add(px)

    for i in range(len(sorted_idx) - 1, -1, -1):
        if len([s for s in selected if "Uplift" in s[0]]) >= n_half:
            break
        px = int(idx_finite[sorted_idx[i]])
        if px not in used:
            v = float(vel_finite[sorted_idx[i]])
            selected.append((f"Uplift ({v:.2f} cm/yr)", px))
            used.add(px)

    med_i  = len(sorted_idx) // 2
    med_px = int(idx_finite[sorted_idx[med_i]])
    if med_px not in used:
        med_v = float(vel_finite[sorted_idx[med_i]])
        selected.append((f"Median ({med_v:.2f} cm/yr)", med_px))

    return selected


# ── Figure 1 — Velocity map ────────────────────────────────────────────────────
def plot_velocity_map(stem, vel_2d, coh_2d, extent):
    vmin, vmax = vel_clim(vel_2d)
    cmap       = plt.cm.RdBu_r   # blue = uplift, red = subsidence

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"width_ratios": [1, 1]})

    ax = axes[0]
    im = ax.imshow(vel_2d, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=extent, origin="upper", interpolation="none")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("LOS velocity (cm/year)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f"{stem}\nMean LOS Velocity", fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude / Easting", fontsize=8)
    ax.set_ylabel("Latitude / Northing", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.02, "Blue = uplift | Red = subsidence",
            transform=ax.transAxes, fontsize=7,
            color="black", backgroundcolor="white",
            verticalalignment="bottom")

    ax2 = axes[1]
    im2 = ax2.imshow(coh_2d, cmap="gray", vmin=0, vmax=1,
                     extent=extent, origin="upper", interpolation="none")
    cb2 = fig.colorbar(im2, ax=ax2, shrink=0.7, pad=0.02)
    cb2.set_label("Mean coherence", fontsize=9)
    cb2.ax.tick_params(labelsize=8)
    ax2.set_title("Mean Coherence\n(DS candidate pixels)",
                  fontsize=10, fontweight="bold")
    ax2.set_xlabel("Longitude / Easting", fontsize=8)
    ax2.tick_params(labelsize=7)

    valid_vel   = vel_2d[np.isfinite(vel_2d)]
    title_range = (f"[{float(valid_vel.min()):.2f} to "
                   f"{float(valid_vel.max()):.2f} cm/yr]"
                   if len(valid_vel) > 0 else "[no valid data]")
    plt.suptitle(f"DS-InSAR Results — {stem}  {title_range}",
                 fontsize=11, y=1.01)
    plt.tight_layout()

    out = PLOTS_DIR / f"{stem}_velocity_map.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out.name}")


# ── Figure 2 — Time series ─────────────────────────────────────────────────────
def plot_timeseries(stem, ts_vec, vel_1d, dates, days):
    selected = select_representative_pixels(vel_1d)
    if not selected:
        print(f"  WARNING: no valid pixels for time series plot")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(selected)))

    for (label, px_idx), color in zip(selected, colors):
        ts = ts_vec[:, px_idx]
        ax.plot(dates, ts, "o-", color=color, label=label,
                markersize=4, linewidth=1.5, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Acquisition date", fontsize=10)
    ax.set_ylabel("Cumulative displacement (cm)", fontsize=10)
    ax.set_title(f"{stem} — Displacement Time Series\n"
                 f"Representative pixels  (negative = subsidence)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()

    ylim  = ax.get_ylim()
    years = sorted(set(d.year for d in dates))
    for yr in years[1:]:
        ax.axvline(datetime(yr, 1, 1), color="gray",
                   linewidth=0.6, linestyle=":", alpha=0.6)
        ax.text(datetime(yr, 1, 15), ylim[0],
                str(yr), fontsize=7, color="gray", va="bottom")

    out = PLOTS_DIR / f"{stem}_timeseries_plot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out.name}")


# ── Figure 3 — Residuals map ───────────────────────────────────────────────────
def plot_residuals_map(stem, res_2d, extent):
    valid = res_2d[np.isfinite(res_2d)]
    if len(valid) == 0:
        print(f"  WARNING: no valid residuals to plot")
        return

    vmax = float(np.percentile(valid, 95))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(res_2d, cmap="hot_r", vmin=0, vmax=vmax,
                   extent=extent, origin="upper", interpolation="none")
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("Mean absolute residual (rad)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(f"{stem}\nInversion Residuals — Quality Map\n"
                 f"(lower = better constrained pixel)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Longitude / Easting", fontsize=8)
    ax.set_ylabel("Latitude / Northing", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.02, 0.98,
            f"Mean: {float(valid.mean()):.3f} rad\n"
            f"95th pct: {float(np.percentile(valid, 95)):.3f} rad",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="top", backgroundcolor="white")
    plt.tight_layout()

    out = PLOTS_DIR / f"{stem}_residuals_map.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out.name}")


# ── Figure 4 — Coherence vs velocity scatter ──────────────────────────────────
def plot_coh_vs_velocity(stem, vel_1d, coh_1d):
    finite = np.isfinite(vel_1d) & np.isfinite(coh_1d)
    vel_f  = vel_1d[finite]
    coh_f  = coh_1d[finite]

    if len(vel_f) == 0:
        print(f"  WARNING: no valid pixels for coherence scatter")
        return

    if len(vel_f) > 50000:
        idx   = np.random.default_rng(42).choice(len(vel_f), 50000, replace=False)
        vel_f = vel_f[idx]
        coh_f = coh_f[idx]

    vmin, vmax = vel_clim(vel_f.reshape(1, -1).squeeze())

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(coh_f, vel_f, c=vel_f, cmap="RdBu_r",
                    vmin=vmin, vmax=vmax,
                    s=0.5, alpha=0.4, rasterized=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label("Velocity (cm/year)", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Mean coherence", fontsize=10)
    ax.set_ylabel("LOS velocity (cm/year)", fontsize=10)
    ax.set_title(f"{stem}\nCoherence vs Velocity\n"
                 f"({len(vel_f):,} pixels shown)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    bins     = np.arange(0.4, 1.01, 0.05)
    bin_mid  = (bins[:-1] + bins[1:]) / 2
    bin_mean = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(len(bins) - 1):
            mask = (coh_f >= bins[i]) & (coh_f < bins[i + 1])
            bin_mean.append(float(np.nanmean(vel_f[mask]))
                            if np.sum(mask) > 0 else np.nan)

    bin_mean   = np.array(bin_mean)
    valid_bins = np.isfinite(bin_mean)
    if np.any(valid_bins):
        ax.plot(bin_mid[valid_bins], bin_mean[valid_bins],
                "k^-", markersize=5, linewidth=1.5,
                label="Bin mean", zorder=5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / f"{stem}_coh_vs_velocity.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out.name}")


# ── Figures 5/6/7 — Cross-network comparison ──────────────────────────────────
def plot_comparison(stems, vel_maps, extents):
    if len(stems) < 2:
        print("  Only one network — skipping comparison plots")
        return

    s1, s2 = stems[0], stems[1]
    v1, v2 = vel_maps[0], vel_maps[1]
    e1      = extents[0]

    all_valid = np.concatenate([v1[np.isfinite(v1)], v2[np.isfinite(v2)]])
    if len(all_valid) == 0:
        print("  WARNING: no valid data for comparison plots")
        return

    abs_max    = float(np.percentile(np.abs(all_valid), 98))
    vmin, vmax = -abs_max, abs_max
    cmap       = plt.cm.RdBu_r

    # Figure 5 — side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, stem, vel, ext in zip(axes, stems, vel_maps, extents):
        im = ax.imshow(vel, cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=ext, origin="upper", interpolation="none")
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("cm/year", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(stem, fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude / Easting", fontsize=8)
        ax.set_ylabel("Latitude / Northing", fontsize=8)
        ax.tick_params(labelsize=7)
    plt.suptitle("Network Comparison — Mean LOS Velocity (cm/year)\n"
                 "Blue = uplift | Red = subsidence",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out1 = PLOTS_DIR / "comparison_velocity_maps.png"
    fig.savefig(out1, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out1.name}")

    # Figure 6 — difference map
    if v1.shape == v2.shape:
        diff       = v1 - v2
        diff[~np.isfinite(v1) | ~np.isfinite(v2)] = np.nan
        valid_diff = diff[np.isfinite(diff)]
        d_abs      = float(np.percentile(np.abs(valid_diff), 98)) if len(valid_diff) else 1.0

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(diff, cmap="PiYG", vmin=-d_abs, vmax=d_abs,
                       extent=e1, origin="upper", interpolation="none")
        cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label("Velocity difference (cm/year)", fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.set_title(f"Velocity Difference: {s1} − {s2}\n"
                     f"Mean diff: {float(np.nanmean(diff)):.3f} cm/yr  "
                     f"RMSD: {float(np.sqrt(np.nanmean(diff**2))):.3f} cm/yr",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Longitude / Easting", fontsize=8)
        ax.set_ylabel("Latitude / Northing", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.tight_layout()
        out2 = PLOTS_DIR / "comparison_difference_map.png"
        fig.savefig(out2, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Written: {out2.name}")
    else:
        print(f"  WARNING: velocity maps differ in shape "
              f"({v1.shape} vs {v2.shape}) — skipping difference map")

    # Figure 7 — histogram comparison
    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#2166ac", "#d6604d"]
    for stem, vel, color in zip(stems, vel_maps, colors):
        valid = vel[np.isfinite(vel)]
        if len(valid) == 0:
            continue
        ax.hist(valid, bins=150, range=(vmin, vmax),
                color=color, alpha=0.55, label=stem,
                density=True, edgecolor="none")
        ax.axvline(float(np.nanmean(valid)), color=color,
                   linewidth=1.5, linestyle="--",
                   label=f"{stem} mean = {float(np.nanmean(valid)):.2f} cm/yr")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.6)
    ax.set_xlabel("LOS velocity (cm/year)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Velocity Distribution Comparison\nNetwork 1 vs Network 2",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    out3 = PLOTS_DIR / "comparison_histograms.png"
    fig.savefig(out3, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Written: {out3.name}")


# ── Per-network processing ─────────────────────────────────────────────────────
def process_network(stem):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    vel_tif     = WORK_DIR / f"{stem}_velocity_map.tif"
    res_tif     = WORK_DIR / f"{stem}_residuals_map.tif"
    ts_tif      = WORK_DIR / f"{stem}_timeseries_stack.tif"
    coh_tif     = WORK_DIR / f"{stem}_mean_coherence.tif"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    dates_path  = WORK_DIR / f"{stem}_date_list.csv"

    for p in [vel_tif, coh_tif, coords_path, dates_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} — run scripts 06 and 07 first.")
            return None, None, None

    vel_2d, extent = load_tif(vel_tif)
    coh_2d, _      = load_tif(coh_tif)
    pixel_coords   = np.load(coords_path)
    dates, days    = load_dates(dates_path)

    vel_1d = vel_2d[pixel_coords[:, 0], pixel_coords[:, 1]]
    coh_1d = coh_2d[pixel_coords[:, 0], pixel_coords[:, 1]]

    n_valid = int(np.sum(np.isfinite(vel_2d)))
    print(f"  Valid pixels   : {n_valid:,}")
    print(f"  Velocity range : {float(np.nanmin(vel_2d)):.3f} – "
          f"{float(np.nanmax(vel_2d)):.3f} cm/year")

    plot_velocity_map(stem, vel_2d, coh_2d, extent)

    if res_tif.exists():
        res_2d, _ = load_tif(res_tif)
        plot_residuals_map(stem, res_2d, extent)
    else:
        print(f"  Skipping residuals plot — {res_tif.name} not found")

    if ts_tif.exists():
        ts_stack, _ = load_tif_all_bands(ts_tif)
        n_ts_dates  = ts_stack.shape[0]
        if n_ts_dates != len(dates):
            n_use    = min(n_ts_dates, len(dates))
            ts_stack = ts_stack[:n_use]
            dates    = dates[:n_use]
            days     = days[:n_use]
        ts_vec = ts_stack[:, pixel_coords[:, 0], pixel_coords[:, 1]]
        plot_timeseries(stem, ts_vec, vel_1d, dates, days)
    else:
        print(f"  Skipping time series plot — {ts_tif.name} not found")

    plot_coh_vs_velocity(stem, vel_1d, coh_1d)

    return vel_2d, coh_2d, extent


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    print("08_plot_results.py")
    print(f"Output dir : {PLOTS_DIR}")
    print(f"Units      : cm/year (velocity)  cm (displacement)")

    vel_files = sorted(WORK_DIR.glob("*_velocity_map.tif"))
    if not vel_files:
        print(f"No *_velocity_map.tif files found in {WORK_DIR}")
        print("Run 07_export_geotiff.py first.")
        return

    stems = [f.stem.replace("_velocity_map", "") for f in vel_files]
    print(f"Found {len(stems)} network(s): {stems}")

    vel_maps = []
    coh_maps = []
    extents  = []

    for stem in stems:
        vel_2d, coh_2d, extent = process_network(stem)
        if vel_2d is not None:
            vel_maps.append(vel_2d)
            coh_maps.append(coh_2d)
            extents.append(extent)

    if len(vel_maps) == 2:
        print("\n" + "=" * 60)
        print("Cross-network comparison figures...")
        plot_comparison(stems, vel_maps, extents)

    print("\n" + "=" * 60)
    print(f"Plotting complete. Figures written to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()