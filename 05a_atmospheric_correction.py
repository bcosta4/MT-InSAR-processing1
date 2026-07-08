"""
05a_atmospheric_correction.py
Standalone atmospheric correction applied to the raw phase matrix BEFORE
inversion. Reconstructs each IFG as a full 2D image, applies a Gaussian
spatial low-pass filter (20 km radius, Bartsch et al. 2026 approach), and
subtracts the long-wavelength component from each IFG independently.

Run this BEFORE 05b_inversion.py. If this script has been run,
05b_inversion.py will automatically use the corrected phase matrix.

Why before inversion (not after):
Correcting individual IFG phases before they enter the design matrix
prevents atmospheric errors from propagating through the inversion and
corrupting multiple epochs in the time series simultaneously.

Atmospheric model assumption:
Long-wavelength phase signals (>20 km) are treated as atmospheric /
ionospheric delay. Permafrost subsidence at Tuktoyaktuk operates at
metre to kilometre scales and is retained in the high-frequency residual.

Parallelisation:
IFGs are processed concurrently using ProcessPoolExecutor (N_WORKERS =
CPU count - 1). Each worker reconstructs one IFG 2D raster, runs the
Gaussian filter, and returns the corrected 1D vector. This gives a
10-15x speedup over serial processing on a typical multi-core machine.

Pixel spacing:
Reads from mean_coherence.tif. Handles three CRS scenarios:
  1. Standard projected CRS (e.g. EPSG 3155 NAD83/BC Albers): is_projected
     = True, transform values in metres — used directly.
  2. LOCAL_CS (e.g. NAD83/BC Albers stored as local): is_projected = False
     but pixel values >> 0.1, so clearly in metres — used directly.
     This was the TC1 bug: is_projected returned False, triggering the
     degree->metre conversion and producing sigma = 0.0 px (no correction).
  3. Geographic CRS (degrees): pixel values << 0.1 — converted to metres
     at 69.5 deg N scene centre.

Outputs (per network, written to WORK_DIR):
  _phase_matrix_atmo.npy      float32 (n_ifgs x n_valid_pixels)
  _atmo_layers.npy            float32 (n_ifgs x n_valid_pixels)
  _atmo_correction_report.txt per-IFG correction statistics
  _atmo_correction_stats.csv  per-IFG correction statistics (CSV)
"""

import os
import csv
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR = Path(r"C:\\bcosta\\DS-InSAR\\tc")

# ── Configuration ──────────────────────────────────────────────────────────────
FILTER_RADIUS_KM         = 20.0   # Bartsch et al. 2026: 20 km
FILTER_TRUNCATE          = 2.0    # kernel half-width in sigma units
                                   # 2.0 = 4x faster than default 4.0,
                                   # negligible quality difference at sigma >> 100 px
PIXEL_SPACING_M_FALLBACK = 27.0   # used only if mean_coherence.tif is missing
MIN_FINITE_FRACTION      = 0.05
NAN_FILL                 = "mean" # "mean" or "zero"
N_WORKERS                = max(1, (os.cpu_count() or 2) - 1)

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_pixel_spacing_from_tif(tif_path):
    """
    Return mean pixel spacing in metres from a GeoTIFF.

    Handles three CRS scenarios correctly:

    1. Standard projected CRS (is_projected=True):
       Transform values are in metres — use directly.

    2. LOCAL_CS (e.g. NAD83/BC Albers written without EPSG authority):
       rasterio reports is_projected=False even though units are metres.
       Guard: if pixel size > 0.1 it cannot be degrees (Sentinel-1 pixels
       are never 0.1 degrees wide), so treat as metres directly.
       Without this guard, LOCAL_CS triggered the degree->metre conversion,
       producing a fictional spacing of ~2,076,782 m and sigma = 0.0 px,
       which subtracted the entire scene mean from every pixel (100% RMS
       reduction, all real signal destroyed).

    3. Geographic CRS (degrees, is_projected=False, pixel size < 0.1):
       Convert degrees to metres at scene latitude 69.5 deg N.
    """
    import rasterio
    with rasterio.open(tif_path) as src:
        xres = abs(src.transform.a)
        yres = abs(src.transform.e)
        crs  = src.crs

    # Branch 1 & 2: projected or LOCAL_CS (units are metres)
    if (crs and crs.is_projected) or (xres > 0.1):
        spacing_m   = (xres + yres) / 2.0
        crs_branch  = "projected" if (crs and crs.is_projected) else "LOCAL_CS (metre guard)"
        return spacing_m, crs_branch

    # Branch 3: geographic (degrees) -> metres at scene latitude
    lat_rad       = np.radians(69.5)
    m_per_deg_lon = 111320.0 * np.cos(lat_rad)
    m_per_deg_lat = 111320.0
    spacing_m     = (xres * m_per_deg_lon + yres * m_per_deg_lat) / 2.0
    return spacing_m, "geographic"


def compute_filter_sigma(filter_radius_km, pixel_spacing_m):
    """
    Convert a spatial filter radius (km) to a Gaussian sigma (pixels).
    Uses radius = 3 * sigma convention.
    """
    radius_px = (filter_radius_km * 1000.0) / pixel_spacing_m
    return radius_px / 3.0


def reconstruct_2d(values_1d, pixel_coords, shape):
    out = np.full(shape, np.nan, dtype=np.float32)
    out[pixel_coords[:, 0], pixel_coords[:, 1]] = values_1d
    return out


def apply_atmospheric_filter(phase_2d, sigma_px, nan_fill="mean", truncate=2.0):
    nan_mask = ~np.isfinite(phase_2d)
    filled   = phase_2d.copy()
    fill_val = float(np.nanmean(phase_2d)) if nan_fill == "mean" else 0.0
    filled[nan_mask] = fill_val

    atmo_layer = gaussian_filter(
        filled, sigma=sigma_px, truncate=truncate
    ).astype(np.float32)
    atmo_layer[nan_mask] = np.nan

    corrected  = phase_2d - atmo_layer
    valid      = np.isfinite(phase_2d)
    rms_before = float(np.sqrt(np.nanmean(phase_2d[valid] ** 2)))
    rms_after  = float(np.sqrt(np.nanmean(corrected[valid] ** 2)))
    atmo_rms   = float(np.sqrt(np.nanmean(atmo_layer[valid] ** 2)))

    return corrected, atmo_layer, rms_before, rms_after, atmo_rms


def load_ifg_metadata(csv_path):
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


# ── Worker function (module-level so ProcessPoolExecutor can pickle it) ────────
def _correct_one_ifg(args):
    """
    Process a single IFG in a worker process.
    Returns (i, corr_1d, atmo_1d, fin_frac, stats_or_None)
    stats = (rms_before, rms_after, atmo_rms) if corrected, None if skipped.
    """
    (i, phase_1d, n_valid, pixel_coords,
     nrows, ncols, sigma_px, nan_fill,
     min_fin_frac, truncate) = args

    n_finite = int(np.sum(np.isfinite(phase_1d)))
    fin_frac = n_finite / n_valid

    if fin_frac < min_fin_frac:
        return (i, phase_1d, np.zeros(n_valid, dtype=np.float32),
                fin_frac, None)

    phase_2d = reconstruct_2d(phase_1d, pixel_coords, (nrows, ncols))
    corr_2d, atmo_2d, rms_b, rms_a, atmo_rms = apply_atmospheric_filter(
        phase_2d, sigma_px, nan_fill, truncate
    )

    corr_1d = corr_2d[pixel_coords[:, 0], pixel_coords[:, 1]]
    atmo_1d = atmo_2d[pixel_coords[:, 0], pixel_coords[:, 1]]

    return i, corr_1d, atmo_1d, fin_frac, (rms_b, rms_a, atmo_rms)


# ── Per-network processing ─────────────────────────────────────────────────────
def process_network(stem):
    print("\n" + "=" * 60)
    print(f"Network : {stem}")

    phase_path  = WORK_DIR / f"{stem}_phase_matrix.npy"
    coords_path = WORK_DIR / f"{stem}_pixel_coords.npy"
    coh_tif     = WORK_DIR / f"{stem}_mean_coherence.tif"
    pairs_path  = WORK_DIR / f"{stem}_ifg_pairs.csv"

    for p in [phase_path, coords_path]:
        if not p.exists():
            print(f"  ERROR: missing {p.name} — run 03_load_phases.py first.")
            return

    phase_matrix = np.load(phase_path)
    pixel_coords = np.load(coords_path)
    n_ifgs, n_valid = phase_matrix.shape

    if coh_tif.exists():
        import rasterio
        pixel_spacing_m, crs_branch = get_pixel_spacing_from_tif(coh_tif)
        with rasterio.open(coh_tif) as src:
            nrows, ncols = src.height, src.width
            crs_name     = src.crs.to_string() if src.crs else "unknown"
        print(f"  CRS           : {crs_name}")
        print(f"  CRS branch    : {crs_branch}")
        print(f"  Pixel spacing : {pixel_spacing_m:.1f} m (from GeoTIFF)")
    else:
        pixel_spacing_m = PIXEL_SPACING_M_FALLBACK
        nrows = int(pixel_coords[:, 0].max()) + 1
        ncols = int(pixel_coords[:, 1].max()) + 1
        crs_branch = "fallback"
        print(f"  Pixel spacing : {pixel_spacing_m:.1f} m (fallback — coherence TIF missing)")

    sigma_px = compute_filter_sigma(FILTER_RADIUS_KM, pixel_spacing_m)

    # Safety check: warn if sigma is suspiciously small
    if sigma_px < 1.0:
        print(f"  [WARN] sigma = {sigma_px:.4f} px — filter is effectively inactive!")
        print(f"         Check CRS and pixel spacing. Expected sigma >> 100 px.")

    print(f"  Phase matrix  : {phase_matrix.shape}")
    print(f"  Raster shape  : {nrows} x {ncols}")
    print(f"  Filter radius : {FILTER_RADIUS_KM} km -> sigma = {sigma_px:.1f} px  "
          f"(truncate={FILTER_TRUNCATE})")
    print(f"  NaN fill      : {NAN_FILL}")
    print(f"  Workers       : {N_WORKERS}")

    ifg_meta = load_ifg_metadata(pairs_path) if pairs_path.exists() else []

    # ── Build args for parallel workers ───────────────────────────────────────
    args_list = [
        (i, phase_matrix[i, :], n_valid, pixel_coords,
         nrows, ncols, sigma_px, NAN_FILL,
         MIN_FINITE_FRACTION, FILTER_TRUNCATE)
        for i in range(n_ifgs)
    ]

    print(f"\n  Correcting {n_ifgs} IFGs in parallel ({N_WORKERS} workers)...")

    # ── Run parallel correction ────────────────────────────────────────────────
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        results = list(executor.map(_correct_one_ifg, args_list))

    phase_corrected = np.full_like(phase_matrix, np.nan)
    atmo_layers     = np.full_like(phase_matrix, np.nan)
    report_rows     = []
    n_corrected = n_skipped = 0

    for i, corr_1d, atmo_1d, fin_frac, stats in results:
        phase_corrected[i, :] = corr_1d
        atmo_layers[i, :]     = atmo_1d

        band_name = ifg_meta[i]["band_name"] if i < len(ifg_meta) else f"IFG_{i:03d}"

        if stats is None:
            n_skipped += 1
            print(f"  [{i+1:02d}/{n_ifgs}] SKIPPED "
                  f"(finite={fin_frac*100:.1f}%) {band_name[:45]}")
            report_rows.append({
                "ifg_index": i, "band_name": band_name,
                "status": "skipped", "fin_pct": round(fin_frac * 100, 1),
                "rms_before": "", "rms_after": "",
                "atmo_rms": "", "rms_change_pct": "",
            })
        else:
            rms_b, rms_a, atmo_rms = stats
            rms_change_pct = (rms_a - rms_b) / rms_b * 100 if rms_b > 0 else 0.0
            n_corrected += 1
            print(f"  [{i+1:02d}/{n_ifgs}] {band_name[:45]:<45} "
                  f"RMS: {rms_b:.3f}->{rms_a:.3f} rad "
                  f"atmo={atmo_rms:.3f} D{rms_change_pct:+.1f}%")
            report_rows.append({
                "ifg_index": i, "band_name": band_name,
                "status": "corrected", "fin_pct": round(fin_frac * 100, 1),
                "rms_before": round(rms_b, 4), "rms_after": round(rms_a, 4),
                "atmo_rms": round(atmo_rms, 4),
                "rms_change_pct": round(rms_change_pct, 2),
            })

    corrected_rows = [r for r in report_rows if r["status"] == "corrected"]
    if corrected_rows:
        rms_changes   = [r["rms_change_pct"] for r in corrected_rows]
        atmo_rms_vals = [r["atmo_rms"]       for r in corrected_rows]
        print(f"\n  IFGs corrected  : {n_corrected} / {n_ifgs}")
        print(f"  IFGs skipped    : {n_skipped}")
        print(f"  Mean RMS change : {np.mean(rms_changes):+.1f}%")
        print(f"  Mean atmo RMS   : {np.mean(atmo_rms_vals):.3f} rad")

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_phase = WORK_DIR / f"{stem}_phase_matrix_atmo.npy"
    out_atmo  = WORK_DIR / f"{stem}_atmo_layers.npy"
    out_rep   = WORK_DIR / f"{stem}_atmo_correction_report.txt"
    out_csv   = WORK_DIR / f"{stem}_atmo_correction_stats.csv"

    np.save(out_phase, phase_corrected)
    np.save(out_atmo,  atmo_layers)

    if report_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)

    report_lines = [
        f"Atmospheric correction report : {stem}\n",
        f"Filter radius  : {FILTER_RADIUS_KM} km\n",
        f"Filter sigma   : {sigma_px:.2f} px\n",
        f"Filter truncate: {FILTER_TRUNCATE}\n",
        f"Pixel spacing  : {pixel_spacing_m:.1f} m\n",
        f"CRS branch     : {crs_branch}\n",
        f"NaN fill       : {NAN_FILL}\n",
        f"Min finite frac: {MIN_FINITE_FRACTION}\n",
        f"Workers        : {N_WORKERS}\n",
        f"IFGs corrected : {n_corrected} / {n_ifgs}\n",
        f"IFGs skipped   : {n_skipped}\n",
    ]
    if corrected_rows:
        report_lines += [
            f"Mean RMS change     : {np.mean(rms_changes):+.2f}%\n",
            f"Mean atmospheric RMS: {np.mean(atmo_rms_vals):.4f} rad\n",
        ]
    Path(out_rep).write_text("".join(report_lines), encoding="utf-8")

    print(f"\n  Written: {out_phase.name} ({phase_corrected.nbytes / 1e6:.1f} MB)")
    print(f"  Written: {out_atmo.name}")
    print(f"  Written: {out_csv.name}")
    print(f"  Written: {out_rep.name}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    phase_files = sorted(WORK_DIR.glob("*_phase_matrix.npy"))
    phase_files = [p for p in phase_files if "atmo" not in p.stem]

    if not phase_files:
        print(f"No *_phase_matrix.npy files found in {WORK_DIR}")
        print("Run 03_load_phases.py first.")
        return

    stems = [f.stem.replace("_phase_matrix", "") for f in phase_files]
    print(f"Found {len(stems)} network(s): {stems}")
    print(f"Parallel workers: {N_WORKERS} (of {os.cpu_count()} logical CPUs)")

    for stem in stems:
        process_network(stem)

    print("\n" + "=" * 60)
    print("Atmospheric correction complete.")
    print("Next: run 05b_inversion.py — it will auto-detect corrected phases.")


if __name__ == "__main__":
    main()
