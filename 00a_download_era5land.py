"""
00b_download_era5land.py
Download ERA5-Land 2m air temperature and soil temperatures for
the Tuktoyaktuk Peninsula area (2019-2024) via the CDS API.

Requirements:
    pip install cdsapi
    CDS API key configured at ~/.cdsapirc (see below)

CDS API key setup:
    1. Register at https://cds.climate.copernicus.eu/
    2. Go to your profile -> API key
    3. Create ~/.cdsapirc with:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <YOUR-UID>:<YOUR-API-KEY>

Output files written to: C:/bcosta/DS-InSAR/era5/nc_raw/
    era5land_t2m_YYYY.nc      -- 2m air temperature [K]
    era5land_stl1_YYYY.nc     -- Soil temp layer 1 (0-7cm) [K]
    era5land_stl2_YYYY.nc     -- Soil temp layer 2 (7-28cm) [K]

Spatial extent: Tuktoyaktuk Peninsula + buffer
    North: 70.5, South: 68.5, West: -135.0, East: -131.5
    This covers ~9 ERA5-Land grid cells across the peninsula (~9km spacing)

Temporal: annual files, summer months only (May-October)
    Reduces file size while covering the full thaw season
"""

import cdsapi
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR = Path(r"C:\bcosta\DS-InSAR\era5\nc_raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [str(y) for y in range(2019, 2025)]  # 2019-2024

# Bounding box [North, West, South, East] — CDS format
AREA = [70.5, -135.0, 68.5, -131.5]

# Thaw season months only (reduces download size ~2x vs full year)
# Include May to capture early thaw; October for late freeze-up
MONTHS = ["05", "06", "07", "08", "09", "10"]

# Hourly data — will be resampled to daily in script 00
TIMES = [f"{h:02d}:00" for h in range(24)]

# Variables to download
VARIABLES = {
    "t2m":  "2m_temperature",           # air temp at 2m
    "stl1": "soil_temperature_level_1",  # 0–7 cm depth
    "stl2": "soil_temperature_level_2",  # 7–28 cm depth
}

# ── Download function ──────────────────────────────────────────────────────────
def download_variable(client, var_short, var_long, year):
    out_path = OUT_DIR / f"era5land_{var_short}_{year}.nc"
    if out_path.exists():
        print(f"  Already exists: {out_path.name} -- skipping")
        return
    print(f"  Downloading {var_long} for {year}...")
    client.retrieve(
        "reanalysis-era5-land",
        {
            "variable": var_long,
            "year": year,
            "month": MONTHS,
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": TIMES,
            "area": AREA,
            "format": "netcdf",
        },
        str(out_path),
    )
    print(f"  Written: {out_path.name}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    c = cdsapi.Client()
    print(f"ERA5-Land download: {len(YEARS)} years x {len(VARIABLES)} variables")
    print(f"Area: N={AREA[0]} W={AREA[1]} S={AREA[2]} E={AREA[3]}")
    print(f"Months: {MONTHS}")
    print(f"Output: {OUT_DIR}\n")

    for year in YEARS:
        print(f"\n--- Year {year} ---")
        for var_short, var_long in VARIABLES.items():
            download_variable(c, var_short, var_long, year)

    print("\nDownload complete.")
    print("Next: update script 00 to point to era5land_t2m_YYYY.nc")
    print("      update script 09 to load 2D grid and compute per-pixel DDT")

if __name__ == "__main__":
    main()