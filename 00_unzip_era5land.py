"""
00_unzip_era5land.py  (v2)
Re-extracts ERA5-Land ZIP archives, naming each output file correctly
by deriving the target name from the ZIP filename rather than the
generic internal name (CDS always uses data_0.nc internally).

Run ONCE before 00b_prepare_temperature.py.
All original .zip files are left untouched so this can be re-run safely.
"""

import zipfile
import shutil
from pathlib import Path

NC_DIR   = Path(r"C:\bcosta\DS-InSAR\era5\nc_raw")
ERA5_DIR = Path(r"C:\bcosta\DS-InSAR\era5")

def is_zip(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False

def extract_zip_with_rename(zip_path, out_dir):
    """
    Extract the first .nc file from zip_path and save it as
    <zip_stem>.nc  (i.e. era5land_t2m_2019.zip -> era5land_t2m_2019.nc).
    The internal filename (data_0.nc) is ignored entirely.
    """
    # Target name = ZIP stem + .nc
    # zip_path.stem = "era5land_t2m_2019" (after the original was renamed .zip)
    target_name = zip_path.stem + ".nc"
    target_path = out_dir / target_name

    if target_path.exists():
        print(f"  Already exists: {target_name} — skipping")
        return target_path

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        # Find the first .nc member
        nc_members = [m for m in members if m.lower().endswith(".nc")]
        if not nc_members:
            print(f"  WARNING: no .nc file inside {zip_path.name}")
            print(f"           Contents: {members}")
            return None
        member = nc_members[0]
        print(f"  {zip_path.name}  ->  extract [{member}]  ->  {target_name}")
        # Extract to a temp name then rename to avoid overwrite clashes
        zf.extract(member, out_dir)
        extracted = out_dir / member
        extracted.rename(target_path)

    size_mb = target_path.stat().st_size / 1024**2
    print(f"           Written: {target_name}  ({size_mb:.1f} MB)")
    return target_path


def main():
    NC_DIR.mkdir(parents=True, exist_ok=True)

    # Find all .zip files in nc_raw and era5 dirs that match our naming convention
    search_dirs = [d for d in [NC_DIR, ERA5_DIR] if d.exists()]
    zip_files = []
    for d in search_dirs:
        zip_files.extend(sorted(d.glob("era5land_*.zip")))

    if not zip_files:
        print("No era5land_*.zip files found.")
        print(f"Expected location: {NC_DIR}")
        print("Check that 00_unzip_era5land.py (v1) already ran and renamed the ZIPs.")
        return

    print(f"Found {len(zip_files)} ZIP file(s) to extract:\n")

    results = {"ok": [], "skip": [], "fail": []}
    for zp in zip_files:
        try:
            out = extract_zip_with_rename(zp, NC_DIR)
            if out and out.exists():
                results["ok"].append(out.name)
            else:
                results["fail"].append(zp.name)
        except Exception as e:
            print(f"  ERROR extracting {zp.name}: {e}")
            results["fail"].append(zp.name)

    # Also handle the leftover data_0.nc if present (from the v1 overwrite)
    leftover = NC_DIR / "data_0.nc"
    if leftover.exists():
        print(f"\nRemoving leftover data_0.nc ({leftover.stat().st_size/1024**2:.1f} MB)")
        leftover.unlink()

    # Summary
    print("\n" + "=" * 50)
    print(f"Extracted OK : {len(results['ok'])}")
    print(f"Skipped      : {len(results['skip'])}")
    print(f"Failed       : {len(results['fail'])}")

    if results["fail"]:
        print("\nFailed files:")
        for f in results["fail"]:
            print(f"  {f}")

    print("\nFiles now in", NC_DIR, ":")
    nc_files = sorted(NC_DIR.glob("*.nc"))
    if not nc_files:
        print("  (none — check extraction errors above)")
    for p in nc_files:
        size_mb = p.stat().st_size / 1024**2
        print(f"  {p.name:<40}  {size_mb:>7.1f} MB")

    # Verify expected files
    expected = (
        [f"era5land_t2m_{y}.nc"  for y in range(2019, 2025)] +
        [f"era5land_stl1_{y}.nc" for y in range(2019, 2025)]
    )
    missing = [e for e in expected if not (NC_DIR / e).exists()]
    if missing:
        print("\nWARNING — still missing expected files:")
        for m in missing:
            print(f"  {m}")
        print("Re-download missing years with 00b_download_era5land.py")
    else:
        print("\nAll 12 expected files present (t2m x6, stl1 x6).")
        print("Next: python 00b_prepare_temperature.py")

if __name__ == "__main__":
    main()
