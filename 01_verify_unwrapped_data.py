import re
import xml.etree.ElementTree as ET
from pathlib import Path

DATE_RE = re.compile(r"\d{2}[A-Za-z]{3}\d{4}")


def find_all_text(root, tag_name):
    return [
        el.text.strip()
        for el in root.iter(tag_name)
        if el.text and el.text.strip()
    ]


def parse_hdr(hdr_path):
    samples, lines = None, None
    try:
        text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return samples, lines
    m = re.search(r"^\s*samples\s*=\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        samples = int(m.group(1))
    m = re.search(r"^\s*lines\s*=\s*(\d+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        lines = int(m.group(1))
    return samples, lines


def analyse_stack(dim_path):
    dim_path = Path(dim_path)
    data_dir = dim_path.with_suffix(".data")

    print("\n" + "=" * 60)
    print(f"Stack : {dim_path.name}")

    tree = ET.parse(dim_path)
    root = tree.getroot()

    ncols_txt  = next((el.text.strip() for el in root.iter("NCOLS")  if el.text), None)
    nrows_txt  = next((el.text.strip() for el in root.iter("NROWS")  if el.text), None)
    nbands_txt = next((el.text.strip() for el in root.iter("NBANDS") if el.text), None)

    ncols  = int(ncols_txt)  if ncols_txt  and ncols_txt.isdigit()  else None
    nrows  = int(nrows_txt)  if nrows_txt  and nrows_txt.isdigit()  else None
    nbands = int(nbands_txt) if nbands_txt and nbands_txt.isdigit() else None

    band_names = find_all_text(root, "BAND_NAME")

    # Match actual SNAP underscore naming
    unw_bands   = [b for b in band_names if b.startswith("Unw_Phase_ifg")]
    coh_bands   = [b for b in band_names if b.startswith("coh_")]
    phase_bands = [b for b in band_names if b.startswith("Phase_ifg")]
    i_bands     = [b for b in band_names if b.startswith("i_ifg_")]
    q_bands     = [b for b in band_names if b.startswith("q_ifg_")]
    int_bands   = [b for b in band_names if b.startswith("Intensity_ifg")]
    elev_bands  = [b for b in band_names if b.startswith("elevation")]

    # Extract date pairs from unwrapped bands
    date_pairs = set()
    for b in unw_bands:
        dates = DATE_RE.findall(b)
        if len(dates) >= 2:
            date_pairs.add((dates[0], dates[1]))

    print(f"\n  [DIM metadata]")
    print(f"  Declared size       : {ncols} cols x {nrows} rows")
    print(f"  Declared bands      : {nbands}")
    print(f"  Bands parsed        : {len(band_names)}")
    print(f"  Unw_Phase_ifg       : {len(unw_bands)}")
    print(f"  coh_                : {len(coh_bands)}")
    print(f"  Phase_ifg           : {len(phase_bands)}")
    print(f"  i_ifg / q_ifg       : {len(i_bands)} / {len(q_bands)}")
    print(f"  Intensity_ifg       : {len(int_bands)}")
    print(f"  elevation           : {len(elev_bands)}")
    print(f"  Interferogram pairs : {len(date_pairs)}")

    if unw_bands:
        print(f"\n  Sample Unw_Phase bands (first 5):")
        for b in unw_bands[:5]:
            print(f"    {b}")

    if date_pairs:
        print(f"\n  Date pairs ({len(date_pairs)} total):")
        for master, slave in sorted(date_pairs):
            print(f"    {master}  ->  {slave}")

    # Check .data/ folder
    print(f"\n  [.data/ ENVI headers]")
    if not data_dir.is_dir():
        print(f"  WARNING: data folder not found")
        return

    hdr_files = sorted(data_dir.glob("*.hdr"))
    unw_hdrs  = [h for h in hdr_files if h.stem.startswith("Unw_Phase_ifg")]
    coh_hdrs  = [h for h in hdr_files if h.stem.startswith("coh_")]

    print(f"  .hdr files found    : {len(hdr_files)}")
    print(f"  Unw_Phase_ifg .hdr  : {len(unw_hdrs)}")
    print(f"  coh_ .hdr           : {len(coh_hdrs)}")

    missing_img = [h.with_suffix(".img").name for h in unw_hdrs
                   if not h.with_suffix(".img").exists()]

    if missing_img:
        print(f"  Missing .img files  : {len(missing_img)}")
        for m in missing_img[:5]:
            print(f"    {m}")
    elif unw_hdrs:
        print(f"  All Unw .img files  : PRESENT")

    hdr_sizes = set()
    for h in unw_hdrs:
        s, l = parse_hdr(h)
        if s and l:
            hdr_sizes.add((s, l))

    if hdr_sizes:
        if len(hdr_sizes) == 1:
            s, l = next(iter(hdr_sizes))
            consistent = "CONSISTENT" if (s == ncols and l == nrows) else f"MISMATCH vs DIM {ncols}x{nrows}"
            print(f"  HDR raster size     : {s} x {l} [{consistent}]")
        else:
            print(f"  HDR sizes INCONSISTENT: {hdr_sizes}")


def main():
    folder = Path(".")
    dim_files = sorted(folder.glob("*.dim"))

    if not dim_files:
        print("No .dim files found. Run from C:\\bcosta\\DS-InSAR\\disp")
        return

    print(f"Found {len(dim_files)} .dim file(s): {[f.name for f in dim_files]}")
    for dim in dim_files:
        analyse_stack(dim)

    print("\n" + "=" * 60)
    print("Verification complete.")


if __name__ == "__main__":
    main()