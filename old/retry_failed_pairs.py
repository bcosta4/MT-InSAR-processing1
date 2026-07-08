import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# Import classes from your main script
from snap_insar_v18 import (
    Config,
    DEMManager,
    SLCFileManager,
    PairGenerator,
    SnapGraphGenerator,
    SnapProcessor,
)

def load_failed_pairs(results_csv: Path):
    df = pd.read_csv(results_csv)
    failed = df[df["status"] == "FAILED_PROCESSING"]  # or .str.contains("FAILED")
    return failed

def main():
    parser = argparse.ArgumentParser(
        description="Retry FAILED_PROCESSING Sentinel-1 InSAR pairs with SNAP 12 (using snap_insar_v18)."
    )
    parser.add_argument("--slc-dir", required=True, help="Directory containing SLC .zip files")
    parser.add_argument("--dem", help="Optional: External DEM path or directory/CSV list")
    parser.add_argument("--output-dir", required=True, help="Same output directory used in the first run")
    parser.add_argument("--ram", type=int, default=8, help="Max heap size for SNAP (GB)")
    parser.add_argument(
        "--results-csv",
        default="processing_results.csv",
        help="Path to the original processing_results.csv"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_csv = Path(args.results_csv)

    if not results_csv.is_file():
        raise FileNotFoundError(f"processing_results.csv not found at {results_csv}")

    # Init config and logging
    config = Config()
    config.output_dir = output_dir
    config.ram_gb = args.ram

    # DEM handling (same as in main script)
    if args.dem:
        dem_manager = DEMManager(args.dem, config.temp_dir, config.logger)
        dem_manager.parse_dem_paths()
        dem_mosaic = dem_manager.get_dem_for_snap()
        config.dem_paths = dem_manager.dem_paths
        config.dem_mosaic_path = dem_mosaic
    else:
        config.logger.info("No DEM specified. Using built-in Copernicus 30m Global DEM.")
        config.dem_mosaic_path = None

    # Discover SLCs and build inventory (same as InSARBatchProcessor.run)
    slc_dir = Path(args.slc_dir)
    file_manager = SLCFileManager(slc_dir, config.logger)
    slc_files = file_manager.discover_slc_files()
    file_manager.debug_filename_parsing()
    if not slc_files:
        config.logger.error("No SLC files found. Exiting.")
        return
    inventory = file_manager.build_inventory()
    if not inventory:
        config.logger.error("Could not build inventory from SLC files. Exiting.")
        return

    pair_gen = PairGenerator(inventory, config.min_bt_days, config.max_bt_days, config.logger)

    # Create DEM-aware graph generator and SNAP processor
    graph_gen = SnapGraphGenerator(config.dem_mosaic_path, config.nlooks_range, config.nlooks_azimuth, config.logger)
    snap_proc = SnapProcessor(config.snap_exe, config.ram_gb, config.logger)

    failed = load_failed_pairs(results_csv)
    config.logger.info(f"Found {len(failed)} FAILED_PROCESSING pairs to retry.")

    for idx, row in failed.iterrows():
        pair_name = row["pair"]
        ref_str, sec_str = pair_name.split("_")
        ref_date = datetime.strptime(ref_str, "%Y%m%d")
        sec_date = datetime.strptime(sec_str, "%Y%m%d")

        config.logger.info(f"\nRetrying failed pair: {pair_name}")

        ref_slc, sec_slc = pair_gen.get_slc_files_for_pair(ref_date, sec_date)
        if not ref_slc or not sec_slc:
            config.logger.error(f"SLC files not found for pair {pair_name}, skipping.")
            continue

        pair_dir = config.output_dir / pair_name
        ifg_dir = pair_dir / f"{pair_name}_ifg"
        graph_xml = graph_gen.create_insar_graph()
        graph_file = pair_dir / "insar_graph.xml"
        graph_gen.save_graph(graph_xml, graph_file)

        success = snap_proc.process_pair(graph_file, ifg_dir, ref_slc, sec_slc)
        if success:
            config.logger.info(f"✓ Retry succeeded for {pair_name}")
        else:
            config.logger.error(f"✗ Retry still failing for {pair_name}")

if __name__ == "__main__":
    main()
