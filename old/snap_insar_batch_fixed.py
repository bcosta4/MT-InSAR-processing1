#!/usr/bin/env python3
"""
Sentinel-1 InSAR Batch Processing with SNAP 12+ - FIXED
=========================================================

FIXES APPLIED:
1. Removed Goldstein-Phase-Filtering (not installed in your SNAP)
2. SNAP 12+ syntax: -Pparameter=value ✓
3. gpt.exe detection ✓
4. Output directory: D:\bcosta4\DS-InSAR ✓

Processing pipeline (IW3, VV only):
- TOPSAR Split → Apply Orbit → Back-Geocoding → ESD
- Interferogram → Deburst → Multilooking → Terrain Correction

Author: InSAR Processing Group
Date: January 2026
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from collections import defaultdict


class Config:
    """Processing configuration."""
    
    def __init__(self):
        self.snap_exe = self.find_snap_gpt()
        self.min_bt_days = 6
        self.max_bt_days = 500
        self.dem_paths = []
        self.dem_mosaic_path = None
        self.output_dir = Path('D:\\bcosta4\\DS-InSAR')
        self.temp_dir = self.output_dir / 'temp'
        self.log_file = self.output_dir / 'processing.log'
        self.nlooks_range = 8
        self.nlooks_azimuth = 2
        self.coherence_threshold = 0.3
        self.ram_gb = 8
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
    
    def find_snap_gpt(self) -> str:
        """Locate SNAP GPT executable."""
        
        snap_gpt_paths = [
            Path('C:\\Program Files\\esa-snap\\bin\\gpt.exe'),
            Path('C:\\Program Files\\esa-snap\\bin\\gpt'),
            Path('C:\\Program Files\\snap\\bin\\gpt.exe'),
            Path('C:\\snap\\bin\\gpt.exe'),
            Path.home() / 'snap' / 'bin' / 'gpt.exe',
            Path('/opt/snap/bin/gpt'),
        ]
        
        for path in snap_gpt_paths:
            if path.exists():
                self.logger_console(f"✓ Found SNAP GPT at: {path}")
                return str(path)
        
        snap_env = os.environ.get('SNAP_EXE')
        if snap_env and Path(snap_env).exists():
            self.logger_console(f"✓ Found SNAP from SNAP_EXE: {snap_env}")
            return snap_env
        
        try:
            cmd = ['where', 'gpt'] if os.name == 'nt' else ['which', 'gpt']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpt_path = result.stdout.strip()
                self.logger_console(f"✓ Found 'gpt' in PATH: {gpt_path}")
                return gpt_path
        except:
            pass
        
        raise FileNotFoundError("SNAP GPT not found. Install SNAP 12+ from https://step.esa.int/")
    
    def logger_console(self, msg: str):
        """Simple console log before logger initialized."""
        print(f"[SNAP CONFIG] {msg}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging to file and console."""
        logger = logging.getLogger('InSARBatch')
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger


class DEMManager:
    """Manages DEM files."""
    
    def __init__(self, dem_input: str, temp_dir: Path, logger: logging.Logger):
        self.dem_input = dem_input
        self.temp_dir = temp_dir
        self.logger = logger
        self.dem_paths = []
        self.dem_mosaic_path = None
    
    def parse_dem_paths(self) -> List[Path]:
        """Parse single or comma-separated DEM paths."""
        dem_list = [p.strip() for p in self.dem_input.split(',')]
        dem_paths = []
        
        for dem_str in dem_list:
            dem_path = Path(dem_str)
            if not dem_path.exists():
                self.logger.error(f"DEM file not found: {dem_path}")
                raise FileNotFoundError(f"DEM not found: {dem_path}")
            dem_paths.append(dem_path)
            self.logger.info(f"Registered DEM: {dem_path.name}")
        
        self.dem_paths = dem_paths
        return dem_paths
    
    def create_vrt_mosaic(self) -> Path:
        """Create VRT mosaic from multiple DEM files."""
        if len(self.dem_paths) == 1:
            self.logger.info(f"Single DEM detected: {self.dem_paths[0].name}")
            self.dem_mosaic_path = self.dem_paths[0]
            return self.dem_paths[0]
        
        self.logger.info(f"Creating VRT mosaic from {len(self.dem_paths)} DEM tiles...")
        vrt_path = self.temp_dir / 'dem_mosaic.vrt'
        
        gdal_cmd = ['gdalbuildvrt', '-separate', '-overwrite', str(vrt_path),
                    *[str(p) for p in self.dem_paths]]
        
        try:
            result = subprocess.run(gdal_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                self.logger.info(f"✓ VRT mosaic created: {vrt_path}")
                self.dem_mosaic_path = vrt_path
                return vrt_path
            else:
                raise RuntimeError(f"gdalbuildvrt failed: {result.stderr}")
        except FileNotFoundError:
            self.logger.error("gdalbuildvrt not found. Ensure GDAL is in PATH.")
            raise
    
    def get_dem_for_snap(self) -> Path:
        """Get DEM path for SNAP processing."""
        if not self.dem_paths:
            self.parse_dem_paths()
        if self.dem_mosaic_path:
            return self.dem_mosaic_path
        return self.create_vrt_mosaic()


class SLCFileManager:
    """Manages Sentinel-1 SLC file inventory."""
    
    def __init__(self, slc_dir: Path, logger: logging.Logger):
        self.slc_dir = Path(slc_dir)
        self.logger = logger
        self.slc_files = []
        self.inventory = {}
    
    def discover_slc_files(self, pattern='*.zip') -> List[Path]:
        """Discover SLC files in directory."""
        self.logger.info(f"Discovering SLC files in {self.slc_dir}")
        slc_files = sorted(self.slc_dir.glob(pattern))
        if not slc_files:
            self.logger.warning(f"No SLC files found matching {pattern}")
            return []
        self.logger.info(f"Found {len(slc_files)} SLC files")
        self.slc_files = slc_files
        return slc_files
    
    def extract_acquisition_date(self, slc_file: Path) -> Optional[datetime]:
        """Extract acquisition date from SLC filename."""
        filename = slc_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 6:
            date_str = parts[5]
            try:
                acq_date = datetime.strptime(date_str[:8], '%Y%m%d')
                return acq_date
            except ValueError:
                return None
        return None
    
    def build_inventory(self) -> Dict:
        """Build inventory of SLC files with acquisition dates."""
        inventory = defaultdict(list)
        
        for slc_file in self.slc_files:
            acq_date = self.extract_acquisition_date(slc_file)
            if acq_date:
                inventory[acq_date].append(slc_file)
                self.logger.debug(f"Catalogued {slc_file.name} (Date: {acq_date.strftime('%Y-%m-%d')})")
        
        self.inventory = dict(inventory)
        self.logger.info(f"Inventory built: {len(self.inventory)} unique acquisition dates")
        return self.inventory


class PairGenerator:
    """Generate valid interferometric pairs based on temporal baseline."""
    
    def __init__(self, inventory: Dict, min_bt: int, max_bt: int, logger: logging.Logger):
        self.inventory = inventory
        self.min_bt = min_bt
        self.max_bt = max_bt
        self.logger = logger
        self.valid_pairs = []
    
    def generate_pairs(self) -> List[Tuple]:
        """Generate all valid interferometric pairs."""
        dates = sorted(self.inventory.keys())
        pairs = []
        
        for i, ref_date in enumerate(dates):
            for sec_date in dates[i+1:]:
                bt = (sec_date - ref_date).days
                if self.min_bt <= bt <= self.max_bt:
                    pairs.append((ref_date, sec_date, bt))
        
        self.valid_pairs = pairs
        self.logger.info(f"Generated {len(pairs)} valid pairs (Bt: {self.min_bt}-{self.max_bt} days)")
        return pairs
    
    def get_slc_files_for_pair(self, ref_date: datetime, sec_date: datetime) -> Tuple:
        """Get SLC file paths for interferometric pair."""
        ref_files = self.inventory.get(ref_date, [])
        sec_files = self.inventory.get(sec_date, [])
        if not ref_files or not sec_files:
            return None, None
        return ref_files[0], sec_files[0]
    
    def filter_by_coherence_expectation(self, land_cover_class: str = 'sparse_tundra') -> List[Tuple]:
        """Pre-filter pairs by expected coherence."""
        tau_values = {'sparse_tundra': 40, 'urban': 180, 'bare_rock': 120, 'grassland': 30, 'forest': 15}
        tau = tau_values.get(land_cover_class, 40)
        
        filtered_pairs = []
        for ref_date, sec_date, bt in self.valid_pairs:
            gamma_0 = 0.58
            expected_gamma = gamma_0 * np.exp(-bt / tau)
            filtered_pairs.append((ref_date, sec_date, bt, expected_gamma))
        
        filtered_pairs.sort(key=lambda x: x[3], reverse=True)
        
        self.logger.info(
            f"Total pairs for processing: {len(filtered_pairs)}\n"
            f"  High coherence (gamma > 0.5): {sum(1 for p in filtered_pairs if p[3] > 0.5)}\n"
            f"  Medium coherence (0.3 < gamma < 0.5): {sum(1 for p in filtered_pairs if 0.3 < p[3] <= 0.5)}\n"
            f"  Low coherence (gamma < 0.3): {sum(1 for p in filtered_pairs if p[3] <= 0.3)}"
        )
        return filtered_pairs


class SnapGraphGenerator:
    """Generate SNAP processing graphs (XML format)."""
    
    def __init__(self, dem_path: Path, nlooks_range: int, nlooks_azimuth: int, logger: logging.Logger):
        self.dem_path = str(dem_path)
        self.nlooks_range = nlooks_range
        self.nlooks_azimuth = nlooks_azimuth
        self.logger = logger
    
    def create_insar_graph(self, ref_slc: str, sec_slc: str) -> str:
        """Generate SNAP processing graph - Goldstein REMOVED (not available)."""
        
        graph = f"""<?xml version="1.0" encoding="UTF-8"?>
<graph id="InSARGraph">
    <version>1.0</version>
    
    <!-- Step 1: TOPSAR Split (reference) - IW3, VV only -->
    <node id="split_ref">
        <operator>TOPSAR-Split</operator>
        <sources/>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <sourceProduct>{ref_slc}</sourceProduct>
            <subswath>IW3</subswath>
            <selectedPolarisations>VV</selectedPolarisations>
        </parameters>
    </node>
    
    <!-- Step 2: Apply Orbit File (reference) -->
    <node id="orbit_ref">
        <operator>Apply-Orbit-File</operator>
        <sources>
            <sourceProduct refid="split_ref"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <orbitType>RESTITUTED</orbitType>
            <polyDegree>3</polyDegree>
            <continueOnFail>false</continueOnFail>
        </parameters>
    </node>
    
    <!-- Step 3: TOPSAR Split (secondary) - IW3, VV only -->
    <node id="split_sec">
        <operator>TOPSAR-Split</operator>
        <sources/>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <sourceProduct>{sec_slc}</sourceProduct>
            <subswath>IW3</subswath>
            <selectedPolarisations>VV</selectedPolarisations>
        </parameters>
    </node>
    
    <!-- Step 4: Apply Orbit File (secondary) -->
    <node id="orbit_sec">
        <operator>Apply-Orbit-File</operator>
        <sources>
            <sourceProduct refid="split_sec"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <orbitType>RESTITUTED</orbitType>
            <polyDegree>3</polyDegree>
            <continueOnFail>false</continueOnFail>
        </parameters>
    </node>
    
    <!-- Step 5: Back-Geocoding (coregistration with DEM) -->
    <node id="backgeo">
        <operator>Back-Geocoding</operator>
        <sources>
            <sourceProduct refid="orbit_ref"/>
            <sourceProduct.1 refid="orbit_sec"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <demName>SRTM 1Sec HGT</demName>
            <externalDEMFile>{self.dem_path}</externalDEMFile>
            <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
            <externalDEMApplyEGM>true</externalDEMApplyEGM>
            <maskOutAreas>true</maskOutAreas>
            <outputRangeShift>true</outputRangeShift>
            <outputAzimuthShift>true</outputAzimuthShift>
        </parameters>
    </node>
    
    <!-- Step 6: Enhanced Spectral Diversity (ESD) - phase alignment -->
    <node id="esd">
        <operator>Enhanced-Spectral-Diversity</operator>
        <sources>
            <sourceProduct refid="backgeo"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <fineWinWidthStr>512</fineWinWidthStr>
            <fineWinHeightStr>512</fineWinHeightStr>
            <fineWinAccAzimuth>16</fineWinAccAzimuth>
            <fineWinAccRange>16</fineWinAccRange>
            <fineWinOversampling>2</fineWinOversampling>
            <useProductAvgPhaseArray>true</useProductAvgPhaseArray>
            <overallAzimuthShift>0.0</overallAzimuthShift>
            <overallRangeShift>0.0</overallRangeShift>
        </parameters>
    </node>
    
    <!-- Step 7: Interferogram generation -->
    <node id="ifg">
        <operator>Interferogram</operator>
        <sources>
            <sourceProduct refid="esd"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
            <flatEarthEstimationDegree>5</flatEarthEstimationDegree>
            <orbitDegree>3</orbitDegree>
            <useMatrixFile>false</useMatrixFile>
            <includeCoherence>true</includeCoherence>
            <coherenceWindowSize>32</coherenceWindowSize>
            <squarePixel>true</squarePixel>
            <forceCoheenceComputation>false</forceCoheenceComputation>
            <subtractTopographicPhase>true</subtractTopographicPhase>
        </parameters>
    </node>
    
    <!-- Step 8: TOPSAR Deburst -->
    <node id="deburst">
        <operator>TOPSAR-Deburst</operator>
        <sources>
            <sourceProduct refid="ifg"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
        </parameters>
    </node>
    
    <!-- Step 9: Multilooking ({self.nlooks_range}x{self.nlooks_azimuth}) -->
    <node id="multilook">
        <operator>Multilook</operator>
        <sources>
            <sourceProduct refid="deburst"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <nRgLooks>{self.nlooks_range}</nRgLooks>
            <nAzLooks>{self.nlooks_azimuth}</nAzLooks>
            <outputIntensity>false</outputIntensity>
            <greyScaleOutput>false</greyScaleOutput>
        </parameters>
    </node>
    
    <!-- Step 10: Terrain Correction -->
    <node id="terrain_correct">
        <operator>Terrain-Correction</operator>
        <sources>
            <sourceProduct refid="multilook"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <demName>SRTM 1Sec HGT</demName>
            <externalDEMFile>{self.dem_path}</externalDEMFile>
            <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
            <externalDEMApplyEGM>true</externalDEMApplyEGM>
            <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
            <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
            <pixelSpacingInMeter>30</pixelSpacingInMeter>
            <mapProjection>WGS84(DD)</mapProjection>
        </parameters>
    </node>
    
    <!-- Output: Final interferogram product -->
    <node id="write_ifg">
        <operator>Write</operator>
        <sources>
            <sourceProduct refid="terrain_correct"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <file>$OUTPUT_IFG</file>
            <formatName>BEAM-DIMAP</formatName>
        </parameters>
    </node>
</graph>
"""
        return graph
    
    def save_graph(self, graph_xml: str, output_file: Path) -> None:
        """Save graph XML to file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(graph_xml)
        self.logger.debug(f"Saved graph XML: {output_file}")


class SnapProcessor:
    """Execute SNAP processing via GPT."""
    
    def __init__(self, gpt_exe: str, ram_gb: int, logger: logging.Logger):
        self.gpt_exe = gpt_exe
        self.ram_gb = ram_gb
        self.logger = logger
    
    def process_pair(self, graph_file: Path, output_ifg: Path, timeout: int = 7200) -> bool:
        """
        Execute SNAP graph processing via GPT.
        SNAP 12+ CORRECT SYNTAX: gpt graph.xml -Pparameter=value
        """
        
        output_ifg.parent.mkdir(parents=True, exist_ok=True)
        
        # SNAP 12+ CORRECT command syntax: -Pparameter=value
        cmd = [self.gpt_exe, str(graph_file)]
        
        # Add parameters as -Pkey=value pairs
        params = {
            'output': str(output_ifg),
            'ramIndex': str(self.ram_gb)
        }
        
        for key, value in params.items():
            cmd.extend([f'-P{key}={value}'])
        
        self.logger.info(f"Executing SNAP 12 GPT: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"✓ Successfully processed interferogram: {output_ifg}")
                return True
            else:
                self.logger.error(f"✗ SNAP processing failed (return code: {result.returncode})")
                if result.stderr:
                    stderr_lines = result.stderr.split('\n')[:15]
                    self.logger.error(f"STDERR:\n" + '\n'.join(stderr_lines))
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ SNAP processing timeout (>{timeout}s)")
            return False
        except Exception as e:
            self.logger.error(f"✗ SNAP execution error: {e}")
            return False


class InSARBatchProcessor:
    """Orchestrate batch InSAR processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = config.logger
        self.results = []
    
    def run(self, slc_input):
        """Main processing workflow."""
        
        self.logger.info("="*80)
        self.logger.info("Sentinel-1 InSAR Batch Processing (SNAP 12+) - STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - SNAP GPT: {self.config.snap_exe}")
        self.logger.info(f"  - Output directory: {self.config.output_dir}")
        self.logger.info(f"  - Multilooking: {self.config.nlooks_range}x{self.config.nlooks_azimuth}")
        self.logger.info("="*80 + "\n")
        
        slc_path = Path(slc_input)
        
        if slc_path.is_file() and slc_path.suffix == '.txt':
            self.logger.info("Reading SLC list from text file...")
            with open(slc_path) as f:
                slc_files = [Path(line.strip()) for line in f if line.strip()]
            slc_dir = slc_files[0].parent if slc_files else slc_path.parent
        else:
            slc_dir = slc_path
            self.logger.info(f"Scanning directory for SLC files...")
        
        file_manager = SLCFileManager(slc_dir, self.logger)
        slc_files = file_manager.discover_slc_files()
        
        if not slc_files:
            self.logger.error("No SLC files found. Exiting.")
            return
        
        inventory = file_manager.build_inventory()
        
        if not inventory:
            self.logger.error("Could not build inventory from SLC files. Exiting.")
            return
        
        pair_gen = PairGenerator(inventory, self.config.min_bt_days, self.config.max_bt_days, self.logger)
        all_pairs = pair_gen.generate_pairs()
        
        if not all_pairs:
            self.logger.error("No valid pairs generated. Exiting.")
            return
        
        filtered_pairs = pair_gen.filter_by_coherence_expectation(land_cover_class='sparse_tundra')
        dem_for_snap = self.config.dem_mosaic_path
        
        graph_gen = SnapGraphGenerator(dem_for_snap, self.config.nlooks_range, 
                                       self.config.nlooks_azimuth, self.logger)
        snap_proc = SnapProcessor(self.config.snap_exe, self.config.ram_gb, self.logger)
        
        self.logger.info(f"\nProcessing {len(filtered_pairs)} interferometric pairs...")
        self.logger.info("-"*80 + "\n")
        
        for pair_idx, (ref_date, sec_date, bt, expected_gamma) in enumerate(filtered_pairs, 1):
            
            pair_name = f"{ref_date.strftime('%Y%m%d')}_{sec_date.strftime('%Y%m%d')}"
            
            self.logger.info(f"\n[{pair_idx}/{len(filtered_pairs)}] Processing pair: {pair_name}")
            self.logger.info(f"    Temporal baseline: {bt} days | Expected coherence: {expected_gamma:.3f}")
            
            ref_slc, sec_slc = pair_gen.get_slc_files_for_pair(ref_date, sec_date)
            
            if not ref_slc or not sec_slc:
                self.logger.error(f"    SLC files not found for this pair")
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'FAILED_MISSING_SLC',
                    'output_dir': None,
                })
                continue
            
            pair_dir = self.config.output_dir / pair_name
            ifg_dir = pair_dir / f"{pair_name}_ifg"
            
            graph_xml = graph_gen.create_insar_graph(str(ref_slc), str(sec_slc))
            graph_file = pair_dir / 'insar_graph.xml'
            graph_gen.save_graph(graph_xml, graph_file)
            
            self.logger.info(f"    Generated SNAP graph: {graph_file.name}")
            
            success = snap_proc.process_pair(graph_file, ifg_dir)
            
            if success:
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'SUCCESS',
                    'output_dir': str(ifg_dir),
                })
                self.logger.info(f"    Output: {ifg_dir}")
            else:
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'status': 'FAILED_PROCESSING',
                    'output_dir': str(ifg_dir),
                })
                self.logger.warning(f"    Processing failed (see log for details)")
        
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate and save processing summary."""
        
        if not self.results:
            self.logger.warning("No processing results to report")
            return
        
        df = pd.DataFrame(self.results)
        
        results_file = self.config.output_dir / 'processing_results.csv'
        df.to_csv(results_file, index=False)
        
        self.logger.info("\n" + "="*80)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("="*80)
        
        successful = df[df['status'] == 'SUCCESS']
        failed = df[df['status'] != 'SUCCESS']
        
        self.logger.info(f"\nTotal pairs processed: {len(df)}")
        self.logger.info(f"  Successful: {len(successful)}")
        self.logger.info(f"  Failed: {len(failed)}")
        
        if len(successful) > 0:
            self.logger.info(f"\nTop 10 pairs by expected coherence:\n")
            top_pairs = df.sort_values('expected_coherence', ascending=False).head(10)
            for idx, row in top_pairs.iterrows():
                status_icon = "OK" if row['status'] == 'SUCCESS' else "XX"
                self.logger.info(
                    f"  {status_icon} {row['pair']}: Bt={row['Bt_days']:3d}d, "
                    f"gamma_exp={row['expected_coherence']:.3f}, status={row['status']}"
                )
        
        self.logger.info(f"\nResults saved to: {results_file}")
        self.logger.info("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sentinel-1 InSAR Batch Processing with SNAP 12+')
    parser.add_argument('--slc-dir', help='Directory containing SLC .zip files')
    parser.add_argument('--slc-list', help='Text file with SLC file paths')
    parser.add_argument('--dem', required=True, help='DEM path')
    parser.add_argument('--min-bt', type=int, default=6, help='Min temporal baseline (days)')
    parser.add_argument('--max-bt', type=int, default=500, help='Max temporal baseline (days)')
    parser.add_argument('--output-dir', default='D:\\bcosta4\\DS-InSAR', help='Output directory')
    parser.add_argument('--nlooks-range', type=int, default=8, help='Multilook range')
    parser.add_argument('--nlooks-az', type=int, default=2, help='Multilook azimuth')
    parser.add_argument('--ram', type=int, default=8, help='Max heap size (GB)')
    
    args = parser.parse_args()
    
    if not args.slc_dir and not args.slc_list:
        parser.error("Must specify either --slc-dir or --slc-list")
    
    config = Config()
    config.min_bt_days = args.min_bt
    config.max_bt_days = args.max_bt
    config.output_dir = Path(args.output_dir)
    config.nlooks_range = args.nlooks_range
    config.nlooks_azimuth = args.nlooks_az
    config.ram_gb = args.ram
    
    dem_manager = DEMManager(args.dem, config.temp_dir, config.logger)
    try:
        dem_manager.parse_dem_paths()
        config.dem_paths = dem_manager.dem_paths
        dem_mosaic = dem_manager.create_vrt_mosaic()
        config.dem_mosaic_path = dem_mosaic
    except Exception as e:
        config.logger.error(f"Failed to initialize DEM(s): {e}")
        sys.exit(1)
    
    processor = InSARBatchProcessor(config)
    slc_input = args.slc_dir if args.slc_dir else args.slc_list
    processor.run(slc_input)


if __name__ == '__main__':
    main()
