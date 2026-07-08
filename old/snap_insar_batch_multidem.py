#!/usr/bin/env python3
"""
Automated Sentinel-1 InSAR Batch Processing with SNAP
======================================================

Processes Sentinel-1 SLC stacks to generate differential interferograms
and coherence products using SNAP's Graph Processing Tool (GPT).

Optimized for: Arctic Coastal Permafrost Monitoring (Tuktoyaktuk Peninsula)
- Subswath: IW3 (nadir-looking, best for subsidence)
- Polarization: VV only
- Multilooking: 8 range × 2 azimuth
- Filtering: Goldstein (coherence threshold 0.3)
- Baseline: 6-500 days
- Processing: ALL pairs (no limit)
- DEM Support: Multiple Copernicus tiles with VRT mosaic
- FIXED: Sentinel-1 filename date parsing

Author: InSAR Processing Group
Date: January 2026

Usage:
  # Multiple DEMs (automatic VRT mosaic):
  python snap_insar_batch_fixed.py \\
    --slc-dir "C:\\bcosta\\S1_raw" \\
    --dem "C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W134_00_DEM.tif,C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W133_00_DEM.tif,C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W132_00_DEM.tif" \\
    --min-bt 6 \\
    --max-bt 500 \\
    --nlooks-range 8 \\
    --nlooks-az 2 \\
    --coherence-threshold 0.3 \\
    --ram 8 \\
    --output-dir "C:\\bcosta\\S1_insar_results"

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
        self.output_dir = Path('./results_insar')
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
        snap_paths = [
            Path('C:\\Program Files\\esa-snap\\bin\\snap64.exe'),
            Path('C:\\Program Files\\snap\\bin\\snap64.exe'),
            Path.home() / 'snap' / 'bin' / 'gpt',
            Path.home() / 'snap' / 'bin' / 'gpt.exe',
            Path('/opt/snap/bin/gpt'),
            Path('/usr/local/snap/bin/gpt'),
        ]
        
        for path in snap_paths:
            if path.exists():
                self.logger_console(f"Found SNAP GPT at: {path}")
                return str(path)
        
        snap_env = os.environ.get('SNAP_EXE')
        if snap_env and Path(snap_env).exists():
            return snap_env
        
        try:
            result = subprocess.run(['which', 'gpt'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        raise FileNotFoundError(
            "SNAP GPT executable not found. "
            "Please install SNAP or set SNAP_EXE environment variable.\n"
            "Expected at: C:\\Program Files\\esa-snap\\bin\\snap64.exe"
        )
    
    def logger_console(self, msg: str):
        """Simple console log before logger initialized."""
        print(f"[INFO] {msg}")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging to file and console."""
        logger = logging.getLogger('InSARBatch')
        logger.setLevel(logging.DEBUG)
        
        logger.handlers.clear()
        
        fh = logging.FileHandler(self.log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger


class DEMManager:
    """Manages single or multiple DEM files with automatic VRT mosaic creation."""
    
    def __init__(self, dem_input: str, temp_dir: Path, logger: logging.Logger):
        """
        Args:
            dem_input: Single DEM path or comma-separated paths
            temp_dir: Temporary directory for VRT files
            logger: Logger instance
        """
        self.dem_input = dem_input
        self.temp_dir = temp_dir
        self.logger = logger
        self.dem_paths = []
        self.dem_mosaic_path = None
    
    def parse_dem_paths(self) -> List[Path]:
        """
        Parse single or comma-separated DEM paths.
        
        Returns:
            List of validated DEM Path objects
        """
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
        """
        Create VRT mosaic from multiple DEM files.
        If single DEM, returns it directly.
        
        Returns:
            Path to DEM or VRT mosaic
        """
        if len(self.dem_paths) == 1:
            self.logger.info(f"Single DEM detected: {self.dem_paths[0].name}")
            self.dem_mosaic_path = self.dem_paths[0]
            return self.dem_paths[0]
        
        self.logger.info(f"Creating VRT mosaic from {len(self.dem_paths)} DEM tiles...")
        
        vrt_path = self.temp_dir / 'copernicus_dem_mosaic.vrt'
        
        gdal_cmd = [
            'gdalbuildvrt',
            '-separate',
            '-overwrite',
            str(vrt_path),
            *[str(p) for p in self.dem_paths]
        ]
        
        try:
            result = subprocess.run(
                gdal_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.logger.info(f"VRT mosaic created: {vrt_path}")
                self.dem_mosaic_path = vrt_path
                return vrt_path
            else:
                self.logger.error(f"gdalbuildvrt failed: {result.stderr}")
                raise RuntimeError(f"Failed to create DEM mosaic: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            self.logger.error("VRT creation timeout")
            raise
        except FileNotFoundError:
            self.logger.error(
                "gdalbuildvrt not found. Ensure GDAL is installed and in PATH. "
                "SNAP includes GDAL; try running from SNAP console."
            )
            raise
    
    def get_dem_for_snap(self) -> Path:
        """
        Get DEM path for SNAP processing (single file or VRT mosaic).
        
        Returns:
            Path to DEM or VRT
        """
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
        """
        Discover SLC files in directory.
        
        Args:
            pattern: Glob pattern for SLC files
        
        Returns:
            Sorted list of SLC file paths
        """
        self.logger.info(f"Discovering SLC files in {self.slc_dir}")
        
        slc_files = sorted(self.slc_dir.glob(pattern))
        
        if not slc_files:
            self.logger.warning(f"No SLC files found matching {pattern}")
            return []
        
        self.logger.info(f"Found {len(slc_files)} SLC files")
        self.slc_files = slc_files
        
        return slc_files
    
    def extract_acquisition_date(self, slc_file: Path) -> Optional[datetime]:
        """
        Extract acquisition date from SLC filename.
        
        Sentinel-1 naming: S1[AB]_IW_SLC__1SDV_20230115T001234_20230115T001302_046945_059FEE_XXXX.zip
        Parts: [0]S1A [1]IW [2]SLC [3](empty) [4]1SDV [5]YYYYMMDDTHHMMSS [6]...
        
        Args:
            slc_file: Path to SLC file
        
        Returns:
            datetime object of acquisition or None
        """
        filename = slc_file.stem
        
        parts = filename.split('_')
        # FIXED: parts[5] contains YYYYMMDDTHHMMSS, not parts[4]
        if len(parts) >= 6:
            date_str = parts[5]  # YYYYMMDDTHHMMSS (was parts[4] which is '1SDV')
            try:
                acq_date = datetime.strptime(date_str[:8], '%Y%m%d')
                return acq_date
            except ValueError as e:
                self.logger.error(f"Cannot extract date from {filename}: {e}")
                return None
        
        return None
    
    def build_inventory(self) -> Dict:
        """
        Build inventory of SLC files with acquisition dates.
        
        Returns:
            Dict {date: [slc_paths]}
        """
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
        """
        Args:
            inventory: Dict {date: [slc_files]}
            min_bt: Minimum temporal baseline (days)
            max_bt: Maximum temporal baseline (days)
            logger: Logger instance
        """
        self.inventory = inventory
        self.min_bt = min_bt
        self.max_bt = max_bt
        self.logger = logger
        self.valid_pairs = []
    
    def generate_pairs(self) -> List[Tuple]:
        """
        Generate all valid interferometric pairs.
        Reference image always precedes secondary chronologically.
        
        Returns:
            List of (reference_date, secondary_date, bt_days) tuples
        """
        dates = sorted(self.inventory.keys())
        pairs = []
        
        for i, ref_date in enumerate(dates):
            for sec_date in dates[i+1:]:
                bt = (sec_date - ref_date).days
                
                if self.min_bt <= bt <= self.max_bt:
                    pairs.append((ref_date, sec_date, bt))
        
        self.valid_pairs = pairs
        
        self.logger.info(
            f"Generated {len(pairs)} valid pairs "
            f"(Bt: {self.min_bt}-{self.max_bt} days)"
        )
        
        return pairs
    
    def get_slc_files_for_pair(self, ref_date: datetime, sec_date: datetime) -> Tuple:
        """
        Get SLC file paths for interferometric pair.
        
        Args:
            ref_date: Reference acquisition date
            sec_date: Secondary acquisition date
        
        Returns:
            (ref_slc_path, sec_slc_path) or (None, None)
        """
        ref_files = self.inventory.get(ref_date, [])
        sec_files = self.inventory.get(sec_date, [])
        
        if not ref_files or not sec_files:
            return None, None
        
        return ref_files[0], sec_files[0]
    
    def filter_by_coherence_expectation(self, land_cover_class: str = 'sparse_tundra') -> List[Tuple]:
        """
        Pre-filter pairs by expected coherence (using baseline criteria).
        
        Args:
            land_cover_class: 'sparse_tundra', 'urban', 'forest', etc.
        
        Returns:
            Filtered pairs list with expected coherence
        """
        tau_values = {
            'sparse_tundra': 40,
            'urban': 180,
            'bare_rock': 120,
            'grassland': 30,
            'forest': 15,
        }
        
        tau = tau_values.get(land_cover_class, 40)
        
        filtered_pairs = []
        
        for ref_date, sec_date, bt in self.valid_pairs:
            gamma_0 = 0.58  # For sparse tundra at 12 days
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
    
    def __init__(self, dem_path: Path, nlooks_range: int, nlooks_azimuth: int,
                 coherence_threshold: float, logger: logging.Logger):
        """
        Args:
            dem_path: Path to Copernicus 30m DEM or VRT mosaic
            nlooks_range: Multilook range factor
            nlooks_azimuth: Multilook azimuth factor
            coherence_threshold: Goldstein coherence threshold
            logger: Logger instance
        """
        self.dem_path = str(dem_path)
        self.nlooks_range = nlooks_range
        self.nlooks_azimuth = nlooks_azimuth
        self.coherence_threshold = coherence_threshold
        self.logger = logger
    
    def create_insar_graph(self, ref_slc: str, sec_slc: str) -> str:
        """
        Generate SNAP processing graph for interferogram generation.
        
        IW3 subswath, VV polarization only.
        
        Args:
            ref_slc: Path to reference SLC
            sec_slc: Path to secondary SLC
        
        Returns:
            XML graph string
        """
        
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
    
    <!-- Step 5: Back-Geocoding (coregistration with DEM mosaic) -->
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
    
    <!-- Step 9: Phase Filtering (Goldstein) with coherence threshold {self.coherence_threshold} -->
    <node id="filter">
        <operator>Goldstein-Phase-Filtering</operator>
        <sources>
            <sourceProduct refid="deburst"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <filterSize>32</filterSize>
            <useCoherence>true</useCoherence>
            <coherenceThreshold>{self.coherence_threshold}</coherenceThreshold>
            <useCoherenceWeighting>true</useCoherenceWeighting>
            <exponent>0.6</exponent>
            <blockSize>64</blockSize>
        </parameters>
    </node>
    
    <!-- Step 10: Multilooking ({self.nlooks_range}x{self.nlooks_azimuth}) -->
    <node id="multilook">
        <operator>Multilook</operator>
        <sources>
            <sourceProduct refid="filter"/>
        </sources>
        <parameters class="com.bc.ceres.binding.dom.XppDomElement">
            <nRgLooks>{self.nlooks_range}</nRgLooks>
            <nAzLooks>{self.nlooks_azimuth}</nAzLooks>
            <outputIntensity>false</outputIntensity>
            <greyScaleOutput>false</greyScaleOutput>
        </parameters>
    </node>
    
    <!-- Step 11: Terrain Correction -->
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
            <pixelSpacingInDegree>0</pixelSpacingInDegree>
            <mapProjection>WGS84(DD)</mapProjection>
            <alignToStandardGrid>false</alignToStandardGrid>
            <standardGridOriginX>0.0</standardGridOriginX>
            <standardGridOriginY>0.0</standardGridOriginY>
            <standardGridSpacingX>1.0</standardGridSpacingX>
            <standardGridSpacingY>1.0</standardGridSpacingY>
            <noDataValueAtSea>true</noDataValueAtSea>
            <saveDEM>false</saveDEM>
            <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
            <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
            <saveSelectedSourceBand>true</saveSelectedSourceBand>
            <outputComplex>false</outputComplex>
            <outputToUTC>false</outputToUTC>
            <scale>1.0</scale>
            <nodataValueAtSea>true</nodataValueAtSea>
            <reGridMethod>false</reGridMethod>
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
            <clearCacheAfterRowWrite>true</clearCacheAfterRowWrite>
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
        """
        Args:
            gpt_exe: Path to SNAP gpt executable
            ram_gb: RAM to allocate (GB)
            logger: Logger instance
        """
        self.gpt_exe = gpt_exe
        self.ram_gb = ram_gb
        self.logger = logger
    
    def process_pair(self, graph_file: Path, output_ifg: Path, 
                     timeout: int = 7200) -> bool:
        """
        Execute SNAP graph processing.
        
        Args:
            graph_file: Path to SNAP graph XML
            output_ifg: Output interferogram path
            timeout: Processing timeout (seconds, default 2 hours)
        
        Returns:
            True if successful, False otherwise
        """
        
        output_ifg.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            self.gpt_exe,
            str(graph_file),
            '-t', str(output_ifg),
            '-c', f'{self.ram_gb}G'
        ]
        
        self.logger.info(f"Executing SNAP: {' '.join(cmd)}")
        self.logger.debug(f"Timeout: {timeout}s, RAM: {self.ram_gb}GB")
        
        try:
            result = subprocess.run(
                cmd,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.logger.info(f"Successfully processed interferogram")
                return True
            else:
                self.logger.error(f"SNAP processing failed (return code: {result.returncode})")
                if result.stderr:
                    self.logger.error(f"STDERR:\n{result.stderr[:500]}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"SNAP processing timeout (>{timeout}s)")
            return False
        except Exception as e:
            self.logger.error(f"SNAP execution error: {e}")
            return False


class CoherenceAnalyzer:
    """Extract and analyze coherence from processed interferograms."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract_coherence_stats(self, ifg_dir: Path) -> Optional[Dict]:
        """
        Extract mean and statistical coherence from interferogram product.
        
        Args:
            ifg_dir: Directory containing SNAP BEAM-DIMAP output
        
        Returns:
            Dict with coherence statistics or None
        """
        
        hdr_files = list(ifg_dir.glob('*.hdr'))
        
        if not hdr_files:
            self.logger.warning(f"No BEAM-DIMAP header found in {ifg_dir}")
            return None
        
        stats = {
            'coherence_file': str(hdr_files[0]),
            'mean_coherence': None,
            'std_coherence': None,
            'median_coherence': None,
            'percent_coherent_0_4': None,
        }
        
        self.logger.debug(f"Coherence extraction placeholder for {hdr_files[0]}")
        
        return stats


class InSARBatchProcessor:
    """Orchestrate batch InSAR processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = config.logger
        self.results = []
    
    def run(self, slc_input):
        """
        Main processing workflow.
        
        Args:
            slc_input: Directory path or text file listing SLC files
        """
        
        self.logger.info("="*80)
        self.logger.info("Sentinel-1 InSAR Batch Processing - STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Configuration:")
        self.logger.info(f"  - SNAP GPT: {self.config.snap_exe}")
        self.logger.info(f"  - DEM(s): {len(self.config.dem_paths)} file(s)")
        for i, dem_path in enumerate(self.config.dem_paths, 1):
            self.logger.info(f"      {i}. {dem_path.name}")
        if self.config.dem_mosaic_path:
            self.logger.info(f"  - DEM mosaic: {self.config.dem_mosaic_path}")
        self.logger.info(f"  - Temporal baseline: {self.config.min_bt_days}-{self.config.max_bt_days} days")
        self.logger.info(f"  - Subswath: IW3, Polarization: VV")
        self.logger.info(f"  - Multilooking: {self.config.nlooks_range}x{self.config.nlooks_azimuth}")
        self.logger.info(f"  - Goldstein coherence threshold: {self.config.coherence_threshold}")
        self.logger.info(f"  - RAM allocation: {self.config.ram_gb}GB")
        self.logger.info(f"  - Output directory: {self.config.output_dir}")
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
        
        pair_gen = PairGenerator(
            inventory,
            self.config.min_bt_days,
            self.config.max_bt_days,
            self.logger
        )
        
        all_pairs = pair_gen.generate_pairs()
        
        if not all_pairs:
            self.logger.error("No valid pairs generated. Exiting.")
            return
        
        filtered_pairs = pair_gen.filter_by_coherence_expectation(land_cover_class='sparse_tundra')
        
        dem_for_snap = self.config.dem_mosaic_path
        
        graph_gen = SnapGraphGenerator(
            dem_for_snap,
            self.config.nlooks_range,
            self.config.nlooks_azimuth,
            self.config.coherence_threshold,
            self.logger
        )
        snap_proc = SnapProcessor(self.config.snap_exe, self.config.ram_gb, self.logger)
        coh_analyzer = CoherenceAnalyzer(self.logger)
        
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
                    'measured_coherence': None,
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
                coh_stats = coh_analyzer.extract_coherence_stats(ifg_dir)
                
                measured_gamma = coh_stats.get('mean_coherence') if coh_stats else None
                
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'measured_coherence': measured_gamma,
                    'status': 'SUCCESS',
                    'output_dir': str(ifg_dir),
                })
                
                self.logger.info(f"    Output: {ifg_dir}")
            else:
                self.results.append({
                    'pair': pair_name,
                    'Bt_days': bt,
                    'expected_coherence': expected_gamma,
                    'measured_coherence': None,
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
            self.logger.info(f"\nTop 15 pairs by expected coherence:\n")
            top_pairs = df.sort_values('expected_coherence', ascending=False).head(15)
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
    
    parser = argparse.ArgumentParser(
        description='Sentinel-1 InSAR Batch Processing with SNAP (Arctic Optimized)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
EXAMPLES:

  Multiple DEMs (automatic VRT mosaic - Tuktoyaktuk coverage):
    python snap_insar_batch_fixed.py \\
      --slc-dir "C:\\bcosta\\S1_raw" \\
      --dem "C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W134_00_DEM.tif,C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W133_00_DEM.tif,C:\\bcosta\\S1_raw\\Copernicus_DSM_COG_10_N69_00_W132_00_DEM.tif" \\
      --min-bt 6 \\
      --max-bt 500 \\
      --nlooks-range 8 \\
      --nlooks-az 2 \\
      --coherence-threshold 0.3 \\
      --ram 8 \\
      --output-dir "C:\\bcosta\\S1_insar_results"

        '''
    )
    
    parser.add_argument(
        '--slc-dir',
        help='Directory containing SLC .zip files'
    )
    parser.add_argument(
        '--slc-list',
        help='Text file with SLC file paths (one per line)'
    )
    parser.add_argument(
        '--dem',
        required=True,
        help='Single DEM path or comma-separated paths for VRT mosaic'
    )
    parser.add_argument(
        '--min-bt',
        type=int,
        default=6,
        help='Minimum temporal baseline in days (default: 6)'
    )
    parser.add_argument(
        '--max-bt',
        type=int,
        default=500,
        help='Maximum temporal baseline in days (default: 500)'
    )
    parser.add_argument(
        '--output-dir',
        default='./results_insar',
        help='Output directory for results (default: ./results_insar)'
    )
    parser.add_argument(
        '--nlooks-range',
        type=int,
        default=8,
        help='Multilook range factor (default: 8)'
    )
    parser.add_argument(
        '--nlooks-az',
        type=int,
        default=2,
        help='Multilook azimuth factor (default: 2)'
    )
    parser.add_argument(
        '--coherence-threshold',
        type=float,
        default=0.3,
        help='Goldstein filter coherence threshold (default: 0.3)'
    )
    parser.add_argument(
        '--ram',
        type=int,
        default=8,
        help='RAM to allocate per SNAP process in GB (default: 8)'
    )
    
    args = parser.parse_args()
    
    if not args.slc_dir and not args.slc_list:
        parser.error("Must specify either --slc-dir or --slc-list")
    
    config = Config()
    config.min_bt_days = args.min_bt
    config.max_bt_days = args.max_bt
    config.output_dir = Path(args.output_dir)
    config.nlooks_range = args.nlooks_range
    config.nlooks_azimuth = args.nlooks_az
    config.coherence_threshold = args.coherence_threshold
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
