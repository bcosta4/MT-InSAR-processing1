import os
import re
import csv
import numpy as np
from collections import defaultdict

def parse_envi_header(hdr_path):
    """
    Parse ENVI .hdr file and extract key parameters.
    Returns dict with 'samples', 'lines', 'data_type' (ENVI code), 'byte_order', 'interleave'
    """
    params = {}
    with open(hdr_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip().strip('{}')
            params[key] = value
    return params

def get_numpy_dtype(envi_type):
    """
    Map ENVI data type code to numpy dtype string (without endian).
    """
    dtype_map = {
        '1': 'u1',      # uint8
        '2': 'i2',      # int16
        '3': 'i4',      # int32
        '4': 'f4',      # float32
        '5': 'f8',      # float64
        '12': 'u2',     # uint16
        '13': 'u4',     # uint32
        '14': 'i8',     # int64
        '15': 'u8'      # uint64
    }
    return dtype_map.get(envi_type, 'f4')

def read_img_data(img_path, params):
    """
    Read .img binary data using params from header.
    Assumes BSQ interleave (common for single band coherence).
    """
    samples = int(params.get('samples', 0))
    lines = int(params.get('lines', 0))
    if samples == 0 or lines == 0:
        raise ValueError("Invalid dimensions in header")
    
    dtype_str = get_numpy_dtype(params.get('data type', '4'))
    byte_order = int(params.get('byte order', 0))
    endian = '<' if byte_order == 0 else '>'
    dtype = np.dtype(endian + dtype_str)
    
    expected_size = samples * lines * dtype.itemsize
    actual_size = os.path.getsize(img_path)
    if actual_size < expected_size:
        raise ValueError(f"File size mismatch: expected {expected_size}, got {actual_size}")
    
    with open(img_path, 'rb') as f:
        data = np.fromfile(f, dtype=dtype, count=samples * lines)
        if data.size != samples * lines:
            raise ValueError("Could not read full data")
        data = data.reshape((lines, samples))
    return data

def process_interferogram(ifg_folder):
    """
    Process one interferogram folder, find coh.img, compute mean/std coherence.
    Returns (date_pair, coh_path, mean_coh, std_coh) or None if failed.
    """
    data_folder = None
    for item in os.listdir(ifg_folder):
        if item.endswith('_ifg.data'):
            data_folder = os.path.join(ifg_folder, item)
            break
    if not data_folder or not os.path.isdir(data_folder):
        print(f"No valid data folder in {ifg_folder}")
        return None
    
    coh_files = [f for f in os.listdir(data_folder) if f.startswith('coh') and f.endswith('.img')]
    if not coh_files:
        print(f"No coh*.img files in {data_folder}")
        return None
    
    # Use first coh file (assume one main coherence per ifg)
    coh_img = os.path.join(data_folder, coh_files[0])
    coh_hdr = coh_img.replace('.img', '.hdr')
    
    if not os.path.exists(coh_hdr):
        print(f"No header for {coh_img}")
        return None
    
    try:
        params = parse_envi_header(coh_hdr)
        data = read_img_data(coh_img, params)
        
        # Coherence typically 0-1, mask <=0 or NaN
        valid_mask = (data > 0) & np.isfinite(data)
        if not np.any(valid_mask):
            print(f"No valid coherence data in {coh_img}")
            return None
        
        valid_data = data[valid_mask]
        mean_coh = np.mean(valid_data)
        std_coh = np.std(valid_data)
        
        date_pair = os.path.basename(ifg_folder)
        return (date_pair, coh_img, mean_coh, std_coh)
    except Exception as e:
        print(f"Error processing {coh_img}: {e}")
        return None

def main():
    root_dir = r'C:\bcosta\DS-InSAR'
    if not os.path.exists(root_dir):
        print(f"Root directory not found: {root_dir}")
        return
    
    results = []
    ifg_folders = [d for d in os.listdir(root_dir) 
                   if os.path.isdir(os.path.join(root_dir, d)) and 
                   re.match(r'\d{8}_\d{8}', d)]
    
    print(f"Found {len(ifg_folders)} interferogram folders")
    
    for ifg_folder_name in ifg_folders:
        ifg_path = os.path.join(root_dir, ifg_folder_name)
        result = process_interferogram(ifg_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid interferograms processed")
        return
    
    # Rank by mean coherence descending
    results.sort(key=lambda x: x[2], reverse=True)
    
    output_csv = os.path.join(root_dir, 'interferogram_coherence_ranking.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File_Path', 'Date_Pair', 'Mean_Coherence', 'Standard_Deviation'])
        for date_pair, file_path, mean_coh, std_coh in results:
            writer.writerow([file_path, date_pair, f'{mean_coh:.4f}', f'{std_coh:.4f}'])
    
    print(f"Results saved to {output_csv}")
    print("Top 5 highest coherence:")
    for i, (_, _, m, s) in enumerate(results[:5]):
        print(f"{i+1}. {m:.4f} ± {s:.4f}")

if __name__ == '__main__':
    main()
