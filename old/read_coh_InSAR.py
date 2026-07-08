# read_coh_snap.py
"""
Lê ficheiros de coerência InSAR do SNAP (formato ENVI .img/.hdr)
Estrutura esperada: D:\bcosta4\DS-InSAR\yyyymmdd_yyyymmdd\yyyymmdd_yyyymmdd_ifg.data\coh_IW*_VV_*.img
"""

import os
import glob
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
import sys

class SNAPCoherenceReader:
    def __init__(self, base_path):
        """
        Args:
            base_path: caminho para DS-InSAR (ex: D:\bcosta4\DS-InSAR\)
        """
        self.base_path = Path(base_path)
        self.results = []
        
    def find_coherence_pairs(self):
        """Encontra todos os pares interferométricos com dados de coerência"""
        pairs = []
        
        # Procura: yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd_ifg.data
        for pair_folder in self.base_path.glob("*_*"):
            if not pair_folder.is_dir():
                continue
            
            ifg_data_folder = pair_folder / f"{pair_folder.name}_ifg.data"
            
            if ifg_data_folder.exists():
                # Procura ficheiros coh_IW*_VV_*.img
                coh_files = list(ifg_data_folder.glob("coh_IW*_VV_*.img"))
                
                if coh_files:
                    pairs.append({
                        'pair_folder': pair_folder,
                        'ifg_data': ifg_data_folder,
                        'coh_files': coh_files,
                        'pair_name': pair_folder.name
                    })
        
        return sorted(pairs, key=lambda x: x['pair_name'])
    
    def read_envi_img(self, img_file):
        """
        Lê ficheiro ENVI .img (binary format)
        
        Returns:
            numpy array ou None se falhar
        """
        try:
            img_file = Path(img_file)
            hdr_file = img_file.with_suffix('.hdr')
            
            if not hdr_file.exists():
                print(f"  ⚠️  Header não encontrado: {hdr_file.name}")
                return None
            
            # Parse header
            metadata = self._parse_envi_hdr(hdr_file)
            
            samples = int(metadata.get('samples', 0))
            lines = int(metadata.get('lines', 0))
            
            if samples == 0 or lines == 0:
                print(f"  ⚠️  Dimensões inválidas: {lines}x{samples}")
                return None
            
            # Lê ficheiro binary (float32)
            data = np.fromfile(img_file, dtype=np.float32, count=samples*lines)
            
            if len(data) != samples * lines:
                print(f"  ⚠️  Tamanho ficheiro incorreto: esperado {samples*lines}, lido {len(data)}")
                return None
            
            data = data.reshape((lines, samples))
            return data
        
        except Exception as e:
            print(f"  ❌ Erro lendo {img_file.name}: {str(e)}")
            return None
    
    def _parse_envi_hdr(self, hdr_file):
        """Parse ficheiro .hdr ENVI (key = value format)"""
        metadata = {}
        try:
            with open(hdr_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith(';'):
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip().strip('{').strip('}')
                        metadata[key] = value
        except Exception as e:
            print(f"  Erro parsing {hdr_file.name}: {e}")
        
        return metadata
    
    def extract_dates(self, pair_name, coh_filename):
        """
        Extrai datas do pair_name (yyyymmdd_yyyymmdd)
        ou do coh filename (coh_IW3_VV_08Sep2023_28Jul2024.img)
        """
        try:
            # Tenta primeiro do coh filename
            # Format: coh_IW3_VV_08Sep2023_28Jul2024.img
            parts = coh_filename.replace('coh_', '').replace('.img', '').split('_')
            if len(parts) >= 4:
                try:
                    master_str = parts[-2]  # "08Sep2023"
                    slave_str = parts[-1]   # "28Jul2024"
                    
                    # Parse datas
                    master_date = datetime.strptime(master_str, '%d%b%Y')
                    slave_date = datetime.strptime(slave_str, '%d%b%Y')
                    
                    baseline = (slave_date - master_date).days
                    
                    return {
                        'master': master_date.strftime('%Y-%m-%d'),
                        'slave': slave_date.strftime('%Y-%m-%d'),
                        'baseline_days': baseline
                    }
                except:
                    pass
            
            # Fallback: usa pair_name
            parts = pair_name.split('_')
            if len(parts) >= 2:
                master = parts[0]
                slave = parts[1]
                master_date = datetime.strptime(master, '%Y%m%d')
                slave_date = datetime.strptime(slave, '%Y%m%d')
                baseline = (slave_date - master_date).days
                
                return {
                    'master': master_date.strftime('%Y-%m-%d'),
                    'slave': slave_date.strftime('%Y-%m-%d'),
                    'baseline_days': baseline
                }
        except Exception as e:
            pass
        
        return {
            'master': 'Unknown',
            'slave': 'Unknown',
            'baseline_days': -1
        }
    
    def process_all(self, output_csv='coherence_stats.csv'):
        """Processa todos os pares InSAR"""
        pairs = self.find_coherence_pairs()
        
        print(f"\n{'='*80}")
        print(f"SNAP InSAR Coherence Reader")
        print(f"{'='*80}")
        print(f"Base path: {self.base_path}")
        print(f"Pares encontrados: {len(pairs)}\n")
        
        if not pairs:
            print("❌ Nenhum par InSAR encontrado!")
            print("   Estrutura esperada: <base>/yyyymmdd_yyyymmdd/yyyymmdd_yyyymmdd_ifg.data/coh_*.img")
            return None
        
        for pair_idx, pair_info in enumerate(pairs, 1):
            pair_name = pair_info['pair_name']
            ifg_data = pair_info['ifg_data']
            coh_files = pair_info['coh_files']
            
            print(f"[{pair_idx}/{len(pairs)}] {pair_name}")
            
            for coh_file in coh_files:
                coh_filename = coh_file.name
                print(f"  → {coh_filename}")
                
                # Lê dados
                data = self.read_envi_img(coh_file)
                if data is None:
                    continue
                
                # Extrai datas
                dates = self.extract_dates(pair_name, coh_filename)
                
                # Calcula estatísticas (coerência válida: 0 < coh <= 1)
                coh_valid = data[(data > 0) & (data <= 1)]
                
                if len(coh_valid) == 0:
                    print(f"    ⚠️  Sem coerência válida")
                    continue
                
                stats = {
                    'pair': pair_name,
                    'iw_band': coh_filename.split('_')[1],  # IW3 etc
                    'master_date': dates['master'],
                    'slave_date': dates['slave'],
                    'baseline_days': dates['baseline_days'],
                    'coh_mean': float(np.mean(coh_valid)),
                    'coh_median': float(np.median(coh_valid)),
                    'coh_std': float(np.std(coh_valid)),
                    'coh_min': float(np.min(coh_valid)),
                    'coh_max': float(np.max(coh_valid)),
                    'valid_pixels': int(len(coh_valid)),
                    'total_pixels': int(data.size),
                    'coverage_pct': float(100 * len(coh_valid) / data.size)
                }
                
                self.results.append(stats)
                
                # Print resumo
                print(f"    ✓ Mean: {stats['coh_mean']:.4f} | "
                      f"Median: {stats['coh_median']:.4f} | "
                      f"Pixels: {stats['valid_pixels']:,}/{stats['total_pixels']:,} "
                      f"({stats['coverage_pct']:.1f}%)")
        
        # Exporta CSV
        if self.results:
            df = pd.DataFrame(self.results)
            df = df.sort_values(['baseline_days', 'pair'])
            df.to_csv(output_csv, index=False)
            
            print(f"\n{'='*80}")
            print(f"✓ Exportado: {output_csv}")
            print(f"{'='*80}")
            print(f"\nResumo:")
            print(f"  Total pares com dados: {len(df)}")
            print(f"  Coerência média (global): {df['coh_mean'].mean():.4f} ± {df['coh_mean'].std():.4f}")
            print(f"  Coerência range: {df['coh_mean'].min():.4f} - {df['coh_mean'].max():.4f}")
            print(f"  Baseline temporal (dias): {int(df['baseline_days'].min())} - {int(df['baseline_days'].max())}")
            print(f"  Cobertura média (pixels válidos): {df['coverage_pct'].mean():.1f}%\n")
            
            return df
        else:
            print("\n❌ Nenhum dado processado com sucesso!")
            return None


# ============== MAIN ==============
if __name__ == '__main__':
    # Ajusta este caminho conforme necessário
    BASE_PATH = r"D:\bcosta4\DS-InSAR"
    
    # Alternativas:
    # BASE_PATH = r"E:\InSAR_data"  # se em outro disco
    # BASE_PATH = "/home/user/insar/DS-InSAR"  # Linux
    
    # Verifica se existe
    if not Path(BASE_PATH).exists():
        print(f"❌ Caminho não encontrado: {BASE_PATH}")
        sys.exit(1)
    
    # Cria leitor
    reader = SNAPCoherenceReader(BASE_PATH)
    
    # Processa
    df = reader.process_all(output_csv='coherence_stats_DS-InSAR.csv')
    
    # Print DataFrame completo (opcional)
    if df is not None:
        print("\nDados completos:")
        print(df.to_string(index=False))
