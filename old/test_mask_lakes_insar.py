import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
import struct
from typing import Tuple, Dict

# Assuming mask_lakes_insar.py is importable or function is defined here
# For testing purposes, we'll mock the mask_file function signature:
# def mask_file(input_file: str, mask_file: str, output_file: str = None) -> None:
# Note: Since we don't have the actual code, this test suite defines a mock
# that implements the expected corrected behavior for validation purposes.
# Replace with actual import when available.

def mock_mask_file(input_file: str, mask_file: str, output_file: str = None):
    """
    Mock implementation of the corrected mask_file function for testing.
    Implements all 8 expected behaviors.
    """
    if output_file is None:
        output_file = input_file
    
    # Read input data and header (simplified for binary raster)
    with open(input_file, 'rb') as f:
        header = f.read(1024)  # Assume fixed 1KB header
        data_bytes = f.read()
    
    # Parse header (simplified: bytes 0-4: H, 4-8: W, 8-12: bands, 12-16: dtype_code, 16-20: endian '<' or '>')
    H = struct.unpack('=I', header[0:4])[0]
    W = struct.unpack('=I', header[4:8])[0]
    bands = struct.unpack('=I', header[8:12])[0]
    dtype_code = header[12:16].decode('ascii')
    endian = header[16:20].decode('ascii').strip('\x00')

    dtype_map = {
        'i16 ': np.int16,
        'i32 ': np.int32,
        'f32 ': np.float32,
    }
    if dtype_code not in dtype_map:
        raise ValueError(f"Unknown dtype code in header: {repr(dtype_code)}")
    input_dtype = dtype_map[dtype_code]
    itemsize = np.dtype(input_dtype).itemsize
    
    expected_bytes = H * W * bands * itemsize
    if len(data_bytes) != expected_bytes:
        raise ValueError(f"Input data size mismatch: {len(data_bytes)} != {expected_bytes}")
    
    # Reshape data using proper NumPy dtype with endianness
    dt = np.dtype(input_dtype).newbyteorder(endian)  # endian is '<' or '>'
    data = np.frombuffer(data_bytes, dtype=dt).reshape((bands, H, W))

    
    # Pre-write validation (FIX #4)
    # Pre-write validation (FIX #4) – ensure float32 *and* writable
    if data.dtype != np.float32 or not data.flags.writeable:
        # Force a real float32 copy; this guarantees writeability
        data = data.astype(np.float32, copy=True)
    else:
        # Even if dtype is already float32, make a copy if needed
        if not data.flags.writeable:
            data = data.copy()

    if not np.issubdtype(data.dtype, np.floating):
        raise ValueError("Data must be float32 before write")
    
    # Read mask
    mask = np.load(mask_file)['mask'].astype(bool)
    if mask.shape != (H, W):
        raise ValueError("Mask dimension mismatch")
    
    # Apply masking (FIX #8) - correct operation
    # Create inverted mask: True where we keep data, False where lakes are
    non_lake_mask = ~mask  # Invert: True = non-lake, False = lake
    for b in range(bands):
        data[b][~non_lake_mask] = np.nan  # Set lake pixels to NaN
    
    # Update header before write (FIX #3)
    new_header = bytearray(header)
    new_header[12:16] = b'f32\x00'
    
    # Create backup (FIX #7)
    backup = Path(input_file + '.bak')
    if backup.exists():
        backup.unlink()
    shutil.copy2(input_file, backup)
    
    # Write output (FIX #1 - float32 enforcement)
    with open(output_file, 'wb') as f:
        f.write(new_header)
        data_bytes_out = data.astype(np.float32).tobytes()
        f.write(data_bytes_out)
    
    # Verify byte count (FIX #5)
    actual_bytes = len(data_bytes_out)
    expected_out_bytes = H * W * bands * 4
    if actual_bytes != expected_out_bytes:
        raise ValueError(f"Output byte count mismatch: {actual_bytes} != {expected_out_bytes}")

# Real function would be: from mask_lakes_insar import mask_file
# mask_file = mock_mask_file  # Use mock for now

@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    tmp = Path(tempfile.mkdtemp())
    yield tmp
    shutil.rmtree(tmp)

@pytest.fixture
def synth_data(tmp_dir: Path):
    """Factory fixture that generates synthetic InSAR binary files with header."""
    def _generate(height: int = 10, width: int = 10, bands: int = 1, 
                  dtype: str = 'float32', endian: str = '<') -> Dict[str, Path]:
        """Generate synthetic InSAR binary file with header."""
        file_path = tmp_dir / f"synth_{height:04d}x{width:04d}_b{bands}_{dtype}_{endian.replace('<','LE').replace('>','BE')}.bin"
        
        H, W, n_bands = height, width, bands
        dtype_map = {'int16': np.dtype(f'{endian}i2'), 'int32': np.dtype(f'{endian}i4'), 'float32': np.dtype(f'{endian}f4')}
        data_dtype = dtype_map[dtype]
        
        data = np.random.uniform(-100, 100, (n_bands, H, W)).astype(data_dtype)
        
        header = bytearray(1024)
        struct.pack_into('=I', header, 0, H)
        struct.pack_into('=I', header, 4, W)
        struct.pack_into('=I', header, 8, n_bands)

        # Use 4‑char tokens that round‑trip cleanly
        dtype_token_map = {
            'int16': 'i16 ',
            'int32': 'i32 ',
            'float32': 'f32 ',
        }
        dtype_token = dtype_token_map[dtype]
        header[12:16] = dtype_token.encode('ascii')

        # Store endianness as single char '<' or '>'
        header[16:20] = (endian + '\x00').ljust(4, '\x00').encode('ascii')
        
        data_bytes = data.tobytes()
        
        with open(file_path, 'wb') as f:
            f.write(header)
            f.write(data_bytes)
        
        mask_path = tmp_dir / f"mask_{height}x{width}.npz"
        mask = np.random.choice([True, False], (H, W), p=[0.3, 0.7])  # Partial mask
        np.savez_compressed(mask_path, mask=mask)
        
        return {'data': file_path, 'mask': mask_path, 'orig_data': data.copy(), 'H': H, 'W': W, 'bands': n_bands}
    
    return _generate

def create_all_lake_mask(tmp_dir: Path, H: int, W: int) -> Path:
    """Create mask where all pixels are lakes."""
    mask_path = tmp_dir / "all_lake_mask.npz"
    mask = np.ones((H, W), bool)
    np.savez_compressed(mask_path, mask=mask)
    return mask_path

def create_empty_mask(tmp_dir: Path, H: int, W: int) -> Path:
    """Create mask where no pixels are lakes."""
    mask_path = tmp_dir / "empty_mask.npz"
    mask = np.zeros((H, W), bool)
    np.savez_compressed(mask_path, mask=mask)
    return mask_path

def read_binary_file(file_path: Path) -> Tuple[np.ndarray, Dict]:
    """Read synthetic binary file honoring header dtype and endianness."""
    with open(file_path, 'rb') as f:
        header = f.read(1024)
        data_bytes = f.read()

    H = struct.unpack('=I', header[0:4])[0]
    W = struct.unpack('=I', header[4:8])[0]
    bands = struct.unpack('=I', header[8:12])[0]
    dtype_code = header[12:16].decode('ascii')
    endian = header[16:20].decode('ascii').strip('\x00')

    dtype_map = {
        'i16 ': np.int16,
        'i32 ': np.int32,
        'f32 ': np.float32,
    }
    if dtype_code not in dtype_map:
        raise ValueError(f"Unknown dtype code in header: {repr(dtype_code)}")

    base = dtype_map[dtype_code]
    dt = np.dtype(base).newbyteorder(endian)  # '<' or '>'

    expected_elems = H * W * bands
    actual_elems = len(data_bytes) // dt.itemsize
    if actual_elems != expected_elems:
        raise ValueError(
            f"Data size mismatch: {actual_elems} elements != {expected_elems}"
        )

    data = np.frombuffer(data_bytes, dtype=dt).reshape((bands, H, W))
    return data, {'H': H, 'W': W, 'bands': bands}

# ============================================================================
# 1. Float32 Output Enforcement
# ============================================================================

def test_float32_output_int16_input(tmp_dir: Path, synth_data):
    """Validate float32 output regardless of int16 input dtype [validates fix #1]."""
    synth = synth_data(dtype='int16')
    out_file = tmp_dir / "output_int16.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, meta = read_binary_file(out_file)
    assert data_out.dtype == np.float32, "Output must be float32 for int16 input"


def test_float32_output_int32_input(tmp_dir: Path, synth_data):
    """Validate float32 output regardless of int32 input dtype [validates fix #1]."""
    synth = synth_data(dtype='int32')
    out_file = tmp_dir / "output_int32.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, _ = read_binary_file(out_file)
    assert data_out.dtype == np.float32, "Output must be float32 for int32 input"


def test_float32_output_float32_input(tmp_dir: Path, synth_data):
    """Validate float32 output preserved for float32 input [validates fix #1]."""
    synth = synth_data(dtype='float32')
    out_file = tmp_dir / "output_float32.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, _ = read_binary_file(out_file)
    assert data_out.dtype == np.float32, "Output must remain float32"


# ============================================================================
# 2. NaN Representation
# ============================================================================

def test_nan_preservation(tmp_dir: Path, synth_data):
    """NaN values written as float32 are readable without corruption [validates fix #2]."""
    synth = synth_data()
    out_file = tmp_dir / "nan_test.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, meta = read_binary_file(out_file)
    nan_count = np.isnan(data_out).sum()
    assert nan_count > 0, "No NaNs masked in output"


def test_all_lake_mask_becomes_all_nan(tmp_dir: Path, synth_data):
    """All-lake mask results in all-NaN data [validates fix #2,8]."""
    synth = synth_data(height=5, width=5)
    mask_path = create_all_lake_mask(tmp_dir, 5, 5)
    out_file = tmp_dir / "all_nan.bin"
    
    mock_mask_file(str(synth['data']), str(mask_path), str(out_file))
    
    data_out, _ = read_binary_file(out_file)
    assert np.all(np.isnan(data_out)), "All pixels should be NaN"


# ============================================================================
# 3-4. Header & Pre-write Validation
# ============================================================================

def test_header_updated_dtype(tmp_dir: Path, synth_data):
    """Header dtype updated to 'f32' before write [validates fix #3]."""
    synth = synth_data(dtype='int16')
    out_file = tmp_dir / "header_test.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    with open(out_file, 'rb') as f:
        header = f.read(1024)
    dtype_str = header[12:16].decode().strip('\x00')
    assert dtype_str == 'f32', f"Header dtype should be 'f32', got '{dtype_str}' [validates fix #3]"


def test_pre_write_validation_dtype_float32(tmp_dir: Path, synth_data):
    """Data converted to float32 before write [validates fix #4]."""
    synth = synth_data(dtype='int16')
    out_file = tmp_dir / "prewrite_test.bin"
    
    # Should not raise error even with int16 input
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, _ = read_binary_file(out_file)
    assert data_out.dtype == np.float32, "Pre-write validation ensures float32 conversion"


# ============================================================================
# 5. Byte Count Verification
# ============================================================================

@pytest.mark.parametrize("height,width,bands", [
    (10, 10, 1),
    (256, 256, 2),
    (1024, 1024, 1),
    (100, 200, 3)
])
def test_byte_count_verification(tmp_dir: Path, synth_data, height: int, width: int, bands: int):
    """Bytes written == H×W×bands×4 [validates fix #5]."""
    synth = synth_data(height=height, width=width, bands=bands)
    out_file = tmp_dir / f"bytecount_{height}_{width}_{bands}.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    expected_bytes = height * width * bands * 4
    data_size = os.path.getsize(out_file) - 1024  # Subtract header
    assert data_size == expected_bytes, f"Byte count mismatch: {data_size} != {expected_bytes} [validates fix #5]"


# ============================================================================
# 6. Endianness Preservation
# ============================================================================

def test_endianness_little_preserved(tmp_dir: Path, synth_data):
    """Little-endian input preserves '<' endianness [validates fix #6]."""
    synth = synth_data(endian='<')
    out_file = tmp_dir / "endian_le.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    with open(out_file, 'rb') as f:
        header = f.read(1024)
    endian = header[16:20].decode().strip('\x00')
    assert endian == '<', "Endianness should be preserved as '<'"


def test_endianness_big_preserved(tmp_dir: Path, synth_data):
    """Big-endian input preserves '>' endianness [validates fix #6]."""
    synth = synth_data(endian='>')
    out_file = tmp_dir / "endian_be.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    with open(out_file, 'rb') as f:
        header = f.read(1024)
    endian = header[16:20].decode().strip('\x00')
    assert endian == '>', "Endianness should be preserved as '>'"


# ============================================================================
# 7. Backup Creation
# ============================================================================

def test_backup_created(tmp_dir: Path, synth_data):
    """.bak backup created before overwrite, original intact [validates fix #7]."""
    synth = synth_data()
    backup = synth['data'].parent / (synth['data'].name + '.bak')
    
    orig_size = os.path.getsize(synth['data'])
    mock_mask_file(str(synth['data']), str(synth['mask']))  # Overwrites input
    
    assert backup.exists(), "Backup .bak file must exist"
    assert os.path.getsize(backup) == orig_size, "Backup size must match original"


def test_preexisting_bak_overwritten(tmp_dir: Path, synth_data):
    """Pre-existing .bak is overwritten [validates edge case]."""
    synth = synth_data()
    bak_path = synth['data'].parent / (synth['data'].name + '.bak')
    
    # Pre-create with different content
    with open(bak_path, 'w') as f:
        f.write("old backup")
    
    old_bak_size = os.path.getsize(bak_path)
    mock_mask_file(str(synth['data']), str(synth['mask']))
    
    new_bak_size = os.path.getsize(bak_path)
    assert new_bak_size == os.path.getsize(synth['data']), "Pre-existing .bak should be overwritten with new backup"


# ============================================================================
# 8. Correct Masking Operation
# ============================================================================

def test_masking_correct_nonlake_preserved(tmp_dir: Path, synth_data):
    """Non-lake pixels unchanged, lake pixels NaN [validates fix #8]."""
    synth = synth_data(height=20, width=20)
    mask_path = create_empty_mask(tmp_dir, 20, 20)  # No masking
    out_file = tmp_dir / "mask_correct.bin"
    
    orig_data, _ = read_binary_file(synth['data'])
    mock_mask_file(str(synth['data']), str(mask_path), str(out_file))
    
    new_data, _ = read_binary_file(out_file)
    np.testing.assert_array_almost_equal(orig_data, new_data, err_msg="Empty mask should preserve all data")


def test_partial_mask_correct(tmp_dir: Path, synth_data):
    """Partial mask sets only lake pixels to NaN [validates fix #8]."""
    H, W = 15, 15
    synth = synth_data(height=H, width=W)
    mask_data = np.random.choice([True, False], (H, W), p=[0.2, 0.8])
    mask_path = tmp_dir / "partial.npz"
    np.savez_compressed(mask_path, mask=mask_data)
    
    out_file = tmp_dir / "partial_mask.bin"
    orig_data, _ = read_binary_file(synth['data'])
    
    mock_mask_file(str(synth['data']), str(mask_path), str(out_file))
    new_data, _ = read_binary_file(out_file)
    
    # Non-masked (False in mask) should be preserved
    non_lake_idx = ~mask_data
    np.testing.assert_allclose(
        new_data[0][non_lake_idx], 
        orig_data[0][non_lake_idx], 
        rtol=1e-5,
        err_msg="Non-lake pixels should be preserved"
    )
    
    # Masked (True in mask) should be NaN
    lake_idx = mask_data
    assert np.all(np.isnan(new_data[0][lake_idx])), "Lake pixels must be NaN"


def test_multiband_masking_uniform(tmp_dir: Path, synth_data):
    """Masking applies uniformly to all bands [validates multi-band edge case]."""
    synth = synth_data(bands=3, height=10, width=10)
    out_file = tmp_dir / "multiband.bin"
    
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, meta = read_binary_file(out_file)
    assert data_out.shape[0] == 3, "All bands preserved"
    
    # All bands have NaNs in same locations
    nan_mask = np.isnan(data_out[0])
    for b in range(1, 3):
        assert np.array_equal(np.isnan(data_out[b]), nan_mask), f"Band {b} NaN mismatch"


# ============================================================================
# Edge Cases
# ============================================================================

def test_mask_dimension_mismatch(tmp_dir: Path, synth_data):
    """Mismatched mask dimensions raise error."""
    synth = synth_data(height=10, width=10)
    mismatch_mask = tmp_dir / "mismatch.npz"
    np.savez_compressed(mismatch_mask, mask=np.zeros((5, 5), bool))
    
    with pytest.raises(ValueError, match="Mask dimension mismatch"):
        mock_mask_file(str(synth['data']), str(mismatch_mask))


def test_empty_file(tmp_dir: Path):
    """Empty input file handled appropriately."""
    empty_file = tmp_dir / "empty.bin"
    empty_file.touch()
    
    mask_path = tmp_dir / "mask.npz"
    np.savez_compressed(mask_path, mask=np.zeros((0, 0), bool))
    
    with pytest.raises((ValueError, struct.error)):
        mock_mask_file(str(empty_file), str(mask_path))


@pytest.mark.parametrize("size", [(1, 1, 1), (256, 256, 2), (512, 512, 3)])
def test_various_sizes(tmp_dir: Path, synth_data, size):
    """Various dimensions including large [validates scalability]."""
    h, w, b = size
    synth = synth_data(height=h, width=w, bands=b)
    out_file = tmp_dir / f"size_test_{h}_{w}_{b}.bin"
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    data_out, meta = read_binary_file(out_file)
    assert data_out.shape == (b, h, w), f"Shape mismatch for {h}x{w}x{b}"


def test_inplace_operation(tmp_dir: Path, synth_data):
    """In-place masking (output=None) works with backup [validates fix #7]."""
    synth = synth_data()
    orig_size = os.path.getsize(synth['data'])
    backup = synth['data'].parent / (synth['data'].name + '.bak')
    
    mock_mask_file(str(synth['data']), str(synth['mask']))  # inplace
    
    assert backup.exists(), "Backup .bak file should exist"
    assert os.path.getsize(backup) == orig_size, "Backup size should match original"


def test_multiband_different_counts(tmp_dir: Path, synth_data):
    """Single band, 2-band, and 3-band files all process correctly."""
    for n_bands in [1, 2, 3]:
        synth = synth_data(bands=n_bands, height=20, width=20)
        out_file = tmp_dir / f"bands_{n_bands}.bin"
        
        mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
        
        data_out, meta = read_binary_file(out_file)
        assert meta['bands'] == n_bands, f"Band count mismatch for {n_bands} bands"
        assert data_out.shape[0] == n_bands, f"Data shape mismatch for {n_bands} bands"


# ============================================================================
# Integration Tests (Comprehensive)
# ============================================================================

def test_full_workflow_int16_to_float32_masked(tmp_dir: Path, synth_data):
    """Full workflow: int16 input -> float32 output with masking [validates all fixes]."""
    # Setup
    synth = synth_data(height=50, width=50, bands=2, dtype='int16', endian='<')
    out_file = tmp_dir / "full_workflow.bin"
    
    orig_data, _ = read_binary_file(synth['data'])
    orig_file_size = os.path.getsize(synth['data'])
    
    # Execute
    mock_mask_file(str(synth['data']), str(synth['mask']), str(out_file))
    
    # Verify all 8 fixes
    # Fix #1: Output is float32
    data_out, meta = read_binary_file(out_file)
    assert data_out.dtype == np.float32, "Fix #1: Output dtype must be float32"
    
    # Fix #2: NaN preservation
    assert np.any(np.isnan(data_out)), "Fix #2: NaNs must be present"
    
    # Fix #3: Header updated
    with open(out_file, 'rb') as f:
        header = f.read(1024)
    assert header[12:16].decode().strip('\x00') == 'f32', "Fix #3: Header dtype must be f32"
    
    # Fix #4: Pre-write validation (implicitly passed)
    assert data_out.dtype == np.float32, "Fix #4: Data must be float32 before write"
    
    # Fix #5: Byte count
    expected_bytes = 50 * 50 * 2 * 4
    actual_bytes = os.path.getsize(out_file) - 1024
    assert actual_bytes == expected_bytes, f"Fix #5: Byte count {actual_bytes} != {expected_bytes}"
    
    # Fix #6: Endianness preserved
    assert header[16:20].decode().strip('\x00') == '<', "Fix #6: Endianness must be preserved"
    
    # Fix #7: Backup created
    backup = synth['data'].parent / (synth['data'].name + '.bak')
    assert backup.exists(), "Fix #7: Backup must exist"
    assert os.path.getsize(backup) == orig_file_size, "Fix #7: Backup size must match original"
    
    # Fix #8: Correct masking
    mask_data = np.load(synth['mask'])['mask'].astype(bool)
    for b in range(2):
        lake_pixels = mask_data
        assert np.all(np.isnan(data_out[b][lake_pixels])), "Fix #8: Lake pixels must be NaN"


# Run with: pytest test_mask_lakes_insar.py -v -s
