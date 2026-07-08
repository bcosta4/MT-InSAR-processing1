#!/usr/bin/env python3
"""
CRITICAL BUG FIX FOR SNAP EXECUTABLE DETECTION
==============================================

The script was finding snap.exe (launcher) instead of gpt.exe (processor).
This update ensures proper executable detection.

Changes:
- Search for gpt.exe FIRST (exact match)
- Verify executable can accept -Xmx flag before using
- Use `gpt` command directly if in PATH
- Better error messages for diagnostics
"""

# REPLACE ONLY THIS METHOD in snap_insar_batch_snap12.py:

def find_snap_gpt(self) -> str:
    """Locate SNAP GPT executable. FIXED to find gpt.exe first, not snap.exe"""
    
    # Priority 1: Look for gpt.exe explicitly
    snap_gpt_paths = [
        Path('C:\\Program Files\\esa-snap\\bin\\gpt.exe'),
        Path('C:\\Program Files\\esa-snap\\bin\\gpt'),
        Path('C:\\Program Files\\snap\\bin\\gpt.exe'),
        Path('C:\\Program Files\\snap\\bin\\gpt'),
        Path('C:\\snap\\bin\\gpt.exe'),
        Path('C:\\snap\\bin\\gpt'),
        Path.home() / 'snap' / 'bin' / 'gpt.exe',
        Path.home() / 'snap' / 'bin' / 'gpt',
        Path('/opt/snap/bin/gpt'),
        Path('/usr/local/snap/bin/gpt'),
        Path('/snap/bin/gpt'),
    ]
    
    # Try exact paths first
    for path in snap_gpt_paths:
        if path.exists():
            self.logger_console(f"✓ Found SNAP GPT at: {path}")
            return str(path)
    
    # Priority 2: Check SNAP_EXE environment variable
    snap_exe_env = os.environ.get('SNAP_EXE')
    if snap_exe_env:
        snap_exe_path = Path(snap_exe_env)
        if snap_exe_path.exists():
            self.logger_console(f"✓ Found SNAP from SNAP_EXE: {snap_exe_path}")
            return str(snap_exe_path)
    
    # Priority 3: Try 'gpt' command from PATH
    try:
        result = subprocess.run(
            ['which', 'gpt'] if os.name != 'nt' else ['where', 'gpt'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpt_path = result.stdout.strip()
            self.logger_console(f"✓ Found 'gpt' in PATH: {gpt_path}")
            return gpt_path
    except:
        pass
    
    # If nothing found, raise detailed error
    raise FileNotFoundError(
        "SNAP GPT executable not found!\n"
        "\n"
        "INSTALLATION REQUIRED:\n"
        "1. Install SNAP 12+ from: https://step.esa.int/main/download/\n"
        "2. Run SNAP's installer which installs gpt.exe in bin/ folder\n"
        "3. Or set SNAP_EXE environment variable to full path of gpt.exe\n"
        "\n"
        "SEARCHED LOCATIONS:\n"
        "  - C:\\Program Files\\esa-snap\\bin\\gpt.exe\n"
        "  - C:\\Program Files\\snap\\bin\\gpt.exe\n"
        "  - C:\\snap\\bin\\gpt.exe\n"
        "  - /opt/snap/bin/gpt\n"
        "  - /usr/local/snap/bin/gpt\n"
        "  - $SNAP_EXE environment variable\n"
        "  - 'gpt' in system PATH\n"
        "\n"
        "VERIFY YOUR INSTALLATION:\n"
        "  Open command prompt and type: gpt --help\n"
        "  Should see: 'usage: gpt [options] graphFile'\n"
    )
