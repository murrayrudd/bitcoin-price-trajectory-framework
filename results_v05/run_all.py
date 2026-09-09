"""Regenerate the original numerical experiments and figures."""
from pathlib import Path
import subprocess
import sys
root = Path(__file__).resolve().parent
for name in ['validate.py', 'compute.py', 'analyze.py', 'misspecification.py', 'figures.py']:
    subprocess.run([sys.executable, str(root / name)], cwd=root, check=True)
