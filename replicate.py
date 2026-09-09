"""Reproduce numerical results and figures; never builds a manuscript."""
from pathlib import Path
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--all', action='store_true', help='Recompute all original numerical records before the trajectory analyses')
args = parser.parse_args()
root = Path(__file__).resolve().parent
scripts = ['results_v05/validate.py']
if args.all:
    scripts += ['results_v05/compute.py', 'results_v05/analyze.py', 'results_v05/misspecification.py']
scripts += ['trajectory/analyze_trajectories.py', 'trajectory/figures_trajectories.py',
            'additional/analyze_revision.py', 'additional/verify_revision.py',
            'additional/make_figure.py', 'results_v05/figures.py']
for script in scripts:
    print(f'Running {script}', flush=True)
    subprocess.run([sys.executable, str(root / script)], cwd=root, check=True)
