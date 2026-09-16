# Bitcoin price trajectory framework — replication code

Computational companion to “Conditions Supporting Bitcoin Price Trajectories: An inverse framework for scenario assessment,” Murray A. Rudd. DOI: https://doi.org/10.2139/ssrn.7436601

This repository contains simulation and analysis code, a numerical protocol, and the small reference records needed for replication. All economic inputs are synthetic. No paper drafts, appendices, reference documents, manuscript builders, or writing package are included. Results describe conditional compatibility, not empirical forecasts or probabilities.

## Requirements

Python 3.12, a C++17 compiler available as `g++`, and the packages pinned in `requirements.txt`. The engine uses a Linux shared library; Windows users should run the code in Ubuntu through WSL. GitHub Desktop is needed only for uploading, not for calculations.

From the repository folder in Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python replicate.py
```

If Python's venv module or the compiler is missing on Ubuntu, install `python3-venv` and `g++` using your system package manager.

The default command checks the auctions, rebuilds the main trajectory calculations, performs the supplementary analysis and independent checks, and generates the figures. It uses the supplied original `records.json`, `analysis.json`, and eight candidate masks as reference inputs. Dense histories and intervention caches are regenerated as needed.

To regenerate all original numerical records as well:

```bash
python replicate.py --all
```

This is the complete calculation route from synthetic parameters. Allow several minutes or longer depending on hardware and approximately 1.1 GB for generated caches. Neither command creates a manuscript. After changing the engine or parameter domains, delete generated `results_v05/outputs/*.npz` caches and run with `--all`.

## Figure outputs

Each figure is generated as PNG and PDF.

| Paper figure | Output stem |
|---|---|
| 1 | `trajectory/figure_1_trajectories` |
| 2 | `trajectory/figure_2_conditions_measurement` |
| 3 | `trajectory/figure_4_continuation` |
| 4 | `results_v05/outputs/figure_2_information` |
| C1 | `additional/figure_C1_planning` |
| D1 | `results_v05/outputs/figure_1_equivalence` |
| D2 | `results_v05/outputs/figure_3_joint_responses` |

The original plotting script also generates `results_v05/outputs/figure_4_regime` as an additional diagnostic. Output filenames retain their original numbering to preserve script compatibility.

## Code and numerical records

- `results_v05/engine.cpp`: economic engine; initial states and ordered parameter menus are specified in `compute.py`.
- `results_v05/validate.py`: independent auction, restart, conservation, and menu checks.
- `results_v05/compute.py`, `analyze.py`, `misspecification.py`: original experiments and diagnostics.
- `trajectory/`: complete-path compatibility analysis and plots.
- `additional/protocol.json`: fixed hypothetical planning queries, measurement allowances, and local refinement rule.
- `additional/`: supplementary analysis, independent witness checks, and planning plot.
- `results_v05/outputs/`: only the two numerical JSON records and eight reference masks are distributed. All other outputs are generated locally.

State order is `[log_price, h_A, h_B, h_C, c_A, c_B, c_C, lagged_log_return]`. O/T parameter order is `[f, d, g, eta, tau]`. Price indices use the main T generating economy's month-36 price as 100. Candidate counts depend on menu spacing and have no probability interpretation.

The scientific engine and analysis scripts are unchanged from the v0.8 source archive. Replication entry points and dependencies have been limited to computations and figures. Generated figures and caches are ignored by Git.
