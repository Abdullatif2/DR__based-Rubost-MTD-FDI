# Proactive Smart-Grid Defense: DRL-Powered Moving Target Defense Against False Data Injection Attacks

This repository contains the code, selected model artifacts, representative outputs, and lightweight sample data associated with the manuscript:

**Proactive Smart-Grid Defense: DRL-Powered Moving Target Defense Against False Data Injection Attacks**

The project studies **moving target defense (MTD)** against **false data injection (FDI)** attacks in cyber-physical power systems using **deep reinforcement learning (DRL)**. In particular, it implements policy-learning-based MTD control using **DDPG** and **TD3** for continuous perturbation selection under AC power-flow-based evaluation.

---

## Overview

Conventional MTD design for power-system state estimation often relies on iterative optimization, which can be computationally expensive for real-time deployment. This repository provides an implementation of a DRL-based alternative in which the perturbation policy is learned offline and then applied online through direct inference.

The codebase supports:

- DRL-based MTD control using **DDPG** and **TD3**
- evaluation under **AC/DC state-estimation-related settings**
- experiments on **IEEE 14-bus** and **IEEE 57-bus** benchmark systems
- generation of representative training curves, action histories, and evaluation outputs
- selected saved models and outputs to facilitate reproducibility

---

## Repository Structure

```text
DRL_MTD_FDI_GitHub_Ready/
├── DRL/                     # DRL agents, replay buffer, noise process, networks
├── utils/                   # Grid modeling, initialization, MTD, and evaluation helpers
├── utils_2/                 # Dataset and result-saving utilities
├── configs/                 # Experiment and neural-network configuration files
├── data/                    # Lightweight/sample datasets for quick setup
├── Output/                  # Selected DDPG/TD3 outputs and checkpoints
├── Output_case57_ddpg_tryout/   # Selected IEEE 57-bus output folder
├── final_simulation_data_newest_one/  # Representative saved simulation artifacts
├── main_DRL.py              # Main training/testing entry point
├── save_dataset.py          # Dataset generation utility
├── evaluation_ac.py         # AC evaluation script
├── evaluation_bdd.py        # BDD-related evaluation script
├── evaluation_dc.py         # DC evaluation script
├── plot_*.py                # Plotting scripts used for analysis and figures
├── score_history_aligned.png
├── score_history_last_200.png
├── environment.yml          # Lightweight conda environment
└── README.md
```

---

## Main Components

### `DRL/`
Contains the reinforcement learning agents and supporting components:
- `ddpg_torch.py`: DDPG implementation
- `td3_torch.py`: TD3 implementation
- `networks.py`, `networks_TD3.py`: actor/critic neural network definitions
- `buffer.py`: replay buffer
- `noise.py`: exploration noise model
- `PS_env.py`, `PS_env_DDET.py`: environment definitions

### `utils/`
Contains power-system and MTD-related utilities, including initialization, grid functions, and MTD routines.

### `configs/`
Configuration files controlling the experiment setup, case selection, network settings, and related parameters.

---

## Installation

A lightweight Conda environment is included.

```bash
conda env create -f environment.yml
conda activate drl-mtd-fdi
```

If you prefer `pip`, install the main requirements manually:

```bash
pip install numpy pandas scipy matplotlib jupyter torch pypower
```

---

## Quick Start

### 1. Configure the experiment

The workflow is primarily configuration-driven. Before running training or evaluation, review the following files:

- `configs/config.py`
- `configs/nn_setting.py`
- `utils/settings.py`

Typical settings include:
- benchmark case selection (e.g., IEEE 14-bus or IEEE 57-bus)
- DRL agent type (`DDPG` or `TD3`)
- training or testing mode
- perturbation ratio / experimental setting
- dataset and output paths

### 2. Train or test a DRL agent

```bash
python main_DRL.py
```

### 3. Generate datasets if needed

```bash
python save_dataset.py
```

### 4. Run evaluation utilities

Representative utilities include:

```bash
python evaluation_ac.py
python evaluation_bdd.py
python evaluation_dc.py
```

### 5. Plot representative results

Examples include:

```bash
python plot_incomplete.py
python plot_incomplete_tryouts_plt_57.py
python plot_actions.py
```

---

## Reproducing Representative Results

This repository includes selected saved artifacts to help reproduce representative trends without requiring a full rerun from scratch.

Included artifacts may help with:
- training-curve visualization
- action-history inspection
- selected checkpoint loading
- representative output inspection for the IEEE 14-bus and IEEE 57-bus settings

Two example training-history figures are already included:
- `score_history_aligned.png`
- `score_history_last_200.png`

These are useful for quickly checking convergence behavior and for reproducing training-stability visualizations in the paper or supplementary material.

---

## Data Availability

To keep the repository lightweight and GitHub-friendly, this package includes **sample/lightweight datasets** and **selected outputs**, rather than every large intermediate artifact generated during the project.

Included sample datasets are located in:
- `data/dataset_case14.csv`
- `data/dataset_case57_training_mini.csv`
- `data/dataset_case_training_tryout.csv`

Please note:
- some full datasets and large intermediate files used during development were intentionally excluded from this public package to keep the repository manageable for standard GitHub upload,
- paths in the main scripts may need to be adjusted depending on your local folder structure,
- if full-scale dataset hosting is needed later, Git LFS or an external storage link can be added.

Additional notes are available in `data/README.md`.

---

## Reproducibility Notes

- The repository retains the code structure used during the main development and manuscript-preparation stages.
- Selected model checkpoints and outputs are included to support quick verification of representative trends.
- For a clean rerun on your own machine, you may wish to update dataset and output paths in the configuration files.
- Because the workflow is configuration-based, reproducibility is best achieved by fixing the case, perturbation settings, agent type, and file paths before running.

---

## Suggested Workflow for New Users

1. Create the environment using `environment.yml`.
2. Inspect `configs/config.py` and `utils/settings.py`.
3. Start with the lightweight sample data in `data/`.
4. Run `main_DRL.py` in a controlled test setting.
5. Use the plotting scripts and saved outputs to verify the expected trends.
6. Extend to your own case settings or larger experiments as needed.

---

## Intended Use

This repository is released to support:
- research reproducibility,
- follow-up experiments on DRL-based MTD for smart-grid security,
- educational use for researchers studying FDI attacks, MTD, and DRL in power systems.

If you use this repository in academic work, please cite the associated paper once the final bibliographic details are available.

---

## Citation

A `CITATION.cff` file and final citation metadata can be added after publication. Until then, please cite the associated manuscript using the title below:

> **Proactive Smart-Grid Defense: DRL-Powered Moving Target Defense Against False Data Injection Attacks**

A BibTeX entry can be added here once the journal, volume, pages, year, and DOI are finalized.

---

## License

Please add your preferred open-source license before making the repository public. A standard permissive license such as **MIT** or **BSD-3-Clause** is often suitable for research code, depending on your preference and institutional requirements.

---

## Contact

For questions related to the manuscript or repository, please contact the corresponding authors through the paper information or update this section with the preferred contact email before public release.

---

## Acknowledgment

This repository accompanies the research manuscript listed above and is prepared for public release to support transparency and reproducibility.
