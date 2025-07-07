**FAME · Fairness‑Aware Multimodal Embedding**

PyTorch implementation of our MLHC 2025 paper "[Equitable Electronic Health Record Prediction with FAME: Fairness‑Aware Multimodal Embedding](https://doi.org/10.48550/arXiv.2506.13104)"

---
## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Folder Structure](#folder-structure)
5. [Quick Start](#quick-start)
6. [Data Preparation](#data-preparation)
7. [Training & Evaluation](#training--evaluation)
8. [Expected Results](#expected-results)
9. [Custom Use](#custom-use)
10. [Contributing](#contributing)
11. [Changelog](#changelog)
12. [Citation](#citation)
13. [License](#license)
14. [Contact](#contact)

---

## Overview

FAME is a **fairness‑aware multimodal AI framework** that fuses **structured EHR**, **clinical notes**, and **demographics** to make clinical predictions *without* amplifying bias across patient sub‑groups (age, ethnicity, insurance).

* Core idea:* **Weight each modality by how fair it is.**
During training, FAME measures fairness with **EDDI** (Error Distribution Disparity Index) and dynamically increases the influence of modalities that treat all groups more equitably.

The repository reproduces every experiment in the paper—baselines, ablations, and full FAME—using public **MIMIC‑III/IV** datasets.

---

## Key Features

* **One‑command pipeline**: from raw ICU tables to final metrics.
* **Automatic fairness tracking**: EDDI & Equalized Odds logged each epoch.
* **Out‑of‑the‑box baselines**: DfC, AdvDebias, FPM, FairEHR‑CLP.
* **Plug‑and‑play modalities**: swap encoders or add new ones (e.g., imaging).
* **Reproducible**: seeds set, deterministic Torch ops where possible.

---

## Architecture

<p align="center">
  <img width="1075" alt="Screenshot 2025-07-07 at 3 07 03 PM" src="https://github.com/user-attachments/assets/497f1821-ff3c-4eda-b1c7-5af0d11edb07" />
</p>


*Figure 1 – FAME combines BEHRT (structured data) and BioClinicalBERT (clinical notes) plus demographics. The fusion layer applies EDDI‑based modality weights and a sigmoid gate, then optimizes a joint BCE + EDDI loss.*

---

## Folder Structure

| File / Folder                     | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| `FinalCode/New/Final/00_data.py`                      | Extract & preprocess MIMIC data (structured + notes) |
| `01_BEHRT.py`                     | Baseline using BEHRT (structured)                    |
| `02_BioClinicalBERT.py`           | Baseline using BioClinicalBERT (notes)               |
| `03_DfC.py`                       | Demographic‑free Classification baseline             |
| `04_AdvDebias.py`                 | Adversarial debiasing baseline                       |
| `05_FPM.py`                       | Fair Patient Model baseline                          |
| `06_FairEHR-CLP.py`               | Contrastive debiasing baseline                       |
| `07_multimodal_average_fusion.py` | Average fusion (3 modalities)                        |
| `08_multimodal_eddi_fusion.py`    | EDDI‑only fusion (no sigmoid)                        |
| `09_multimodal_sigmoid_fusion.py` | Sigmoid‑only fusion (no EDDI)                        |
| `10_FAME.py`                      | **Full FAME** – EDDI + Sigmoid + joint loss          |
| `requirements.txt`                | Python dependencies                                  |
| `docs/figures/`                   | Architecture & result visuals                        |
| `tests/`                          | Unit tests & CI scripts                              |

---

## Quick Start

> Tested on **Python ≥3.9** and **PyTorch ≥2.1** with a single GPU (≥12 GB VRAM).

```bash
# 1. Clone
$ git clone https://github.com/NikkieHooman/FAME.git
$ cd FAME

# 2. (Optional) Create & activate virtual environment
$ python -m venv venv
$ source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt
#   or, with conda:
$ conda env create -f environment.yml && conda activate fame

# 4. Pre‑process MIMIC (≈15 min)
$ python 00_data.py --mimic_root /path/to/mimic --out_dir data/

# 5. Train FAME on all three tasks
$ python 10_FAME.py --tasks mortality readmission ventilation --lambda 0.8 \
                    --epochs 30 --batch_size 32
```

CI will automatically run `pytest` on push / PR to validate basic functionality.

---

## Data Preparation

1. **MIMIC Credentials** – Sign the PhysioNet credentialing agreement and download **MIMIC‑III v1.4** or **MIMIC‑IV v2.2**. Place CSVs in e.g. `/mnt/mimic`.
2. Run `00_data.py` to generate

   * `structured.pkl` – 2‑hour‑binned vitals, labs, demographics
   * `notes.pkl` – token‑id chunks (≤512 tokens) and CLS vectors
   * `labels.csv` – binary labels (mortality, readmission, ventilation)
3. Sensitive attributes auto‑extracted: **age buckets, ethnicity, insurance**.

*No patient data is committed to this repo.*

---

## Training & Evaluation

Each script shares a common CLI—swap the filename to try another model.

```bash
# Baseline: BEHRT on mortality
python 01_BEHRT.py --task mortality

# Contrastive baseline (FairEHR‑CLP)
python 06_FairEHR-CLP.py --task readmission --temp 0.07

# Ablation: EDDI‑only fusion
python 08_multimodal_eddi_fusion.py --task ventilation --lambda 0.8
```

Metrics print every epoch and are logged to `outputs/logs/*.csv`:

```
Epoch 5 | AUROC 0.943 | AUPRC 0.817 | EDDI 0.44 | EO 4.25
```

TensorBoard is supported via `--tensorboard` flag.

---

## Expected Results

<details>
<summary>5‑run averages from the paper (Table 3)</summary>

| Task             | AUROC ↑  | AUPRC ↑  | EDDI % ↓ | EO % ↓   |
| ---------------- | -------- | -------- | -------- | -------- |
| **Mortality**    | **0.94** | **0.82** | **0.44** | **4.25** |
| **LOS ≥ 7 days** | **1.00** | **1.00** | **0.02** | **0.06** |
| **Ventilation**  | **0.84** | **0.97** | **2.77** | **0.55** |

</details>

---

## Custom Use

Want to apply FAME to your own dataset or add an imaging modality?

1. **Create** a `CustomDataset` in `data_loader.py` that returns a dict with keys `structured`, `text`, `demo`, `label` (+ your new key, e.g., `image`).
2. **Add** your encoder in `models/encoders.py` and register it in `__init__.py`.
3. **Pass** `--image_encoder your_cnn_name` from CLI. The fairness engine handles extra modalities automatically.

---

## Contributing

We welcome pull requests! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** for our coding guidelines, branch strategy, and how to run local tests. By contributing you agree to follow the **[Code of Conduct](CODE_OF_CONDUCT.md)**.

---

## Changelog

See **[CHANGELOG.md](CHANGELOG.md)** for a curated list of notable updates.

---

## Citation

If FAME is useful in your work, please cite us ✨

```bibtex
@misc{lastname2024fame,
  title  = {Equitable Electronic Health Record Prediction with FAME: Fairness‑Aware Multimodal Embedding},
  author = {Lastname, Firstname and Lastname, Firstname},
  year   = {2024},
  note   = {Machine Learning for Healthcare (under review)},
  url    = {https://github.com/NikkieHooman/FAME}
}
```

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for the full text.

---

## Contact

Questions or ideas? Feel free to open an issue or email **[name@email.edu](mailto:name@email.edu)**.

*Happy (fair) modeling ♥️*
