# Methanation Code and Data

This repository contains the dataset and code used in the study **"Unveiling Key Variables in CO2 Methanation Using Machine Learning for Ni-Based Catalysts."**

## Repository Structure
- **`data/`**: Contains the dataset (`data_art1.xlsx`).
- **`scripts/`**: Python scripts for:
  - Data preprocessing (`preprocess.py`).
  - Model training and evaluation (`models.py`).
  - Interpretability analysis (`interpretability.py`).

## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/lamaramaciel/methanation-code-data.git
   pip install -r requirements.txt
   python scripts/preprocess.py
   python scripts/models.py
   python scripts/interpretability.py
