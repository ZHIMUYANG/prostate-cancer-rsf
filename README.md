---
title: Prostate Cancer CSS Survival Predictor
emoji: 🎗️
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.57.0"
app_file: app.py
pinned: false
license: mit
---

# Prostate Cancer CSS Survival Predictor

Random Survival Forest (RSF) model for predicting **cancer-specific survival** in prostate cancer patients.

## Input Variables
- Age (35-90)
- T Stage (T1/T2/T3/T4)
- Gleason Grade (Low/Mid/High)
- PSA (ng/mL)
- Surgery (RP): Yes/No
- Radiation: Yes/No
- Chemotherapy: Yes/No
- Race: White/Black/Other
- Marital Status: Married/Unmarried

## Model
- Random Survival Forest (100 trees, max_depth=5)
- Trained on n=4,780 patients (SEER database)
- Endpoint: Cancer-specific survival (CSS)
- C-index: 0.697 (test set)

## Citation
*This tool is for research purposes only. Not for clinical use.*
