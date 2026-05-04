#!/bin/bash
set -e
export PYTHONPATH="$(pwd)"
python steps/01_preprocess.py
python steps/02_baselines.py
python steps/03_train_cnns.py
python steps/04_hyperparameter_search.py
python steps/05a_window_study.py
python steps/05b_lead_time_study.py
python steps/06_ablation.py
python steps/07_compare_results.py
