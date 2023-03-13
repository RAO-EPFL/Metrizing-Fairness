# Offline Classification and Regression Experiments

Implementation of MFL and baselines for statistical parity fair regression and classification.

## Classification
Use `run.py` to run the experiments, analyse them with `Analyze_Classification_Results.ipynb`. Contains the following implementations:
- `run_benchmark.py`: Implementation of MFL
- `fair_KDE.py`: Implementation of fair KDE baseline [Cho et al.].
- `MMD_fair_run.py`: Implementation of fair Sinkhorn baseline [Oneto et al.].
- `zafar_classification.py`: Implementation of fair logistic regression [Zafar et al.]

## Regression
Use `run_regression.py` to run the experiments, analyse them with `Analyze_Regression_Results.ipynb`. Contatins the following implementations:
- `run_benchmark_regression.py`: Implementation of MFL
- `baseline_convex_fair_regression.py`: Implementation of fair convex regression [Berk et al.]