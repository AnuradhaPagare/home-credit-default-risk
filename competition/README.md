# Home Credit Default Risk: LightGBM + CatBoost blend

Predicting the probability that a loan applicant will have payment difficulties, for the in-class competition
*Home Credit Default Risk* (GCI Global 2026), scored by ROC-AUC.

**Result: 5-fold CV ROC-AUC 0.7625 +/- 0.0032**, up from 0.670 for the starting tutorial notebook.
Expect roughly 0.76 on the leaderboard (see caveats).

## Approach
- **Data:** only the provided `train.csv` (171,202 rows, 8.07% defaults) and `test.csv` (61,500 rows). No external data, no hand-labelling.
- **Features (69):** the 32 raw columns plus engineered ones: credit/income, annuity/income and credit/annuity ratios, age and employment in years,
  combined `EXT_SOURCE` statistics and interactions, missingness flags. The `DAYS_EMPLOYED = 365243` placeholder (18% of rows) and "XNA" categories are treated as missing.
- **Models:** Optuna-tuned LightGBM (3 seeds, fixed number of rounds) blended 0.727 / 0.273 with CatBoost.
- **Validation:** 5-fold stratified CV with identical folds for every model; per-fold AUC, mean and standard deviation, train vs validation gap.

## Results (same 5 folds)
| Model | CV AUC |
|---|---|
| Original tutorial (RandomForest, 5 features) | 0.670 |
| Logistic Regression | 0.740 |
| Extra Trees | 0.744 |
| LightGBM, raw 32 features | 0.7505 |
| XGBoost | 0.7581 |
| CatBoost | 0.7583 |
| LightGBM + engineered features | 0.7591 |
| LightGBM, tuned, 1 seed | 0.7615 |
| LightGBM, tuned, 3-seed average | 0.7619 |
| **LightGBM (3 seeds) + CatBoost blend** | **0.7625** |

More feature engineering (interactions, ratios, categorical combinations) and feature pruning did **not** help; details in `performance_report.md`.

## Repository layout
```
README.ipynb               competition instructions (provided)
tutorial.ipynb             final pipeline, end to end (run this)
solution_pipeline.ipynb    leakage audit, investigation of the reported 1.000 score, experiment record
performance_report.md      full report
experiment_results.csv     every experiment: model, features, hyperparameters, fold scores, timing
feature_importance.csv     gain per feature
HomeCredit_columns_description.xlsx   data dictionary
input/                     put train.csv, test.csv, sample_submission.csv here (not in the repo)
output/Submission.csv      final predictions; output/archive/ holds earlier versions
scripts/                   experiment code (features, tuning, ablations, blending)
```

## How to run
1. `pip install -r requirements.txt`
2. Place `train.csv`, `test.csv` and `sample_submission.csv` in `input/`.
3. Open `tutorial.ipynb` from the project folder and run all cells (about 20 minutes on one CPU; faster with more cores).
   It writes `output/Submission.csv`. Seeds and folds are fixed, so reruns reproduce the same predictions.
4. `solution_pipeline.ipynb` reruns the audit; its last section needs the saved predictions in `preds/` (kept out of git by default,
   regenerate them with the scripts in `scripts/` or skip that section).

## Data and leakage checks
No leakage was found: no duplicate rows, no signal in the ID or row order, no suspiciously predictive column, and train and test are
nearly indistinguishable (adversarial AUC 0.51). The one structural relationship is 204 repeat-customer pairs in train, worth at most about
+0.0004 AUC and not used. A reported leaderboard score of 1.000 could not be verified and cannot be explained by anything in the provided data;
see the investigation section of `performance_report.md`.

## Caveats
- Hyperparameters and blend weights were chosen on the same folds that report the score, so a few thousandths of optimism are likely.
- Sampling noise alone moves this model's AUC by about +/-0.003 on a test-sized sample, and more on a small public-leaderboard subset.
- The competition data belongs to the competition organisers and is not included here.

## Credits
Competition: Home Credit Default Risk (originally on Kaggle, 2018), run as an in-class competition by GCI Global 2026.
Libraries: LightGBM, CatBoost, XGBoost, scikit-learn, Optuna, pandas, NumPy.
