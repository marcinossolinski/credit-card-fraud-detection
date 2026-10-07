# Credit Card Fraud Detection

Comparison of three gradient-boosting and ensemble models for detecting fraudulent credit card transactions in a **highly imbalanced dataset** (only 0.17% of transactions are fraud).

![Model comparison](results/model_comparison.png)

## Approach

1. **Data** – 284,807 real card transactions (492 frauds), anonymised PCA features `V1–V28`, plus `Time` and `Amount`.
2. **Split** – stratified 80/20 train/test split, so both sets keep the original fraud ratio.
3. **Scaling** – `StandardScaler` fitted on the training set only.
4. **Class imbalance** – **SMOTE** oversampling applied to the training set only (the test set stays untouched, so there is no data leakage).
5. **Models** – Random Forest, XGBoost and LightGBM.
6. **Evaluation** – precision, recall and F1 for the fraud class, confusion matrix, training and prediction time. Accuracy is deliberately not used, because a model that never flags fraud would still score 99.8%.

## Results

Test set: 56,962 transactions, 98 of them fraudulent.

| Model | Precision | Recall | F1 | Train time |
|---|---|---|---|---|
| **Random Forest** | **0.87** | 0.83 | **0.85** | 151 s |
| XGBoost | 0.72 | **0.86** | 0.79 | 3.5 s |
| LightGBM | 0.49 | 0.85 | 0.62 | 7.0 s |

- **Random Forest** gives the best balance: it catches 83% of frauds with the fewest false alarms.
- **XGBoost** catches the most frauds and trains over 40× faster, at the cost of more false alarms – a good choice when missing a fraud is more expensive than a manual review.
- **LightGBM** with default settings produces too many false positives; it would need threshold or hyperparameter tuning.

Results were measured on a 2-core machine; times will differ on other hardware.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · LightGBM · Matplotlib

## How to run

1. Download `creditcard.csv` from [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put it in the `data/` folder (the file is ~150 MB, so it is not stored in this repository).
2. Install dependencies and run:

```bash
pip install -r requirements.txt
python fraud_detection.py
```

The script prints a comparison table and saves `results/metrics.csv` and `results/model_comparison.png`. You can also pass a custom path: `python fraud_detection.py path/to/creditcard.csv`.

Code comments and chart labels are in Polish (university project).

## Authors

Marcin Ossoliński, Maksymilian Rak – Rzeszów University of Technology
