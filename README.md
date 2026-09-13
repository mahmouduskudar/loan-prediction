# Loan Default Prediction

Machine learning project that predicts whether a borrower will default on a loan. Built as a data science coding challenge: train on labeled borrower records, then score an unlabeled test set with the probability of default.

The goal is to help a financial institution flag high-risk loans so support can be directed to the right borrowers.

## Dataset

The data is a 2021 sample of borrowers from a financial institution.

| File | Rows | Description |
|------|------|-------------|
| `Datasets/train.csv` | 255,347 | Training sample with ground-truth `Default` labels |
| `Datasets/test.csv` | 109,435 | Test sample without labels (used for submission) |
| `Datasets/data_descriptions.csv` | — | Column names, types, and descriptions |
| `prediction_submission.csv` | 109,435 | Generated default probabilities for the test set |

**Target:** `Default` — `1` if the borrower defaulted, `0` otherwise.

### Features

| Column | Type | Description |
|--------|------|-------------|
| `LoanID` | identifier | Unique loan ID |
| `Age` | integer | Borrower age |
| `Income` | integer | Annual income |
| `LoanAmount` | integer | Amount borrowed |
| `CreditScore` | integer | Creditworthiness score |
| `MonthsEmployed` | integer | Months of employment |
| `NumCreditLines` | integer | Number of open credit lines |
| `InterestRate` | float | Loan interest rate |
| `LoanTerm` | integer | Term length in months |
| `DTIRatio` | float | Debt-to-income ratio |
| `Education` | string | Highest education (PhD, Master's, Bachelor's, High School) |
| `EmploymentType` | string | Full-time, Part-time, Self-employed, Unemployed |
| `MaritalStatus` | string | Single, Married, Divorced |
| `HasMortgage` | string | Yes / No |
| `HasDependents` | string | Yes / No |
| `LoanPurpose` | string | Home, Auto, Education, Business, Other |
| `HasCoSigner` | string | Yes / No |

## Approach

1. **Preprocessing** — median imputation and scaling for numeric fields; most-frequent imputation and one-hot encoding for categoricals. One notebook also adds polynomial interaction features.
2. **Feature selection** — recursive feature elimination with cross-validation (RFECV) using logistic regression, scored on ROC AUC.
3. **Class imbalance** — SMOTE oversampling on the training split.
4. **Model** — a soft-voting ensemble of logistic regression and random forest.
5. **Evaluation** — accuracy, precision, recall, F1, and ROC AUC on a held-out validation split.

### Validation results

From `LoanDefaultPrediction.ipynb` (validation set after SMOTE):

| Metric | Score |
|--------|-------|
| Accuracy | 0.8896 |
| Precision | 0.8725 |
| Recall | 0.9125 |
| F1 | 0.8921 |
| ROC AUC | 0.9579 |

These scores are on a balanced validation split, so they overstate performance on the original (imbalanced) default rate.

## Notebooks

| Notebook | What it does |
|----------|----------------|
| `LoanDefaultPrediction.ipynb` | Coursera challenge notebook: data load, preprocessing, ensemble training, metrics, and `prediction_df` submission |
| `Loan_Predicion.ipynb` | Colab version of the same pipeline, with polynomial features and a more step-by-step layout |

Both notebooks currently read `train.csv` and `test.csv` from the working directory. Copy them out of `Datasets/` first, or change the `read_csv` paths to `Datasets/train.csv` and `Datasets/test.csv`.

## How to run

```bash
git clone https://github.com/mahmouduskudar/loan-prediction.git
cd loan-prediction
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib scikit-learn imbalanced-learn jupyter
jupyter notebook
```

Open either notebook and run all cells. The challenge notebook writes a `prediction_df` with columns `LoanID` and `predicted_probability`. The checked-in file `prediction_submission.csv` is that output.

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn (SMOTE), Matplotlib, Jupyter.
