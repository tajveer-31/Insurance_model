# Property Insurance Premium Prediction

Predicting the **annual premium** of a home insurance policy from policyholder, property and cover details, using regression models in a single Jupyter notebook. The project also looks at overfitting: which models memorise the training data, and how much hyperparameter tuning helps.

## Contents

| File | Description |
|---|---|
| `Property_insurance_premium.ipynb` | Full workflow: cleaning, feature engineering, EDA, modelling, tuning |
| `home_insurance_dataset.csv` | Input data (not included here; place it next to the notebook) |
| `best_premium_model.pkl` | Created by the last cell: best tuned model, scaler and column list |

## Dataset

- 256,136 policies × 42 columns; **188,988 rows remain after cleaning**.
- **Target:** `ANNUAL_PREMIUM`.
- Feature groups: policyholder (date of birth, employment, marital status, gender), property (year built, bedrooms, roof/wall construction, listed status), cover (buildings/contents cover, sums insured, no-claims discount years), security (alarm, locks, safe, neighbourhood watch) and risk (flooding, subsidence, risk-rated areas).

## Workflow

1. **Cleaning**
   - Parsed date columns in mixed formats.
   - Dropped the 67,115 rows that were missing all the core policy fields.
   - Filled `PAYMENT_FREQUENCY` with 2.
   - Imputed `RISK_RATED_AREA_B` and `RISK_RATED_AREA_C` with the median.
   - Forward-filled `QUOTE_DATE`.
   - Removed policies with a premium of 0 or less.
2. **Feature engineering**
   - Created `AGE` and `PROPERTY_AGE` at cover start.
   - Split dates into year and month features.
   - Created `Year_of_Construction`.
   - Dropped the raw date columns.
3. **Encoding and scaling**
   - Y/N columns became 0/1.
   - One-hot encoded `OCC_STATUS`, `P1_EMP_STATUS`, `P1_MAR_STATUS`, `P1_GENDER` and `PAYMENT_METHOD`.
   - Scaled numeric features with `StandardScaler`. This gives 71 model features.
4. **EDA**: premium distribution, employment status, build year, gender and payment method counts, premium vs. age and property age, correlation heatmap.
5. **Modelling:** 80/20 train/test split (`random_state=42`).
   - Decision Tree, Linear Regression, Ridge, Lasso and Random Forest, with train and test R² compared.
   - Gradient Boosting, XGBoost, LightGBM and SVR are also included, but their evaluation cells error with current scikit-learn (see Known issues).
6. **Hyperparameter tuning:** `RandomizedSearchCV` with 3-fold CV on a 40k-row sample of the training set. The best settings are refit on the full training set, models are ranked by CV score, and the winner is saved with `joblib`.

## Results

Test-set R² before and after tuning:

| Model | Train R² (before → after) | Test R² (before → after) | Train–test gap (before → after) |
|---|---|---|---|
| Decision Tree | 1.000 → 0.722 | 0.452 → 0.652 | 0.549 → 0.070 |
| Random Forest | 0.963 → 0.784 | 0.698 → 0.698 | 0.266 → 0.086 |
| Ridge | 0.626 → 0.626 | 0.593 → 0.593 | 0.033 → 0.033 |
| Lasso | 0.626 → 0.626 | 0.592 → 0.593 | 0.033 → 0.033 |

**Best model by CV: Random Forest.** Test R² 0.698, MAE ≈ 32.3, RMSE ≈ 55.4.

What this shows:
- **The linear models were never overfitting.** Train and test R² are within 0.03 of each other, so Ridge and Lasso regularization changes nothing. The 0.59 test R² is roughly the ceiling for a linear model on these features.
- **The Decision Tree and Random Forest were overfitting.** Tuning cut their train–test gaps sharply (Decision Tree 0.55 → 0.07). For the Decision Tree, test R² also improved a lot (0.45 → 0.65).
- **Tuning the Random Forest closed the gap but did not raise test R².** It stayed at about 0.70, so tuning made it more stable rather than more accurate.

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost lightgbm joblib jupyter
jupyter notebook Property_insurance_premium.ipynb
```

Put `home_insurance_dataset.csv` in the same folder and run the cells top to bottom. Tuning is the slow part. In the reported run, Lasso took about 4 minutes and the Random Forest search about 2 minutes. To speed it up, lower `SEARCH_ROWS` or `n_iter`.

## Using the saved model

```python
import joblib, pandas as pd

bundle = joblib.load("best_premium_model.pkl")
model, scaler = bundle["model"], bundle["scaler"]

# new_row: a one-row DataFrame prepared exactly like training
# (same cleaning, same one-hot columns, same column order)
new_row = new_row.reindex(columns=bundle["columns"], fill_value=0)
new_row[bundle["numerical_cols"]] = scaler.transform(new_row[bundle["numerical_cols"]])
print(model.predict(new_row))
```

## Known issues and next steps

- **Gradient Boosting, XGBoost, LightGBM and SVR evaluation cells fail on current scikit-learn.**
- **The scaler is fit on all rows before the train/test split.** This leaks a small amount of test information. Fit it on `x_train` only, ideally inside a `Pipeline`.
- **The "Training R²" cell under Ridge scores the Decision Tree** (`model`), not Ridge. Change it to `ridge`.
- **The split is random.** If policies repeat per customer or the data is time-ordered, use `GroupKFold` or a time-based split for a more honest estimate.
- **Rows with no `QUOTE_DATE` are forward-filled**, which assumes the file order carries meaning. Consider a missing-date flag instead.
- **Ideas:** tune the boosting models, try a log-transformed target (premiums are right-skewed), and check feature importances.