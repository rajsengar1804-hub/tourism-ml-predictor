# Tourism ML: Rating Regression + VisitMode Classification

This project uses the supplied `cleaned_tourism_data(1).csv` and intentionally does **not** start with EDA.

## 1. Rating regression
1. Feature engineering: categorical frequency encoding + month sine/cosine features; identifier columns TransactionId/UserId removed; VisitModeId removed as a duplicate code.
2. Feature selection: Variance Threshold → multicollinearity filter (absolute correlation > 0.95) → Random Forest model-based importance.
3. Baselines: Random Forest, XGBoost, CatBoost, LightGBM.
4. Select best baseline by lowest validation RMSE.
5. Hyperparameter tuning of the selected model.
6. Save `rating_regression_model.joblib`.

## 2. VisitMode classification
1. Feature engineering: same frequency/cyclical approach; `VisitMode` and `VisitModeId` are excluded to prevent target leakage; Rating is retained as a predictor.
2. Feature selection: Variance Threshold → multicollinearity filter → Random Forest model-based importance.
3. Baselines: Random Forest, XGBoost, CatBoost, LightGBM.
4. Compare `class_weight=balanced` against SMOTE using Macro F1/Balanced Accuracy.
5. Select the better balancing strategy.
6. Hyperparameter tuning.
7. Save `visitmode_classification_model.joblib`.

## 3. Streamlit
Run from this folder:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app has separate prediction modes for Rating and VisitMode and uses the exact saved feature-engineering objects used during training.

## Important result
The dataset contains 52,922 rows and 5 VisitMode classes: Business, Couples, Family, Friends and Solo. The classification experiment selected `class_weight=balanced` over SMOTE based on Macro F1.
