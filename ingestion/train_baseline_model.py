from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bist100_gold_features.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

MODEL_FILE = ARTIFACT_DIR / "bist100_model.pkl"
FEATURES_FILE = ARTIFACT_DIR / "feature_columns.json"
PREDICTIONS_FILE = PROJECT_ROOT / "data" / "processed" / "model_predictions_test.csv"


# -------------------------------------------------------
# Main training
# -------------------------------------------------------

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading gold feature dataset...")
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of tickers: {df['ticker'].nunique()}")

    target_col = "target_outperform_20d"

    feature_cols = [
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "price_to_ma20",
        "price_to_ma50",
        "usdtry_return_20d",
        "bist100_return_20d",
        "volume",
        "usdtry",
        "piyasadegeri_mn_tl",
        "piyasadegeri_mn_usd",
        "halkaacik_pd_mn_tl",
        "halkaacik_pd_mn_usd",
        "sermaye_mn_tl",
    ]

    # Keep only features that exist in the dataset
    feature_cols = [col for col in feature_cols if col in df.columns]

    print("\nFeatures used:")
    for col in feature_cols:
        print(f"- {col}")

    # ---------------------------------------------------
    # Time-based split
    # ---------------------------------------------------

    train_df = df[df["date"] < "2021-01-01"].copy()
    valid_df = df[(df["date"] >= "2021-01-01") & (df["date"] < "2023-01-01")].copy()
    test_df = df[df["date"] >= "2023-01-01"].copy()

    print("\nSplit sizes:")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {valid_df.shape}")
    print(f"Test: {test_df.shape}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # ---------------------------------------------------
    # Benchmark model: Logistic Regression
    # ---------------------------------------------------

    print("\nTraining Logistic Regression benchmark...")

    logistic_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_model.fit(X_train, y_train)

    valid_pred_lr = logistic_model.predict(X_valid)
    valid_prob_lr = logistic_model.predict_proba(X_valid)[:, 1]

    print("\nLogistic Regression Validation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_valid, valid_pred_lr):.4f}")
    print(f"Precision: {precision_score(y_valid, valid_pred_lr):.4f}")
    print(f"Recall:    {recall_score(y_valid, valid_pred_lr):.4f}")
    print(f"F1:        {f1_score(y_valid, valid_pred_lr):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_valid, valid_prob_lr):.4f}")

    # ---------------------------------------------------
    # ML model: Random Forest
    # ---------------------------------------------------

    print("\nTraining Random Forest model...")

    random_forest_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=50,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    random_forest_model.fit(X_train, y_train)

    valid_pred_rf = random_forest_model.predict(X_valid)
    valid_prob_rf = random_forest_model.predict_proba(X_valid)[:, 1]

    print("\nRandom Forest Validation Metrics:")
    print(f"Accuracy:  {accuracy_score(y_valid, valid_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_valid, valid_pred_rf):.4f}")
    print(f"Recall:    {recall_score(y_valid, valid_pred_rf):.4f}")
    print(f"F1:        {f1_score(y_valid, valid_pred_rf):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_valid, valid_prob_rf):.4f}")

    # ---------------------------------------------------
    # Choose model based on validation ROC-AUC
    # ---------------------------------------------------

    lr_auc = roc_auc_score(y_valid, valid_prob_lr)
    rf_auc = roc_auc_score(y_valid, valid_prob_rf)

    if rf_auc >= lr_auc:
        best_model = random_forest_model
        best_model_name = "Random Forest"
    else:
        best_model = logistic_model
        best_model_name = "Logistic Regression"

    print(f"\nBest model based on validation ROC-AUC: {best_model_name}")

    # ---------------------------------------------------
    # Evaluate best model on test set
    # ---------------------------------------------------

    test_pred = best_model.predict(X_test)
    test_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nBest Model Test Metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"Precision: {precision_score(y_test, test_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, test_pred):.4f}")
    print(f"F1:        {f1_score(y_test, test_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # ---------------------------------------------------
    # Save predictions for later UI/backtest
    # ---------------------------------------------------

    predictions = test_df[
        [
            "date",
            "ticker",
            "close",
            "bist100",
            "future_return_20d",
            "future_bist100_return_20d",
            target_col,
        ]
    ].copy()

    predictions["predicted_probability"] = test_prob
    predictions["predicted_class"] = test_pred
    predictions["model_name"] = best_model_name

    predictions.to_csv(PREDICTIONS_FILE, index=False)

    # ---------------------------------------------------
    # Save model and features
    # ---------------------------------------------------

    joblib.dump(best_model, MODEL_FILE)

    with open(FEATURES_FILE, "w") as f:
        json.dump(feature_cols, f, indent=2)

    print("\nSaved artifacts:")
    print(f"Model: {MODEL_FILE}")
    print(f"Features: {FEATURES_FILE}")
    print(f"Predictions: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()
