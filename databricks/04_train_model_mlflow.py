# Databricks notebook source
from pyspark.sql import functions as F

gold_df = spark.table("workspace.default.bist100_gold_features")

print("Gold rows:", gold_df.count())
print("Gold columns:", len(gold_df.columns))

gold_df.printSchema()
display(gold_df.limit(10))

# COMMAND ----------

import pandas as pd
import numpy as np

gold_pdf = gold_df.toPandas()
gold_pdf["date"] = pd.to_datetime(gold_pdf["date"])

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
    "market_cap_mn_tl",
    "market_cap_mn_usd",
    "free_float_market_cap_mn_tl",
    "free_float_market_cap_mn_usd",
    "capital_mn_tl",
]

target_col = "target_outperform_20d"

train_df = gold_pdf[gold_pdf["date"] < "2021-01-01"].copy()
valid_df = gold_pdf[
    (gold_pdf["date"] >= "2021-01-01") & 
    (gold_pdf["date"] < "2023-01-01")
].copy()
test_df = gold_pdf[gold_pdf["date"] >= "2023-01-01"].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_valid = valid_df[feature_cols]
y_valid = valid_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("Train shape:", X_train.shape)
print("Validation shape:", X_valid.shape)
print("Test shape:", X_test.shape)

print("\nTrain target distribution:")
print(y_train.value_counts(normalize=True).sort_index())

print("\nValidation target distribution:")
print(y_valid.value_counts(normalize=True).sort_index())

print("\nTest target distribution:")
print(y_test.value_counts(normalize=True).sort_index())

# COMMAND ----------

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# COMMAND ----------

mlflow.set_experiment("/Users/muhammed.altindal27@ncf.edu/bist100_outperformance_prediction")

print("MLflow experiment is ready.")

# COMMAND ----------

def evaluate_model(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


print("Evaluation function is ready.")

# COMMAND ----------

models = {
    "logistic_regression": Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
best_model = None
best_model_name = None
best_valid_auc = -1

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        valid_metrics = evaluate_model(model, X_valid, y_valid)
        test_metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("target", target_col)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_valid))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("features", ",".join(feature_cols))

        for metric_name, metric_value in valid_metrics.items():
            mlflow.log_metric(f"validation_{metric_name}", metric_value)

        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        mlflow.sklearn.log_model(model, artifact_path="model")

        row = {
            "model": model_name,
            **{f"validation_{k}": v for k, v in valid_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        results.append(row)

        print("\nModel:", model_name)
        print("Validation metrics:", valid_metrics)
        print("Test metrics:", test_metrics)

        if valid_metrics["roc_auc"] > best_valid_auc:
            best_valid_auc = valid_metrics["roc_auc"]
            best_model = model
            best_model_name = model_name

results_df = pd.DataFrame(results)

print("\nBest model based on validation ROC-AUC:", best_model_name)
display(results_df)

# COMMAND ----------

import joblib
import json
from pathlib import Path

EXPORT_DIR = Path("/Volumes/workspace/default/bist100_raw/model_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

model_path = EXPORT_DIR / "bist100_model.pkl"
features_path = EXPORT_DIR / "feature_columns.json"

joblib.dump(best_model, model_path)

with open(features_path, "w") as f:
    json.dump(feature_cols, f, indent=2)

print("Best model:", best_model_name)
print("Model saved to:", model_path)
print("Feature columns saved to:", features_path)

# COMMAND ----------

gold_export_path = "/Volumes/workspace/default/bist100_raw/model_exports/bist100_gold_features.csv"

gold_pdf.to_csv(gold_export_path, index=False)

print("Gold features exported to:")
print(gold_export_path)

print("Exported shape:", gold_pdf.shape)