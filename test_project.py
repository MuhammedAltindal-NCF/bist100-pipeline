from pathlib import Path
import json
import sys

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bist100_gold_features.csv"
MODEL_FILE = PROJECT_ROOT / "artifacts" / "bist100_model.pkl"
FEATURES_FILE = PROJECT_ROOT / "artifacts" / "feature_columns.json"


def main():
    print("BIST100 Outperformance Predictor Test")
    print("=" * 40)

    try:
        df = pd.read_csv(DATA_FILE)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        model = joblib.load(MODEL_FILE)

        with open(FEATURES_FILE, "r") as f:
            feature_cols = json.load(f)

        latest_rows = (
            df.sort_values("date")
            .groupby("ticker", as_index=False)
            .tail(1)
            .copy()
        )

        if latest_rows.empty:
            print("FAIL: No data available for prediction.")
            sys.exit(1)

        sample = latest_rows.iloc[[0]]
        ticker = sample["ticker"].iloc[0]
        date = sample["date"].iloc[0]

        X = sample[feature_cols]

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0, 1])

        label = "Outperform" if prediction == 1 else "Not Outperform"

        print(f"Ticker: {ticker}")
        print(f"Date: {date}")
        print(f"Prediction: {prediction}")
        print(f"Label: {label}")
        print(f"Probability: {probability:.4f}")

        print("\nPASS")
        sys.exit(0)

    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
