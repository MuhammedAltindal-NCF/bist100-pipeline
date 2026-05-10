from pathlib import Path
import pandas as pd
import numpy as np


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bist100_all_stocks_raw.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "bist100_gold_features.csv"


# -------------------------------------------------------
# Main feature engineering
# -------------------------------------------------------

def main():
    print("Reading raw combined dataset...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Raw shape: {df.shape}")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Sort correctly for time series calculations
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Keep only rows with essential fields
    essential_cols = ["date", "ticker", "close", "bist100"]
    df = df.dropna(subset=essential_cols)

    # ---------------------------------------------------
    # Feature engineering by ticker
    # ---------------------------------------------------

    print("Creating stock-level features...")

    g = df.groupby("ticker", group_keys=False)

    # Stock returns
    df["return_1d"] = g["close"].pct_change(1)
    df["return_5d"] = g["close"].pct_change(5)
    df["return_20d"] = g["close"].pct_change(20)

    # Rolling volatility based on daily returns
    df["volatility_20d"] = (
        g["return_1d"]
        .rolling(window=20, min_periods=20)
        .std()
        .reset_index(level=0, drop=True)
    )

    # Moving averages
    df["ma20"] = (
        g["close"]
        .rolling(window=20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["ma50"] = (
        g["close"]
        .rolling(window=50, min_periods=50)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["price_to_ma20"] = df["close"] / df["ma20"]
    df["price_to_ma50"] = df["close"] / df["ma50"]

    # Volume ratio
    if "volume" in df.columns:
        df["volume_ma20"] = (
            g["volume"]
            .rolling(window=20, min_periods=20)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df["volume_ratio_20d"] = df["volume"] / df["volume_ma20"]
    else:
        df["volume_ratio_20d"] = np.nan

    # USDTRY return
    if "usdtry" in df.columns:
        df["usdtry_return_20d"] = g["usdtry"].pct_change(20)
    else:
        df["usdtry_return_20d"] = np.nan

    # BIST100 return
    df["bist100_return_20d"] = g["bist100"].pct_change(20)

    # Future returns for target
    df["future_return_20d"] = g["close"].shift(-20) / df["close"] - 1
    df["future_bist100_return_20d"] = g["bist100"].shift(-20) / df["bist100"] - 1

    # Target: stock outperforms BIST100 over next 20 trading days
    df["target_outperform_20d"] = (
        df["future_return_20d"] > df["future_bist100_return_20d"]
    ).astype(int)

    # ---------------------------------------------------
    # Optional valuation and size features
    # ---------------------------------------------------

    # Keep useful original columns if they exist
    useful_optional_cols = [
        "volume",
        "usdtry",
        "piyasadegeri_mn_tl",
        "piyasadegeri_mn_usd",
        "halkaacik_pd_mn_tl",
        "halkaacik_pd_mn_usd",
        "sermaye_mn_tl",
    ]

    selected_cols = [
        "date",
        "ticker",
        "close",
        "bist100",
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "price_to_ma20",
        "price_to_ma50",
        "usdtry_return_20d",
        "bist100_return_20d",
        "future_return_20d",
        "future_bist100_return_20d",
        "target_outperform_20d",
    ]

    for col in useful_optional_cols:
        if col in df.columns and col not in selected_cols:
            selected_cols.append(col)

    gold = df[selected_cols].copy()

    # Remove rows where model features or target cannot be calculated
    model_required_cols = [
        "return_5d",
        "return_20d",
        "volatility_20d",
        "volume_ratio_20d",
        "price_to_ma20",
        "price_to_ma50",
        "usdtry_return_20d",
        "bist100_return_20d",
        "future_return_20d",
        "future_bist100_return_20d",
    ]

    gold = gold.dropna(subset=model_required_cols)

    # Remove infinite values
    gold = gold.replace([np.inf, -np.inf], np.nan)
    gold = gold.dropna(subset=model_required_cols)

    # Save
    gold.to_csv(OUTPUT_FILE, index=False)

    print("\nDONE")
    print(f"Gold shape: {gold.shape}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nTarget distribution:")
    print(gold["target_outperform_20d"].value_counts(normalize=True))

    print("\nDate range:")
    print(gold["date"].min(), "to", gold["date"].max())

    print("\nNumber of tickers:")
    print(gold["ticker"].nunique())

    print("\nPreview:")
    print(gold.head())


if __name__ == "__main__":
    main()
