# Databricks notebook source
from pyspark.sql import functions as F

silver_df = spark.table("workspace.default.bist100_silver")

print("Silver rows:", silver_df.count())
print("Silver columns:", len(silver_df.columns))

silver_df.printSchema()
display(silver_df.limit(10))

# COMMAND ----------

import pandas as pd
import numpy as np

pdf = silver_df.toPandas()

pdf["date"] = pd.to_datetime(pdf["date"])
pdf = pdf.sort_values(["ticker", "date"]).reset_index(drop=True)

print("Pandas shape:", pdf.shape)
print("Date range:", pdf["date"].min(), "to", pdf["date"].max())
print("Number of tickers:", pdf["ticker"].nunique())

display(pdf.head())

# COMMAND ----------

# Stock-level historical features
pdf["return_1d"] = pdf.groupby("ticker")["close"].pct_change(1)
pdf["return_5d"] = pdf.groupby("ticker")["close"].pct_change(5)
pdf["return_20d"] = pdf.groupby("ticker")["close"].pct_change(20)

pdf["volatility_20d"] = (
    pdf.groupby("ticker")["return_1d"]
    .rolling(20)
    .std()
    .reset_index(level=0, drop=True)
)

pdf["ma20"] = (
    pdf.groupby("ticker")["close"]
    .rolling(20)
    .mean()
    .reset_index(level=0, drop=True)
)

pdf["ma50"] = (
    pdf.groupby("ticker")["close"]
    .rolling(50)
    .mean()
    .reset_index(level=0, drop=True)
)

pdf["price_to_ma20"] = pdf["close"] / pdf["ma20"]
pdf["price_to_ma50"] = pdf["close"] / pdf["ma50"]

pdf["volume_ma20"] = (
    pdf.groupby("ticker")["volume"]
    .rolling(20)
    .mean()
    .reset_index(level=0, drop=True)
)

pdf["volume_ratio_20d"] = pdf["volume"] / pdf["volume_ma20"]

# Market and FX features
pdf["usdtry_return_20d"] = pdf.groupby("ticker")["usdtry"].pct_change(20)
pdf["bist100_return_20d"] = pdf.groupby("ticker")["bist100"].pct_change(20)

print("Features created.")
display(
    pdf[
        [
            "date",
            "ticker",
            "close",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "price_to_ma20",
            "volume_ratio_20d",
            "usdtry_return_20d",
            "bist100_return_20d",
        ]
    ].head(10)
)

# COMMAND ----------

# Future 20-trading-day returns
pdf["future_return_20d"] = (
    pdf.groupby("ticker")["close"]
    .shift(-20) / pdf["close"] - 1
)

pdf["future_bist100_return_20d"] = (
    pdf.groupby("ticker")["bist100"]
    .shift(-20) / pdf["bist100"] - 1
)

# Binary target:
# 1 = stock outperforms BIST100 over next 20 trading days
# 0 = otherwise
pdf["target_outperform_20d"] = (
    pdf["future_return_20d"] > pdf["future_bist100_return_20d"]
).astype(int)

print("Target created.")

print("Target distribution:")
print(pdf["target_outperform_20d"].value_counts(normalize=True).sort_index())

display(
    pdf[
        [
            "date",
            "ticker",
            "close",
            "future_return_20d",
            "future_bist100_return_20d",
            "target_outperform_20d",
        ]
    ].dropna().head(10)
)

# COMMAND ----------

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

base_cols = [
    "date",
    "ticker",
    "close",
    "future_return_20d",
    "future_bist100_return_20d",
    "target_outperform_20d",
]

gold_pdf = pdf[base_cols + feature_cols].dropna().copy()

print("Gold shape:", gold_pdf.shape)
print("Date range:", gold_pdf["date"].min(), "to", gold_pdf["date"].max())
print("Number of tickers:", gold_pdf["ticker"].nunique())

print("Target distribution:")
print(gold_pdf["target_outperform_20d"].value_counts(normalize=True).sort_index())

display(gold_pdf.head(10))

# COMMAND ----------

gold_sdf = spark.createDataFrame(gold_pdf)

gold_sdf.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.bist100_gold_features"
)

print("Gold Delta table created: workspace.default.bist100_gold_features")
print("Rows:", gold_sdf.count())
print("Columns:", len(gold_sdf.columns))

display(gold_sdf.limit(10))

# COMMAND ----------

from pyspark.sql import functions as F

gold_df = spark.table("workspace.default.bist100_gold_features")

print("Gold rows:", gold_df.count())
print("Gold columns:", len(gold_df.columns))

gold_df.printSchema()
display(gold_df.limit(10))