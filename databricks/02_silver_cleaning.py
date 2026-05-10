# Databricks notebook source
from pyspark.sql import functions as F

bronze_df = spark.table("workspace.default.bist100_bronze")

print("Bronze rows:", bronze_df.count())
print("Bronze columns:", len(bronze_df.columns))
print(bronze_df.columns)

display(bronze_df.limit(5))

# COMMAND ----------

# Rename Turkish/cleaned bronze columns into English silver-layer column names

rename_map = {
    "tarih": "date",
    "kapanis_tl": "close",
    "min_tl": "min_price",
    "max_tl": "max_price",
    "aof_tl": "avg_price",
    "hacim_tl": "volume",
    "sermaye_mn_tl": "capital_mn_tl",
    "usdtry": "usdtry",
    "bist_100": "bist100",
    "piyasadegeri_mn_tl": "market_cap_mn_tl",
    "piyasadegeri_mn_usd": "market_cap_mn_usd",
    "halkaacik_pd_mn_tl": "free_float_market_cap_mn_tl",
    "halkaacik_pd_mn_usd": "free_float_market_cap_mn_usd",
    "ticker": "ticker",
    "source_file": "source_file",
}

silver_df = bronze_df

for old_name, new_name in rename_map.items():
    if old_name in silver_df.columns:
        silver_df = silver_df.withColumnRenamed(old_name, new_name)

print("Silver columns:")
print(silver_df.columns)

display(silver_df.limit(5))

# COMMAND ----------

from pyspark.sql import functions as F

# Convert date from dd-MM-yyyy string to date type
silver_df = silver_df.withColumn(
    "date",
    F.to_date(F.col("date"), "dd-MM-yyyy")
)

# Cast numeric columns to double
numeric_cols = [
    "close",
    "min_price",
    "max_price",
    "avg_price",
    "volume",
    "capital_mn_tl",
    "usdtry",
    "bist100",
    "market_cap_mn_tl",
    "market_cap_mn_usd",
    "free_float_market_cap_mn_tl",
    "free_float_market_cap_mn_usd",
]

for c in numeric_cols:
    silver_df = silver_df.withColumn(c, F.col(c).cast("double"))

# Basic cleaning
silver_df = (
    silver_df
    .dropDuplicates(["date", "ticker"])
    .filter(F.col("date").isNotNull())
    .filter(F.col("ticker").isNotNull())
    .filter(F.col("close").isNotNull())
)

print("Silver rows:", silver_df.count())
print("Silver columns:", len(silver_df.columns))

silver_df.printSchema()
display(silver_df.limit(10))

# COMMAND ----------

silver_df.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.bist100_silver"
)

print("Silver Delta table created: workspace.default.bist100_silver")
print("Rows:", silver_df.count())
print("Columns:", len(silver_df.columns))

display(silver_df.limit(10))