# Databricks notebook source
from pathlib import Path

RAW_DIR = Path("/Volumes/workspace/default/bist100_raw")

files = sorted(RAW_DIR.glob("*.xlsx"))

print("Number of Excel files:", len(files))
print("First 5 files:")
for f in files[:5]:
    print(f.name)

# COMMAND ----------

# MAGIC %pip install openpyxl

# COMMAND ----------

import pandas as pd
import re
from pathlib import Path

RAW_DIR = Path("/Volumes/workspace/default/bist100_raw")

files = sorted(RAW_DIR.glob("*.xlsx"))

all_frames = []

for file in files:
    # Extract ticker from filename: AEFES2011010120260422.xlsx -> AEFES
    ticker_match = re.match(r"([A-Z]+)", file.stem)
    ticker = ticker_match.group(1) if ticker_match else file.stem
    
    temp = pd.read_excel(file)
    temp["ticker"] = ticker
    temp["source_file"] = file.name
    
    all_frames.append(temp)

bronze_pdf = pd.concat(all_frames, ignore_index=True)

print("Bronze shape:", bronze_pdf.shape)
display(bronze_pdf.head())

# COMMAND ----------

import unicodedata
import re

def clean_col_name(col):
    col = str(col).strip()
    col = col.replace("ı", "i").replace("İ", "i")
    col = unicodedata.normalize("NFKD", col)
    col = "".join([c for c in col if not unicodedata.combining(c)])
    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")
    return col

bronze_pdf.columns = [clean_col_name(c) for c in bronze_pdf.columns]

print(bronze_pdf.columns.tolist())
display(bronze_pdf.head())

# COMMAND ----------

bronze_sdf = spark.createDataFrame(bronze_pdf)

bronze_sdf.write.format("delta").mode("overwrite").saveAsTable(
    "workspace.default.bist100_bronze"
)

print("Bronze Delta table created: workspace.default.bist100_bronze")
print("Rows:", bronze_sdf.count())
print("Columns:", len(bronze_sdf.columns))

display(bronze_sdf.limit(10))