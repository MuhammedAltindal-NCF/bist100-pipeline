# BIST100 Stock Outperformance Predictor

This project is an end-to-end distributed data pipeline and machine learning application for predicting whether a selected BIST100 stock is likely to outperform the BIST100 index over the next 20 trading days.

The project uses historical BIST100 stock data, transforms raw Excel files into a structured medallion architecture in Databricks, trains machine learning models with MLflow tracking, and serves the prediction results through a Streamlit web application.

---

## Project Goal

The main prediction question is:

**Will a selected BIST100 stock outperform the BIST100 index over the next 20 trading days?**

The target variable is binary:

- `1`: the stock's next 20-trading-day return is greater than the BIST100's next 20-trading-day return
- `0`: otherwise

The model uses historical price, volume, valuation, market index, and USDTRY-based features to estimate the probability that a stock will outperform the BIST100 benchmark.

This application is intended as a baseline academic predictive analytics system and should not be interpreted as financial advice.

---

## Live Application

Live Streamlit URL:

```text
https://bist100-pipeline.streamlit.app/

## Repository Structure

bist100-pipeline/
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── bist100_model.pkl
│   └── feature_columns.json
│
├── data/
│   └── processed/
│       └── bist100_gold_features.csv
│
├── databricks/
│   ├── 01_ingest_excel_to_bronze.py
│   ├── 02_silver_cleaning.py
│   ├── 03_gold_features.py
│   └── 04_train_model_mlflow.py
│
├── ingestion/
├── notebooks/
├── reports/
├── README.md
├── requirements.txt
└── test_project.py

## Data Source

The raw dataset consists of 100 Excel files, one for each current BIST100 stock. Each file contains historical daily stock-level market data, including:

date
close price
minimum price
maximum price
average price
volume
capital
USDTRY
BIST100 index value
market capitalization
free-float market capitalization

The original Excel files did not contain a ticker column. During ingestion, the ticker symbol was extracted from each Excel filename and added to the dataset before the files were combined.

Example:
AEFES2011010120260422.xlsx → AEFES
AKBNK2011010120260422.xlsx → AKBNK
BIMAS2011010120260422.xlsx → BIMAS

## Distributed Pipeline Architecture

The project follows a medallion architecture using Databricks and Delta tables.

Raw Excel Files
    ↓
Databricks Volume
    ↓
Bronze Delta Table
    ↓
Silver Delta Table
    ↓
Gold Feature Table
    ↓
MLflow Model Training
    ↓
Exported Model Artifacts
    ↓
Streamlit Application
    ↓
Test Script

## Raw Storage

The 100 raw Excel files were uploaded to a Databricks Unity Catalog volume:
/Volumes/workspace/default/bist100_raw/

This gives the project a cloud/distributed storage layer instead of relying only on local files.

## Bronze Layer

The Bronze layer ingests the raw Excel files from the Databricks volume.

Main tasks:

read all Excel files
extract ticker symbols from filenames
add ticker and source_file columns
combine all stock files into one dataset
clean raw column names into machine-readable format
save the result as a Delta table

Bronze table:workspace.default.bist100_bronze

Bronze output:
Rows: 278,468
Columns: 15

## Silver Layer

The Silver layer cleans and standardizes the Bronze data.

Main tasks:

rename Turkish column names into English
convert date fields into proper date type
cast numeric fields into numeric types
remove duplicate ticker-date rows
remove invalid rows with missing date, ticker, or close price

Silver table:

workspace.default.bist100_silver

Silver output:

Rows: 278,468
Columns: 15

Silver columns include:

date
close
min_price
max_price
avg_price
volume
capital_mn_tl
usdtry
bist100
market_cap_mn_tl
market_cap_mn_usd
free_float_market_cap_mn_tl
free_float_market_cap_mn_usd
ticker
source_file

## Gold Layer

The Gold layer creates the machine-learning-ready feature table.

Gold table:

workspace.default.bist100_gold_features

Gold output:

Rows: 251,410
Columns: 21

The Gold table is used for model training, prediction, and the Streamlit application.

## Feature Engineering

The model uses historical stock, market, volume, valuation, and currency-based features.

Main features include:

return_5d
return_20d
volatility_20d
volume_ratio_20d
price_to_ma20
price_to_ma50
usdtry_return_20d
bist100_return_20d
volume
usdtry
market_cap_mn_tl
market_cap_mn_usd
free_float_market_cap_mn_tl
free_float_market_cap_mn_usd
capital_mn_tl

The prediction target is:

target_outperform_20d = 1 if future_stock_return_20d > future_bist100_return_20d
target_outperform_20d = 0 otherwise
Machine Learning Approach

This project is a supervised binary classification task.

Two models were trained and compared:

Logistic Regression
used as the benchmark model
selected based on validation ROC-AUC
Random Forest
used as a non-linear comparison model

The data was split using a time-based split instead of a random split:

Train:       before 2021
Validation: 2021–2022
Test:       2023 and later

A time-based split is used to reduce look-ahead bias and better reflect the real forecasting setting.

## MLflow Tracking

Model training was tracked in Databricks MLflow.

Experiment:

bist100_outperformance_prediction

MLflow runs:

logistic_regression
random_forest

For each run, the project logged:

model name
target variable
number of train rows
number of validation rows
number of test rows
number of features
validation accuracy
validation precision
validation recall
validation F1
validation ROC-AUC
test accuracy
test precision
test recall
test F1
test ROC-AUC
trained model artifact

## Model Results

The selected model was Logistic Regression based on validation ROC-AUC.

Approximate results:

Model	Validation ROC-AUC	Test ROC-AUC
Logistic Regression	0.5225	0.4982
Random Forest	0.5135	0.5190

The model results show weak out-of-sample predictive power, which is expected in short-term financial prediction. The main value of this project is the complete distributed pipeline and reproducible model-serving workflow rather than producing a production trading strategy.

Exported Artifacts

After MLflow training, the selected model and feature list were exported for use in the Streamlit application.

Exported files:

artifacts/bist100_model.pkl
artifacts/feature_columns.json
data/processed/bist100_gold_features.csv

These files allow the Streamlit app and test script to load the trained model and generate predictions.

## Streamlit Application

The Streamlit application provides three main sections:

1. Single Stock Prediction

The user can select a BIST100 ticker and view:

predicted probability of outperforming BIST100
predicted class: Outperform or Not Outperform
latest close price
historical close price chart
latest feature values used by the model
2. BIST100-Wide Comparison

This section compares multiple stocks visually using a normalized multi-line performance chart.

Key design choices:

each stock starts at 100 on its own first available date
stocks that entered the dataset later appear only from the date their data begins
each stock is shown with a different line color
hover shows only the selected stock's details
log scale can be enabled for readability when very high long-term returns dominate the chart

The app also shows:

historical best performer in the selected comparison group
top historical performers by normalized index
latest model prediction ranking
data availability by ticker
3. Model Notes

This section explains:

prediction target
selected model
alternative model
Databricks pipeline
limitations
interpretation notes
How to Install Requirements

Install the required Python packages:

pip install -r requirements.txt

The main packages used are:

pandas
numpy
scikit-learn
joblib
streamlit
plotly
openpyxl

The project uses scikit-learn==1.6.1 because the exported model artifact was trained and saved with that version.

How to Run the Streamlit App

From the project root directory, run:

streamlit run app/streamlit_app.py

Then open the local Streamlit URL shown in the terminal.

How to Run the Test Script

The test script verifies that the exported model, feature list, and Gold dataset can be loaded successfully. It then produces one sample prediction and prints a PASS message if the prediction pipeline works.

Run:

python test_project.py

Example output:

BIST100 Outperformance Predictor Test
========================================
Ticker: TSKB
Date: 2026-03-25 00:00:00
Prediction: 0
Label: Not Outperform
Probability: 0.0000

PASS

A successful run exits with code 0 and prints PASS.

## Important Limitations

1. Survivorship Bias

The project uses the current BIST100 constituent list. It does not reconstruct historical BIST100 membership. This may introduce survivorship bias.

2. Weak Predictive Signal

The test ROC-AUC is close to 0.50, which suggests that the current feature set has limited out-of-sample predictive power for short-term stock outperformance.

3. Financial Market Difficulty

Short-term stock prediction is inherently difficult. Prices are affected by many factors that are not fully captured in this dataset, including macroeconomic conditions, interest rates, inflation, political events, company news, and investor sentiment.

4. Long-Term Return Interpretation

Very high normalized returns may occur when a stock starts from a very low historical price or when price history is affected by corporate actions such as splits, bonus issues, or adjustment differences. These results should be interpreted carefully.

5. Not Financial Advice

This project is an academic prototype and should not be used for real trading decisions.

Future Improvements

Possible future improvements include:

reconstructing historical BIST100 constituent membership
adding sector classifications
adding sector-relative valuation features
including macroeconomic variables such as inflation, interest rates, and exchange-rate volatility
adding rolling beta against BIST100
adding risk-adjusted return metrics
adding portfolio-level backtesting
adding Databricks Model Registry stage management
serving the model through a REST API endpoint
testing more advanced models such as XGBoost or LightGBM

What I Learned

The most challenging part of the project was transforming many individual stock Excel files into one consistent time-series dataset without mixing tickers or introducing look-ahead bias.

I also learned that financial prediction requires careful target construction, time-based validation, and honest reporting of model limitations. Even when predictive accuracy is weak, a well-designed pipeline is valuable because it makes the data ingestion, transformation, modeling, and serving workflow reproducible from end to end.

Final Summary

This project demonstrates the full path from raw stock-level Excel files to a working machine learning application:

Raw Excel files
→ Databricks Volume
→ Bronze Delta table
→ Silver Delta table
→ Gold feature table
→ MLflow-tracked model
→ Exported model artifacts
→ Streamlit web application
→ Test script

The result is a reproducible distributed data pipeline with predictive intelligence and an interactive web interface for exploring BIST100 stock outperformance predictions.