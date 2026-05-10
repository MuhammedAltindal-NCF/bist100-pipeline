# BIST100 Stock Outperformance Predictor

## 1. What It Predicts

This project predicts whether a selected BIST100 stock is likely to outperform the BIST100 index over the next 20 trading days. The prediction is framed as a supervised binary classification problem. The target variable equals 1 if the stock’s future 20-day return is greater than the BIST100’s future 20-day return, and 0 otherwise.

## 2. Data Source

The dataset consists of historical BIST100 stock-level Excel files collected from Bloomberg/Excel. Each file represents one stock and contains daily price, volume, USDTRY, BIST100 index level, market value, free-float market value, and capital information. The files were based on the current BIST100 constituent list, so historical coverage differs by stock. This creates a possible survivorship bias, which is treated as a limitation of the project.

## 3. Pipeline Architecture

The pipeline starts with local Excel files and converts them into one combined raw dataset. A Python ingestion script extracts the ticker from each filename, adds it as a column, standardizes column names, and saves the result as a raw combined CSV. A second feature-engineering script creates a gold feature table with stock returns, rolling volatility, moving-average ratios, volume ratios, USDTRY returns, BIST100 returns, and the future outperformance target. A model-training script then trains supervised classification models and saves the selected model and feature list as artifacts. Finally, a Streamlit application loads the saved model and gold dataset to serve predictions through a Streamlit web application interface.

Pipeline flow:

Local Excel files → Raw combined table → Gold feature table → ML model → Saved artifacts → Streamlit app → Test script

## 4. Model Approach

The project uses Logistic Regression as the benchmark model and Random Forest as a more flexible comparison model. The data is split using a time-based approach to reduce look-ahead bias: observations before 2021 are used for training, 2021–2022 for validation, and 2023 onward for testing. The selected model is Logistic Regression based on validation ROC-AUC. On the test set, the model achieved approximately 0.493 accuracy, 0.574 F1-score, and 0.505 ROC-AUC. These results suggest that short-term stock outperformance is difficult to predict with the current feature set, but the model provides a reproducible baseline.

## 5. What I Learned

The hardest part was transforming many individual stock Excel files into one consistent time-series dataset without mixing tickers or introducing look-ahead bias. I also learned that financial prediction requires careful target construction, time-based validation, and honest reporting of weak out-of-sample results. If I rebuilt the project, I would add historical index membership, sector information, macroeconomic variables, and more advanced portfolio-level evaluation.
