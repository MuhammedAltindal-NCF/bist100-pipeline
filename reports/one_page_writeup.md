BIST100 Stock Outperformance Predictor

1. What It Predicts

This project predicts whether a selected BIST100 stock will outperform the BIST100 index over the next 20 trading days. The task is framed as a supervised binary classification problem. The target equals 1 if the stock’s future 20-day return is greater than the BIST100’s future 20-day return, and 0 otherwise.

2. Data Source

The dataset consists of historical BIST100 stock-level Excel files collected from Türkiye İş Bankası’s publicly available market data website. Each file represents one stock and includes daily price, volume, USDTRY, BIST100 index level, market value, free-float market value, and capital information. Since the original files did not include a ticker column, ticker symbols were extracted from filenames during ingestion. The project uses the current BIST100 constituent list, so survivorship bias is a limitation.

3. Pipeline Architecture

The project uses Databricks and Delta tables to build a medallion-style pipeline. Raw Excel files are uploaded to a Databricks volume. The Bronze layer combines all raw files and adds ticker information. The Silver layer cleans column names, converts dates and numeric fields, and standardizes the dataset. The Gold layer creates machine-learning features such as stock returns, rolling volatility, moving-average ratios, volume ratios, USDTRY returns, BIST100 returns, and the future outperformance target.

Pipeline flow:

Raw Excel files → Databricks Volume → Bronze Delta table → Silver Delta table → Gold feature table → MLflow model training → Saved artifacts → Streamlit app → Test script

4. Model Approach

I trained Logistic Regression and Random Forest models and tracked the runs in Databricks MLflow. The data was split using a time-based approach to reduce look-ahead bias: before 2021 for training, 2021–2022 for validation, and 2023 onward for testing. Logistic Regression was selected based on validation ROC-AUC. On the test set, the model achieved approximately 0.493 accuracy, 0.574 F1-score, and 0.505 ROC-AUC. These results show that short-term stock outperformance is difficult to predict, but the project provides a complete reproducible baseline.

5. What I Learned

The hardest part was transforming many individual stock Excel files into one consistent time-series dataset without mixing tickers or introducing look-ahead bias. I also learned that financial prediction requires careful target construction, time-based validation, and honest reporting of weak results. If I improved the project, I would add historical BIST100 membership, sector information, macroeconomic variables, rolling beta, and portfolio-level evaluation.