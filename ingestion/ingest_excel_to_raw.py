from pathlib import Path
import re
import pandas as pd
import numpy as np


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_EXCEL_DIR = PROJECT_ROOT / "data" / "raw_excel"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "bist100_all_stocks_raw.csv"


# -------------------------------------------------------
# Helper functions
# -------------------------------------------------------

def extract_ticker(filename: str) -> str:
    """
    Extract ticker from filename.

    Example:
    BIMAS2011010120260422.xlsx -> BIMAS
    GARAN2011010120260422.xlsx -> GARAN
    """
    stem = Path(filename).stem
    match = re.match(r"([A-Z]+)", stem)
    if match:
        return match.group(1)
    return stem


def clean_column_name(col: str) -> str:
    """
    Convert Turkish column names into cleaner English names.
    """
    col = str(col).strip()

    mapping = {
        "Tarih": "date",
        "Kapanış(TL)": "close",
        "Kapanis(TL)": "close",
        "Min(TL)": "min_price",
        "Max(TL)": "max_price",
        "AOF(TL)": "avg_price",
        "Hacim(TL)": "volume",
        "Sermaye(m": "capital",
        "Sermaye(m)": "capital",
        "USDTRY": "usdtry",
        "BIST 100": "bist100",
        "PiyasaDeğe": "market_cap",
        "PiyasaDeğer": "market_cap",
        "PiyasaDeğeri": "market_cap",
        "PiyasaDegeri": "market_cap",
        "HalkaAçık": "free_float",
        "HalkaAcik": "free_float",
        "HalkaAçık PD(m USD)": "free_float_market_cap_usd",
        "PD(m USD)": "market_cap_usd",
        "PD/DD": "pb_ratio",
    }

    if col in mapping:
        return mapping[col]

    # fallback cleaning for unexpected column names
    col = col.replace("ğ", "g").replace("Ğ", "G")
    col = col.replace("ü", "u").replace("Ü", "U")
    col = col.replace("ş", "s").replace("Ş", "S")
    col = col.replace("ı", "i").replace("İ", "I")
    col = col.replace("ö", "o").replace("Ö", "O")
    col = col.replace("ç", "c").replace("Ç", "C")

    col = col.lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = col.strip("_")

    return col


def clean_numeric_series(s: pd.Series) -> pd.Series:
    """
    Convert numeric-looking strings to numeric.
    """
    if s.dtype == "object":
        s = (
            s.astype(str)
            .str.replace(",", ".", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace("\xa0", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

    return pd.to_numeric(s, errors="coerce")


# -------------------------------------------------------
# Main ingestion
# -------------------------------------------------------

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_EXCEL_DIR.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No Excel files found in {RAW_EXCEL_DIR}")

    all_frames = []

    print(f"Found {len(files)} Excel files.")

    for file in files:
        ticker = extract_ticker(file.name)

        try:
            df = pd.read_excel(file)
        except Exception as e:
            print(f"FAILED reading {file.name}: {e}")
            continue

        # Clean column names
        df.columns = [clean_column_name(c) for c in df.columns]

        # Add ticker and source filename before combining
        df["ticker"] = ticker
        df["source_file"] = file.name

        # Parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

        # Convert non-date/non-id columns to numeric where possible
        id_cols = {"date", "ticker", "source_file"}

        for col in df.columns:
            if col not in id_cols:
                df[col] = clean_numeric_series(df[col])

        all_frames.append(df)

        print(f"Loaded {file.name}: {len(df)} rows, ticker={ticker}")

    combined = pd.concat(all_frames, ignore_index=True)

    # Sort by ticker and date
    if "date" in combined.columns:
        combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Save combined raw dataset
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nDONE")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nColumns:")
    print(list(combined.columns))

    print("\nPreview:")
    print(combined.head())


if __name__ == "__main__":
    main()
