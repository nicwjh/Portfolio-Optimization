import os
import pandas as pd
import numpy as np

# Input directory where cleaned data is stored
input_dir = "data_cleaned"
output_file = "preds/covariance_matrix.csv"

# List of tickers
nasdaq100_tickers = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "COST",
    "NFLX", "TMUS", "ASML", "CSCO", "ADBE", "AMD", "PEP", "LIN", "INTU", "AZN",
    "ISRG", "TXN", "QCOM", "CMCSA", "BKNG", "AMGN", "PDD", "AMAT", "HON", "ARM",
    "PANW", "VRTX", "ADP", "GILD", "SBUX", "MU", "INTC", "ADI", "MELI", "LRCX",
    "CTAS", "MDLZ", "PYPL", "REGN", "KLAC", "SNPS", "CRWD", "CDNS", "ABNB", "MAR",
    "MRVL", "FTNT", "DASH", "WDAY", "ORLY", "CEG", "CSX", "ADSK", "TEAM", "CHTR",
    "TTD", "ROP", "PCAR", "NXPI", "CPRT", "MNST", "FANG", "PAYX", "AEP", "ODFL",
    "FAST", "ROST", "KDP", "DDOG", "BKR", "EA", "VRSK", "CTSH", "LULU", "XEL",
    "KHC", "GEHC", "EXC", "MCHP", "CCEP", "IDXX", "ZS", "TTWO", "CSGP", "ANSS",
    "ON", "DXCM", "CDW", "BIIB", "WBD", "GFS", "ILMN", "MDB", "MRNA", "DLTR", "WBA"
]

# DataFrame to store returns
returns_df = pd.DataFrame()

# Process each stock
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
    
    df = pd.read_csv(input_file, parse_dates=["Date"], index_col="Date")
    
    if "Adj Close" not in df.columns:
        print(f"'Adj Close' column missing in {ticker}")
        continue

    # Calculate simple returns
    df[f"{ticker}_Returns"] = df["Adj Close"].pct_change()

    # Add to returns DataFrame
    returns_df[ticker] = df[f"{ticker}_Returns"]

# Drop rows with NaN values (from pct_change)
returns_df.dropna(inplace=True)

# Calculate covariance matrix
cov_matrix = returns_df.cov()

# Save covariance matrix to CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
cov_matrix.to_csv(output_file)
print(f"Covariance matrix saved to {output_file}")