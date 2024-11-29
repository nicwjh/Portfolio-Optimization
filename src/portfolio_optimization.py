import numpy as np
import pandas as pd
from scipy.optimize import minimize
import gc

# Define the risk-free rate
rF = 0.0443  # Current risk-free rate (4.43% - 1-year US T-bill)

methods = ["pcr", "rf", "wma", "gru"]

predicted_returns_dir = "preds"
covariance_matrix_file = f"{predicted_returns_dir}/covariance_matrix.csv"

avg_pred_returns = []

for method in methods:
    # Load predicted returns
    preds = pd.read_csv(
        f"{predicted_returns_dir}/{method}_predicted_returns.csv",
        usecols=["Date", "Ticker", "Simple_Returns"],
        dtype={"Date": str, "Ticker": str, "Simple_Returns": np.float32}
    )
    preds.rename(columns={"Simple_Returns": f"Predicted_Returns_{method.upper()}"}, inplace=True)

    # Convert 'Date' to datetime format
    preds['Date'] = pd.to_datetime(preds['Date'], errors='coerce')

    # Normalize dates to remove time component and timezone
    preds['Date'] = preds['Date'].dt.tz_localize(None).dt.normalize()

    # Drop rows with invalid dates
    preds.dropna(subset=['Date'], inplace=True)

    # Replace infinite values and drop NaNs
    preds.replace([np.inf, -np.inf], np.nan, inplace=True)
    preds.dropna(subset=[f'Predicted_Returns_{method.upper()}'], inplace=True)

    # Calculate average predicted return per ticker
    avg_preds = preds.groupby('Ticker')[f'Predicted_Returns_{method.upper()}'].mean().reset_index()
    avg_pred_returns.append(avg_preds)

# Merge average predicted returns per ticker from all methods
from functools import reduce
combined_avg_returns = reduce(lambda left, right: pd.merge(left, right, on='Ticker', how='inner'), avg_pred_returns)

# Check if combined_avg_returns is empty
if combined_avg_returns.empty:
    raise ValueError("No overlapping tickers found among methods after merging.")

# Calculate Expected Return per Ticker as mean across methods
predicted_return_cols = [f'Predicted_Returns_{method.upper()}' for method in methods]
combined_avg_returns['Expected_Return'] = combined_avg_returns[predicted_return_cols].mean(axis=1)

# Prepare grouped_data for optimization
grouped_data = combined_avg_returns[['Ticker', 'Expected_Return']]

### Load and Filter Covariance Matrix ###
# Load covariance matrix without specifying dtype
cov_matrix_df = pd.read_csv(
    covariance_matrix_file,
    index_col=0
)

# Convert the data to float
cov_matrix_df = cov_matrix_df.astype(np.float32)

cov_matrix_df.index = cov_matrix_df.index.astype(str)
cov_matrix_df.columns = cov_matrix_df.columns.astype(str)

# Get unique tickers 
unique_tickers = grouped_data["Ticker"].values

# Ensure the covariance matrix includes all the unique tickers
missing_tickers = set(unique_tickers) - set(cov_matrix_df.index)
if missing_tickers:
    raise ValueError(f"Tickers {missing_tickers} are missing in the covariance matrix.")

# Filter the covariance matrix to include only the tickers in unique_tickers
cov_matrix_df = cov_matrix_df.loc[unique_tickers, unique_tickers]
cov_matrix = cov_matrix_df.values

# Ensure covariance matrix matches the number of assets
n_assets = len(unique_tickers)
if cov_matrix.shape != (n_assets, n_assets):
    raise ValueError("Covariance matrix size does not match the number of unique tickers.")
