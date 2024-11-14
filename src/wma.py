import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# Directory containing the cleaned CSV files
input_dir = "data_cleaned"
output_file = "preds/wma_validation_results.csv"

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

# Function to calculate Dynamic Weighted Moving Average (WMA)
def dynamic_wma_forecast(data, window=22, horizon=22):
    """
    Predicts a sequence of future values using a rolling Weighted Moving Average.
    
    Parameters:
        data (pd.Series): Historical adjusted close prices.
        window (int): Lookback period for the moving average.
        horizon (int): Number of days to forecast.
    
    Returns:
        list: Predicted values for the next `horizon` days.
    """
    weights = np.arange(1, window + 1)  # [1, 2, ..., window]
    forecast = []
    rolling_data = data[-window:].tolist()  # Get the last `window` values
    
    for _ in range(horizon):
        wma = np.dot(rolling_data, weights) / weights.sum()
        forecast.append(wma)
        rolling_data.pop(0)  # Remove the oldest value
        rolling_data.append(wma)  # Add the new forecasted value
    
    return forecast

# Sliding window validation for WMA
def sliding_window_validation(data, window=22, horizon=22, step=22):
    """
    Performs sliding window validation for WMA and computes MSE for each window.
    
    Parameters:
        data (pd.Series): Historical adjusted close prices.
        window (int): Lookback period for the moving average.
        horizon (int): Number of days to forecast.
        step (int): Step size for sliding the validation window.
    
    Returns:
        list: Mean Squared Error (MSE) for each validation window.
    """
    n = len(data)
    mse_list = []
    
    for start in range(0, n - window - horizon + 1, step):
        train_end = start + window
        test_end = train_end + horizon
        
        # Split data into training and testing sets
        train_data = data[start:train_end]
        test_data = data[train_end:test_end]
        
        # Forecast using WMA
        predictions = dynamic_wma_forecast(train_data, window=window, horizon=horizon)
        
        # Calculate MSE for the window
        mse = mean_squared_error(test_data, predictions)
        mse_list.append(mse)
    
    return mse_list

# Function to calculate normalized MSE
def calculate_normalized_mse(mse, prices):
    """
    Normalize MSE by the mean price of the stock.
    
    Parameters:
        mse (float): Mean Squared Error for the stock.
        prices (pd.Series): Historical adjusted close prices of the stock.
    
    Returns:
        float: Normalized MSE for the stock.
    """
    mean_price = prices.mean()
    if mean_price == 0:
        return np.nan  # Avoid division by zero
    return mse / (mean_price**2)

# List to store validation results
validation_results = []
normalized_mses = []

# Process each stock
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
    
    # Load the cleaned data
    df = pd.read_csv(input_file)
    
    # Ensure the data has 'Adj Close'
    if 'Adj Close' not in df.columns:
        print(f"'Adj Close' column missing in {ticker}")
        continue
    
    # Perform sliding window validation
    historical_prices = df['Adj Close']
    mse_values = sliding_window_validation(historical_prices, window=22, horizon=22, step=22)
    avg_mse = np.mean(mse_values)
    
    # Calculate normalized MSE
    normalized_mse = calculate_normalized_mse(avg_mse, historical_prices)
    if not np.isnan(normalized_mse):
        normalized_mses.append(normalized_mse)
    
    # Append validation results
    validation_results.append({
        "Ticker": ticker,
        "Average_MSE": avg_mse,
        "Normalized_MSE": normalized_mse,
        "MSE_List": mse_values  # Optional: Save all window MSEs for further analysis
    })

# Create a DataFrame from validation results
validation_df = pd.DataFrame(validation_results)

# Save the validation results to a single CSV file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Compute overall normalized MSE
overall_normalized_mse = np.mean(normalized_mses)
print(f"Overall Normalized MSE for WMA: {overall_normalized_mse}")