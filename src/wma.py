import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

input_dir = "data_cleaned"
output_file = "preds/wma_validation_results.csv"
returns_file = "preds/wma_predicted_returns.csv"

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

def dynamic_wma_forecast(data, window=22, horizon=22):
    """
    Predicts a sequence of future values using a rolling Weighted Moving Average.
    """
    weights = np.arange(1, window + 1)
    forecast = []
    rolling_data = data[-window:].tolist()

    for _ in range(horizon):
        wma = np.dot(rolling_data, weights) / weights.sum()
        forecast.append(wma)
        rolling_data.pop(0)
        rolling_data.append(wma)
    
    return forecast

def sliding_window_validation(data, window=22, horizon=22, step=22):
    """
    Performs sliding window validation for WMA and computes MSE and predictions.
    """
    n = len(data)
    mse_list = []
    predictions_list = []

    for start in range(0, n - window - horizon + 1, step):
        train_end = start + window
        test_end = train_end + horizon

        # Train/test split
        train_data = data[start:train_end]
        test_data = data[train_end:test_end]

        # Forecast using WMA
        predictions = dynamic_wma_forecast(train_data, window=window, horizon=horizon)
        predictions_list.append((predictions, test_data.index[:len(predictions)]))
        
        # Calculate MSE for the window
        mse = mean_squared_error(test_data, predictions)
        mse_list.append(mse)
    
    return mse_list, predictions_list

def calculate_predicted_returns(predictions, test_dates):
    """
    Calculate simple returns based on predicted values.
    """
    # Convert predictions to a NumPy array for calculations
    predictions = np.array(predictions)
    
    # Initialize simple_returns array
    simple_returns = np.zeros(len(predictions))
    
    # Calculate simple returns
    simple_returns[1:] = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    
    return pd.DataFrame({
        "Date": test_dates,
        "Predicted_Adj_Close": predictions,
        "Simple_Returns": simple_returns
    })

def calculate_normalized_mse(mse, prices):
    """
    Normalize MSE by the mean price of the stock.
    """
    mean_price = prices.mean()
    if mean_price == 0:
        return np.nan  # Avoid division by zero
    return mse / (mean_price**2)

validation_results = []
normalized_mses = []
predicted_returns_list = []

# Process each stock
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
 
    df = pd.read_csv(input_file, parse_dates=["Date"], index_col="Date")
    
    if 'Adj Close' not in df.columns:
        print(f"'Adj Close' column missing in {ticker}")
        continue
    
    # Perform sliding window validation
    historical_prices = df['Adj Close']
    mse_values, predictions_data = sliding_window_validation(historical_prices, window=22, horizon=22, step=22)
    avg_mse = np.mean(mse_values)
    
    # Calculate normalized MSE
    normalized_mse = calculate_normalized_mse(avg_mse, historical_prices)
    if not np.isnan(normalized_mse):
        normalized_mses.append(normalized_mse)
    
    validation_results.append({
        "Ticker": ticker,
        "Average_MSE": avg_mse,
        "Normalized_MSE": normalized_mse,
        "MSE_List": mse_values  
    })

    # Calculate predicted returns
    for predictions, test_dates in predictions_data:
        predicted_returns = calculate_predicted_returns(predictions, test_dates)
        predicted_returns["Ticker"] = ticker
        predicted_returns_list.append(predicted_returns)

validation_df = pd.DataFrame(validation_results)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Combine and save predicted returns
all_returns_df = pd.concat(predicted_returns_list, ignore_index=True)
returns_file = os.path.join("preds", "wma_predicted_returns.csv")
all_returns_df.to_csv(returns_file, index=False)
print(f"Predicted returns saved to {returns_file}")

# Compute overall normalized MSE
overall_normalized_mse = np.mean(normalized_mses)
print(f"Overall Normalized MSE for WMA: {overall_normalized_mse}")