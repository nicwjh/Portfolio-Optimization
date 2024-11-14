import os
import pandas as pd
import numpy as np

# Directory containing the cleaned CSV files
input_dir = "data_cleaned"
output_file = "preds/wma_preds.csv"

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
    # Create weights for WMA (higher weights for more recent observations)
    weights = np.arange(1, window + 1)  # [1, 2, ..., window]
    
    # Initialize the forecasted sequence
    forecast = []
    
    # Start with the historical data
    rolling_data = data[-window:].tolist()  # Get the last `window` values
    
    for _ in range(horizon):
        # Calculate WMA for the current rolling window
        wma = np.dot(rolling_data, weights) / weights.sum()
        
        # Append the forecasted value
        forecast.append(wma)
        
        # Update the rolling window with the new forecast
        rolling_data.pop(0)  # Remove the oldest value
        rolling_data.append(wma)  # Add the new forecasted value
    
    return forecast

# List to store all predictions
all_predictions = []

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
    
    # Calculate dynamic WMA predictions
    historical_prices = df['Adj Close']
    dynamic_predictions = dynamic_wma_forecast(historical_prices, window=20, horizon=22)
    
    # Append predictions to the list
    all_predictions.append({
        "Ticker": ticker,
        **{f"Day_{i+1}": dynamic_predictions[i] for i in range(22)}
    })

# Create a DataFrame from all predictions
predictions_df = pd.DataFrame(all_predictions)

# Save the consolidated predictions to a single CSV file
predictions_df.to_csv(output_file, index=False)
print(f"All Dynamic WMA predictions saved to {output_file}")