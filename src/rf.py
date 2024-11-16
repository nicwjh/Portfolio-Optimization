import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Input and output directories
input_dir = "data_cleaned"
output_file = "preds/rf_validation_results.csv"
returns_file = "preds/rf_predicted_returns.csv"

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

def rf_forecast(train_data, test_data, features):
    """
    Performs Random Forest Regression (RF) for forecasting.
    """
    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    # Train the model
    train_features = train_data[features].values
    train_target = train_data["Adj Close"].values
    model.fit(train_features, train_target)

    # Predict for the test set
    test_features = test_data[features].values
    return model.predict(test_features)

def calculate_predicted_returns(predictions, test_data):
    """
    Calculate simple returns based on predicted values.
    
    Parameters:
        predictions (np.array): Predicted prices for the test set.
        test_data (pd.DataFrame): Test dataset with corresponding dates.
    
    Returns:
        pd.DataFrame: DataFrame containing predicted prices and simple returns.
    """
    if len(predictions) < 2:
        raise ValueError("Predictions array must have at least two elements to calculate returns.")

    # Ensure matching lengths for dates
    valid_dates = test_data.index[:len(predictions)]

    # Simple Returns: (P_t - P_t-1) / P_t-1
    simple_returns = np.zeros(len(predictions))
    simple_returns[1:] = (predictions[1:] - predictions[:-1]) / predictions[:-1]

    return pd.DataFrame({
        "Date": valid_dates,
        "Predicted_Adj_Close": predictions,
        "Simple_Returns": simple_returns
    })


def sliding_window_validation_rf(data, features, window=15, horizon=22, step=22):
    """
    Performs sliding window validation for Random Forest.
    """
    mse_list = []
    predictions_list = []
    n = len(data)

    for start in range(0, n - window - horizon + 1, step):
        train_data = data[start : start + window]
        test_data = data[start + window : start + window + horizon]

        # Forecast using Random Forest
        predictions = rf_forecast(train_data, test_data, features)
        mse = mean_squared_error(test_data["Adj Close"], predictions)
        mse_list.append(mse)
        predictions_list.append(predictions)

    return mse_list, predictions_list

def calculate_normalized_mse(mse, prices):
    """
    Calculate normalized MSE.
    """
    mean_price = prices.mean()
    if mean_price == 0:
        return np.nan  # Avoid division by zero
    return mse / (mean_price**2)

validation_results = []
predicted_returns_list = []

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

    # Features for Random Forest
    features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]

    # Perform sliding window validation
    mse_values, predictions_list = sliding_window_validation_rf(df, features, window=15, horizon=22, step=22)
    avg_mse = np.mean(mse_values)
    normalized_mse = calculate_normalized_mse(avg_mse, df["Adj Close"])
    
    # Save validation results
    validation_results.append({
        "Ticker": ticker,
        "Average_MSE": avg_mse,
        "Normalized_MSE": normalized_mse,
        "MSE_List": mse_values
    })

    # Calculate predicted returns (Simple Returns)
    for predictions in predictions_list:
        predicted_returns = calculate_predicted_returns(predictions, df)
        predicted_returns["Ticker"] = ticker
        predicted_returns_list.append(predicted_returns)

validation_df = pd.DataFrame(validation_results)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Combine and save predicted returns
all_returns_df = pd.concat(predicted_returns_list, ignore_index=True)
all_returns_df.to_csv(returns_file, index=False)
print(f"Predicted returns saved to {returns_file}")