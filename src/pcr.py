import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

input_dir = "data_cleaned"
output_dir = "preds"
output_file = os.path.join(output_dir, "pcr_validation_results_with_returns.csv")

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

def calculate_normalized_mse(mse, prices):
    mean_price = prices.mean()
    
    # Handle division by 0 edge case
    if mean_price == 0:         
        return np.nan  
    return mse / (mean_price**2)

def pcr_forecast(train_data, test_data, features):
    """
    Perform Principal Components Regression (PCR) for forecasting.
    
    Parameters:
        train_data (pd.DataFrame): Training data containing features and target.
        test_data (pd.DataFrame): Testing data containing features.
        features (list): List of feature column names.
    
    Returns:
        np.array: Forecasted values for the test set.
    """
    pca = PCA()
    model = LinearRegression()

    # Extract training features and target vector
    train_features = train_data[features].values
    train_target = train_data["Adj Close"].values

    # Fit PCA
    reduced_features = pca.fit_transform(train_features)

    # Fit regression model
    model.fit(reduced_features, train_target)

    # Transform test features and predict
    test_features = test_data[features].values
    reduced_test_features = pca.transform(test_features)
    return model.predict(reduced_test_features)

def sliding_window_validation_pcr_with_returns(data, features, window=22, horizon=22, step=22):
    """
    Perform sliding window validation for PCR, compute MSE and predicted returns.
    
    Parameters:
        data (pd.DataFrame): Data containing features and target.
        features (list): List of feature column names.
        window (int): Lookback period for training.
        horizon (int): Number of days to forecast.
        step (int): Step size for sliding the validation window.
    
    Returns:
        list: Mean Squared Errors (MSE) for each validation window.
        pd.DataFrame: DataFrame containing predicted prices and returns.
    """
    n = len(data)
    mse_list = []
    predicted_returns = []

    for start in range(0, n - window - horizon + 1, step):
        # Train/test split
        train_data = data[start : start + window]
        test_data = data[start + window : start + window + horizon]

        # Make predictions
        predictions = pcr_forecast(train_data, test_data, features)
        mse = mean_squared_error(test_data["Adj Close"], predictions)
        mse_list.append(mse)

        # Calculate predicted returns
        pred_prices = pd.DataFrame({
            "Date": test_data["Date"].values,
            "Predicted_Adj_Close": predictions
        })
        pred_prices["Predicted_Returns"] = pred_prices["Predicted_Adj_Close"].pct_change()
        predicted_returns.append(pred_prices)

    # Combine all predicted returns into one DataFrame
    predicted_returns_df = pd.concat(predicted_returns, ignore_index=True)
    return mse_list, predicted_returns_df

# Validation results
validation_results = []
normalized_mses = []
all_predicted_returns = []

# Process each stock
os.makedirs(output_dir, exist_ok=True)
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
    
    df = pd.read_csv(input_file)
    
    if "Adj Close" not in df.columns:
        print(f"'Adj Close' column missing in {ticker}")
        continue

    # Features for PCR
    features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]

    # Perform sliding window validation
    mse_values, predicted_returns = sliding_window_validation_pcr_with_returns(df, features)
    avg_mse = np.mean(mse_values)
    normalized_mse = calculate_normalized_mse(avg_mse, df["Adj Close"])
    
    if not np.isnan(normalized_mse):
        normalized_mses.append(normalized_mse)

    # Append results
    validation_results.append({
        "Ticker": ticker,
        "Average_MSE": avg_mse,
        "Normalized_MSE": normalized_mse,
        "MSE_List": mse_values
    })
    
    # Save predicted returns
    predicted_returns["Ticker"] = ticker
    all_predicted_returns.append(predicted_returns)

# Combine all predicted returns into one DataFrame and save
predicted_returns_df = pd.concat(all_predicted_returns, ignore_index=True)
predicted_returns_file = os.path.join(output_dir, "pcr_predicted_returns.csv")
predicted_returns_df.to_csv(predicted_returns_file, index=False)
print(f"Predicted returns saved to {predicted_returns_file}")

# Save validation results
validation_df = pd.DataFrame(validation_results)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Overall normalized MSE of PCR
overall_normalized_mse = np.mean(normalized_mses)
print(f"Overall Normalized MSE for PCR: {overall_normalized_mse}")