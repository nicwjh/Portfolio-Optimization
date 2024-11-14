import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Directory containing the preprocessed cleaned CSV files
input_dir = "data_cleaned"
output_file = "preds/rf_optimized_validation_results.csv"

# List of tickers
nasdaq100_tickers = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "AVGO", "COST",
    # (truncated for brevity, include the full list here)
]

# Optimized Random Forest implementation
def rf_forecast(train_data, test_data, features):
    """
    Perform Random Forest Regression (RF) for forecasting.
    
    Parameters:
        train_data (pd.DataFrame): Training data containing features and target.
        test_data (pd.DataFrame): Testing data containing features.
        features (list): List of feature column names.
    
    Returns:
        np.array: Forecasted values for the test set.
    """
    model = RandomForestRegressor(
        n_estimators=20,  # Reduced number of trees
        max_depth=5,  # Limit tree depth
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Extract training features and target
    train_features = train_data[features].values
    train_target = train_data["Adj Close"].values

    # Fit the Random Forest model
    model.fit(train_features, train_target)

    # Predict for the test set
    test_features = test_data[features].values
    return model.predict(test_features)

# Sliding window validation for Random Forest
def sliding_window_validation_rf(data, features, window=15, horizon=22, step=22):
    """
    Perform sliding window validation for Random Forest and compute MSE.
    
    Parameters:
        data (pd.DataFrame): Data containing features and target.
        features (list): List of feature column names.
        window (int): Lookback period for training.
        horizon (int): Number of days to forecast.
        step (int): Step size for sliding the validation window.
    
    Returns:
        list: Mean Squared Errors (MSE) for each validation window.
    """
    n = len(data)
    mse_list = []

    for start in range(0, n - window - horizon + 1, step):
        # Training and testing splits
        train_data = data[start : start + window]
        test_data = data[start + window : start + window + horizon]

        # Forecast using Random Forest
        predictions = rf_forecast(train_data, test_data, features)
        mse = mean_squared_error(test_data["Adj Close"], predictions)
        mse_list.append(mse)

    return mse_list

# Function to calculate normalized MSE
def calculate_normalized_mse(mse, prices):
    mean_price = prices.mean()
    if mean_price == 0:
        return np.nan  # Avoid division by zero
    return mse / (mean_price**2)

# Validation results
validation_results = []
normalized_mses = []

# Process each stock
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
    
    # Load cleaned data
    df = pd.read_csv(input_file)
    
    if "Adj Close" not in df.columns:
        print(f"'Adj Close' column missing in {ticker}")
        continue

    # Features for Random Forest
    features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]

    # Perform sliding window validation
    mse_values = sliding_window_validation_rf(df, features, window=15, horizon=22, step=22)
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

# Create a DataFrame from validation results
validation_df = pd.DataFrame(validation_results)

# Save results to a CSV file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Overall normalized MSE
overall_normalized_mse = np.mean(normalized_mses)
print(f"Overall Normalized MSE for Random Forest: {overall_normalized_mse}")