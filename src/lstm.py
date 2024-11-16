import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Directory containing the cleaned CSV files
input_dir = "data_cleaned"
output_file = "preds/lstm_tensorflow_validation_results.csv"

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

# Build the LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=input_shape, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Sliding window validation for LSTM
def sliding_window_validation_lstm(data, features, window=22, horizon=22, step=22):
    mse_list = []

    for start in range(0, len(data) - window - horizon + 1, step):
        train_data = data.iloc[start:start + window]
        test_data = data.iloc[start + window:start + window + horizon]

        # Prepare training data
        x_train = train_data[features].values
        y_train = train_data["Adj Close"].values
        x_train = np.expand_dims(x_train, axis=0)  # Add batch dimension

        # Prepare testing data
        x_test = test_data[features].values
        y_test = test_data["Adj Close"].values
        x_test = np.expand_dims(x_test, axis=0)  # Add batch dimension

        # Build and train model
        model = build_lstm_model(input_shape=(x_train.shape[1], len(features)))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(x_train, y_train, epochs=20, batch_size=16, verbose=0, callbacks=[early_stopping])

        # Predict
        predictions = model.predict(x_test).flatten()

        # Calculate MSE
        mse = mean_squared_error(y_test[:horizon], predictions[:horizon])
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

    # Features for LSTM
    features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]

    # Perform sliding window validation
    mse_values = sliding_window_validation_lstm(df, features, window=22, horizon=22, step=22)
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
print(f"Overall Normalized MSE for LSTM (TensorFlow): {overall_normalized_mse}")