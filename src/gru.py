import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

input_dir = "data_cleaned"
output_file = "preds/gru_validation_results.csv"
returns_file = "preds/gru_predicted_returns.csv"

start_date = "2023-11-01"
end_date = "2024-11-01"

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

def build_gru_model(input_shape):
    """
    Builds and compiles a GRU model.
    """
    model = Sequential([
        Input(shape=input_shape),
        GRU(16, activation='relu', return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def calculate_predicted_returns(predictions, test_data):
    """
    Calculate simple returns based on predicted values.
    """
    if len(predictions) < 2:
        raise ValueError("Predictions array must have at least two elements to calculate returns.")

    valid_dates = test_data.index[:len(predictions)]

    simple_returns = np.zeros(len(predictions))
    simple_returns[1:] = (predictions[1:] - predictions[:-1]) / predictions[:-1]

    return pd.DataFrame({
        "Date": valid_dates,
        "Predicted_Adj_Close": predictions,
        "Simple_Returns": simple_returns
    })

def sliding_window_validation_gru(data, features, sequence_length=20, horizon=22, step=22):
    """
    Perform sliding window validation for GRU.
    """
    mse_list = []
    predictions_list = []
    n = len(data)

    for start in range(0, n - sequence_length - horizon + 1, step):
        train_data = data.iloc[start:start + sequence_length]
        test_data = data.iloc[start + sequence_length:start + sequence_length + horizon]

        x_train = []
        y_train = []
        for i in range(len(train_data) - 1):
            x_train.append(train_data[features].iloc[i:i + 1].values)
            y_train.append(train_data["Adj Close"].iloc[i + 1])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = []
        for i in range(len(test_data)):
            x_test.append(test_data[features].iloc[i:i + 1].values)

        x_test = np.array(x_test)

        # Skip empty windows
        if x_train.shape[0] == 0 or x_test.shape[0] == 0:
            continue

        # Build and train the GRU model
        model = build_gru_model(input_shape=(x_train.shape[1], x_train.shape[2]))
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        model.fit(
            x_train, y_train,
            epochs=1,  # Reduced epochs for memory optimization (computational limits, so epochs == 1)
            batch_size=min(16, x_train.shape[0]),  # Smaller batch size - computational limits
            verbose=0,
            callbacks=[early_stopping]
        )

        # Predict for test data
        predictions = model.predict(x_test).flatten()
        predictions_list.append(predictions)

        # Compute MSE
        mse = mean_squared_error(test_data["Adj Close"], predictions)
        mse_list.append(mse)

    return mse_list, predictions_list

def calculate_normalized_mse(mse, prices):
    """
    Calculate normalized MSE.
    """
    mean_price = prices.mean()
    if mean_price == 0:
        return np.nan
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

    # Filter data for the specified date range
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    
    if df.empty:
        print(f"No data available for {ticker} in the specified date range.")
        continue

    # Features for GRU
    features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]

    # Perform sliding window validation
    mse_values, predictions_list = sliding_window_validation_gru(df, features, sequence_length=20, horizon=22, step=22)
    avg_mse = np.mean(mse_values)
    normalized_mse = calculate_normalized_mse(avg_mse, df["Adj Close"])
    
    validation_results.append({
        "Ticker": ticker,
        "Average_MSE": avg_mse,
        "Normalized_MSE": normalized_mse,
        "MSE_List": mse_values
    })

    # Calculate predicted returns
    for predictions in predictions_list:
        predicted_returns = calculate_predicted_returns(predictions, df)
        predicted_returns["Ticker"] = ticker
        predicted_returns_list.append(predicted_returns)

# Export validation results
validation_df = pd.DataFrame(validation_results)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
validation_df.to_csv(output_file, index=False)
print(f"Validation results saved to {output_file}")

# Combine and export predicted returns
all_returns_df = pd.concat(predicted_returns_list, ignore_index=True)
all_returns_df.to_csv(returns_file, index=False)
print(f"Predicted returns saved to {returns_file}")