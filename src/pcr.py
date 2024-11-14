import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer

# Directories for input and output
input_dir = "data_cleaned"
output_file = "preds/pcr_preds.csv"

# List of NASDAQ 100 tickers
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

# Function to perform PCA and regression
def perform_pca_and_regression(features, target, max_components=10):
    imputer = SimpleImputer(strategy='mean')
    features = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    if features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError("Insufficient data for PCA")
    max_components = min(max_components, min(features.shape[0], features.shape[1]))
    features = (features - features.mean()) / features.std()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for n in range(1, max_components + 1):
        pca = PCA(n_components=n)
        reduced_features = pca.fit_transform(features)
        model = Ridge(alpha=1.0)
        score = cross_val_score(model, reduced_features, target, cv=kf, scoring='neg_mean_squared_error').mean()
        scores.append(score)
    optimal_pcs = np.argmax(scores) + 1
    print(f"Optimal number of components: {optimal_pcs}")
    pca = PCA(n_components=optimal_pcs)
    reduced_features = pca.fit_transform(features)
    model = Ridge(alpha=1.0)
    model.fit(reduced_features, target)
    return model, pca, optimal_pcs

# Process each stock
all_predictions = []
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
    if not os.path.exists(input_file):
        print(f"File not found for {ticker}")
        continue
    df = pd.read_csv(input_file)
    required_columns = ['20-Day Returns', '20-Day Volatility', 'Normalized 20-Day Returns', 'Normalized 20-Day Volatility']
    if not all(col in df.columns for col in required_columns):
        print(f"Required columns missing in {ticker}")
        continue
    features = df[required_columns]
    target = df['Adj Close']
    try:
        model, pca, optimal_pcs = perform_pca_and_regression(features, target)
    except ValueError as e:
        print(f"Skipping {ticker} due to insufficient data: {e}")
        continue
    last_features = features.iloc[-pca.n_components_:].values
    rolling_window = pca.transform(last_features)
    predictions = []
    for _ in range(22):
        # Predict the next day's price
        next_pred = model.predict(rolling_window.mean(axis=0).reshape(1, -1))[0]
        next_pred = max(min(next_pred, target.max()), target.min())  # Clamp predictions
        predictions.append(next_pred)

        # Use the last actual feature row as a base
        new_features = last_features[-1].reshape(1, -1)  # Use the last row of actual features
        new_features[:, 0] = next_pred  # Replace only the first column (e.g., price)

        # Standardize the new features
        standardized_features = (new_features - features.mean().values) / features.std().values
        transformed_features = pca.transform(standardized_features)  # Transform into PCA space

        # Update the rolling window dynamically
        rolling_window = np.roll(rolling_window, -1, axis=0)  # Shift the rolling window
        rolling_window[-1, :] = 0.9 * rolling_window[-1, :] + 0.1 * transformed_features  # Weighted update
    all_predictions.append({"Ticker": ticker, **{f"Day_{i+1}": predictions[i] for i in range(22)}})

# Save the predictions
predictions_df = pd.DataFrame(all_predictions)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
predictions_df.to_csv(output_file, index=False)
print(f"All PCR predictions saved to {output_file}")