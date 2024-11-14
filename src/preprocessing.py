import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

# Define the input and output directories
input_dir = "data"
output_dir = "data_cleaned"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to preprocess a single file
def preprocess_file(file_path, ticker):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Remove the first two rows and rename columns
    data_cleaned = data.iloc[2:].copy()
    data_cleaned.columns = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    
    # Convert 'Date' column to datetime and numeric columns to floats
    data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')
    numeric_columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_columns:
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
    
    # Drop rows with missing data
    data_cleaned.dropna(inplace=True)
    
    # Generate 20-day returns
    data_cleaned['20-Day Returns'] = data_cleaned['Adj Close'].pct_change(periods=20)
    
    # Generate rolling 20-day volatility (standard deviation of daily returns over 20 days)
    data_cleaned['Daily Returns'] = data_cleaned['Adj Close'].pct_change()
    data_cleaned['20-Day Volatility'] = data_cleaned['Daily Returns'].rolling(window=20).std()
    
    # Normalize 20-day returns and 20-day volatility
    data_cleaned['Normalized 20-Day Returns'] = (
        (data_cleaned['20-Day Returns'] - data_cleaned['20-Day Returns'].mean()) / 
        data_cleaned['20-Day Returns'].std()
    )
    data_cleaned['Normalized 20-Day Volatility'] = (
        (data_cleaned['20-Day Volatility'] - data_cleaned['20-Day Volatility'].mean()) / 
        data_cleaned['20-Day Volatility'].std()
    )
    
    # PCA-specific preprocessing: handle missing values and standardize features
    pca_features = ['20-Day Returns', '20-Day Volatility', 'Normalized 20-Day Returns', 'Normalized 20-Day Volatility']
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    if all(feature in data_cleaned.columns for feature in pca_features):
        # Impute missing values and standardize features
        data_cleaned[pca_features] = imputer.fit_transform(data_cleaned[pca_features])
        data_cleaned[pca_features] = scaler.fit_transform(data_cleaned[pca_features])
    
    # Drop intermediate 'Daily Returns' column
    data_cleaned.drop(columns=['Daily Returns'], inplace=True)
    
    # Save the cleaned data
    output_path = os.path.join(output_dir, f"{ticker}_cleaned_data.csv")
    data_cleaned.to_csv(output_path, index=False)
    print(f"Processed and saved: {ticker}")

# Loop through all tickers and preprocess each file
for ticker in nasdaq100_tickers:
    input_file = os.path.join(input_dir, f"{ticker}_historical_data.csv")
    if os.path.exists(input_file):
        preprocess_file(input_file, ticker)
    else:
        print(f"File not found for {ticker}: {input_file}")