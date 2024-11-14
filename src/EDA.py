import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing the cleaned CSV files
input_dir = "data_cleaned"

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

# Combine all cleaned files into a single DataFrame for analysis
def load_cleaned_data(input_dir, tickers):
    dataframes = []
    for ticker in tickers:
        file_path = os.path.join(input_dir, f"{ticker}_cleaned_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["Ticker"] = ticker  # Add ticker as a column
            dataframes.append(df)
        else:
            print(f"File not found: {ticker}")
    return pd.concat(dataframes, ignore_index=True)

# Load the cleaned data
cleaned_data = load_cleaned_data(input_dir, nasdaq100_tickers)

# Overview of the data
print("Data Overview:")
print(cleaned_data.head())
print("\nSummary Statistics:")
print(cleaned_data.describe())

# 1. Missing Values Analysis
print("\nMissing Values:")
print(cleaned_data.isnull().sum())

# 2. Distribution of Adjusted Close Prices
plt.figure(figsize=(10, 6))
sns.histplot(cleaned_data["Adj Close"], kde=True, bins=50, color="blue")
plt.title("Distribution of Adjusted Close Prices")
plt.xlabel("Adjusted Close Price")
plt.ylabel("Frequency")
plt.show()

# 3. Average Adjusted Close Price Per Ticker
average_prices = cleaned_data.groupby("Ticker")["Adj Close"].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
average_prices.plot(kind="bar", color="green")
plt.title("Average Adjusted Close Price Per Ticker")
plt.xlabel("Ticker")
plt.ylabel("Average Adjusted Close Price")
plt.xticks(rotation=90)
plt.show()

# 4. Correlation Matrix for Features
correlation_features = ["20-Day Returns", "20-Day Volatility", "Normalized 20-Day Returns", "Normalized 20-Day Volatility"]
correlation_data = cleaned_data[correlation_features].dropna()
correlation_matrix = correlation_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# 5. Time-Series Plot for Adjusted Close Price of Top 5 Stocks
top_5_tickers = average_prices.head(5).index
plt.figure(figsize=(12, 6))
for ticker in top_5_tickers:
    stock_data = cleaned_data[cleaned_data["Ticker"] == ticker]
    plt.plot(stock_data["Date"], stock_data["Adj Close"], label=ticker)
plt.title("Time-Series Plot of Adjusted Close Prices (Top 5 Stocks)")
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.legend()
plt.show()

# 6. Volatility Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x="Ticker", y="20-Day Volatility", data=cleaned_data)
plt.title("Boxplot of 20-Day Volatility by Ticker")
plt.xticks([])
plt.ylabel("20-Day Volatility")
plt.show()

# Save summary statistics to a CSV
summary_stats = cleaned_data.describe()
summary_stats.to_csv("eda_summary_statistics.csv")
print("Summary statistics saved to eda_summary_statistics.csv")