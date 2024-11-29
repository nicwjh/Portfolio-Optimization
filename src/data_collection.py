import yfinance as yf
import pandas as pd

def fetch_nasdaq100_data(start_date, end_date, interval="1d"):
    """
    Fetch historical data for NASDAQ 100 companies from Yahoo Finance.

    Parameters:
    - start_date: Start date for historical data (YYYY-MM-DD).
    - end_date: End date for historical data (YYYY-MM-DD).
    - interval: Data interval ("1d", "1wk", "1mo").

    Returns:
    - A dictionary of DataFrames, one for each stock.
    """
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


    stock_data = {}
    
    for ticker in nasdaq100_tickers:
        print(f"Fetching data for {ticker}...")
        try:
            data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
            stock_data[ticker] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    return stock_data

start_date = "2023-11-01"
end_date = "2024-11-01"

nasdaq100_data = fetch_nasdaq100_data(start_date, end_date)

for ticker, data in nasdaq100_data.items():
    if not data.empty:
        data.to_csv(f"data/{ticker}_historical_data.csv")
        print(f"Saved {ticker} data to CSV.")

print("Data collection completed.")