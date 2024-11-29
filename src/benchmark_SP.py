import yfinance as yf
import numpy as np
import pandas as pd

START_DATE = "2023-11-01"
END_DATE = "2024-11-01"
RISK_FREE_RATE = 0.0443  # Annualized risk-free rate (4.43%)

def calculate_sp500_benchmark(start_date, end_date, risk_free_rate):
    """
    Fetches S&P 500 data, calculates returns, volatility, and Sharpe ratio over a specified period.

    Args:
        start_date (str): Start date of the period (YYYY-MM-DD).
        end_date (str): End date of the period (YYYY-MM-DD).
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        dict: A dictionary containing S&P 500 metrics (cumulative return, volatility, Sharpe ratio).
    """
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(start=start_date, end=end_date)

    if data.empty:
        raise ValueError("No data retrieved for S&P 500. Check the date range or ticker symbol.")

    # Calculate daily returns
    data["Daily_Returns"] = data["Close"].pct_change()

    data.dropna(subset=["Daily_Returns"], inplace=True)

    # Calculate cumulative return
    cumulative_return = (1 + data["Daily_Returns"]).prod() - 1

    # Calculate annualized volatility
    daily_volatility = data["Daily_Returns"].std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days in a year

    # Calculate annualized Sharpe ratio
    daily_risk_free_rate = risk_free_rate / 252
    excess_daily_returns = data["Daily_Returns"] - daily_risk_free_rate
    annualized_sharpe_ratio = (excess_daily_returns.mean() / excess_daily_returns.std()) * np.sqrt(252)

    # Return results as a dictionary
    metrics = {
        "Cumulative_Return": cumulative_return,
        "Annualized_Volatility": annualized_volatility,
        "Annualized_Sharpe_Ratio": annualized_sharpe_ratio,
    }

    return metrics

if __name__ == "__main__":
    try:
        sp500_metrics = calculate_sp500_benchmark(START_DATE, END_DATE, RISK_FREE_RATE)
        print("S&P 500 Benchmark Metrics:")
        print(f"Cumulative Return: {sp500_metrics['Cumulative_Return']:.2%}")
        print(f"Annualized Volatility: {sp500_metrics['Annualized_Volatility']:.2%}")
        print(f"Annualized Sharpe Ratio: {sp500_metrics['Annualized_Sharpe_Ratio']:.2f}")
    except Exception as e:
        print(f"Error: {e}")