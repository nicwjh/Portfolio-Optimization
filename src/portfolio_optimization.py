import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functools import reduce

START_DATE = "2023-11-01"
END_DATE = "2024-11-01"
RISK_FREE_RATE = 0.0443  # Annualized risk-free rate (4.43%)
ANNUALIZATION_FACTOR = np.sqrt(252)  
TARGET_RETURN = 0.10  

methods = ["pcr", "rf", "wma", "gru"]

predicted_returns_dir = "preds"
covariance_matrix_file = f"{predicted_returns_dir}/covariance_matrix.csv"

avg_pred_returns = []

for method in methods:
    
    preds = pd.read_csv(
        f"{predicted_returns_dir}/{method}_predicted_returns.csv",
        usecols=["Date", "Ticker", "Simple_Returns"],
        dtype={"Date": str, "Ticker": str, "Simple_Returns": np.float32}
    )
    preds.rename(columns={"Simple_Returns": f"Predicted_Returns_{method.upper()}"}, inplace=True)

    # Convert 'Date' to datetime format
    preds['Date'] = pd.to_datetime(preds['Date'], errors='coerce')

    # Normalize dates to remove time component and timezone
    preds['Date'] = preds['Date'].dt.tz_localize(None).dt.normalize()

    # Drop rows with invalid dates
    preds.dropna(subset=['Date'], inplace=True)

    # Replace infinite values and drop NaNs
    preds.replace([np.inf, -np.inf], np.nan, inplace=True)
    preds.dropna(subset=[f'Predicted_Returns_{method.upper()}'], inplace=True)

    # Calculate average predicted return per ticker
    avg_preds = preds.groupby('Ticker')[f'Predicted_Returns_{method.upper()}'].mean().reset_index()
    avg_pred_returns.append(avg_preds)

# Merge average predicted returns per ticker from all methods
combined_avg_returns = reduce(lambda left, right: pd.merge(left, right, on='Ticker', how='inner'), avg_pred_returns)

if combined_avg_returns.empty:
    raise ValueError("No overlapping tickers found among methods after merging.")

# Calculate Expected Return per Ticker as mean across methods
predicted_return_cols = [f'Predicted_Returns_{method.upper()}' for method in methods]
combined_avg_returns['Expected_Return'] = combined_avg_returns[predicted_return_cols].mean(axis=1)

# Prepare grouped_data for optimization
grouped_data = combined_avg_returns[['Ticker', 'Expected_Return']]

### Load and Filter Covariance Matrix ###
cov_matrix_df = pd.read_csv(covariance_matrix_file, index_col=0).astype(np.float32)
cov_matrix_df.index = cov_matrix_df.index.astype(str)
cov_matrix_df.columns = cov_matrix_df.columns.astype(str)

unique_tickers = grouped_data["Ticker"].values

missing_tickers = set(unique_tickers) - set(cov_matrix_df.index)
if missing_tickers:
    raise ValueError(f"Tickers {missing_tickers} are missing in the covariance matrix.")

cov_matrix_df = cov_matrix_df.loc[unique_tickers, unique_tickers]
cov_matrix = cov_matrix_df.values

n_assets = len(unique_tickers)
if cov_matrix.shape != (n_assets, n_assets):
    raise ValueError("Covariance matrix size does not match the number of unique tickers.")
  

def sparse_portfolio_optimization(expected_returns, cov_matrix, l1_lambda, target_return):
    """
    Perform sparse portfolio optimization using L1 regularization with a target return constraint.

    Args:
        expected_returns (np.ndarray): Array of expected returns.
        cov_matrix (np.ndarray): Covariance matrix.
        l1_lambda (float): L1 regularization strength.
        target_return (float): Target return for the portfolio.

    Returns:
        np.ndarray: Optimal sparse portfolio weights.
    """
    n_assets = len(expected_returns)

    def objective(weights):
        # Portfolio variance + L1 regularization penalty
        variance = weights @ (cov_matrix @ weights)
        penalty = l1_lambda * np.sum(np.abs(weights))
        return variance + penalty

    # Constraints: weights sum to 1 and achieve the target return
    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1},  # Fully invested
        {"type": "eq", "fun": lambda weights: weights @ expected_returns - target_return}  # Target return
    ]

    # Bounds for weights (long-only portfolio)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess (equal distribution)
    initial_weights = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    return result.x

lambdas = [0.01, 0.05, 0.5, 1, 5, 10, 25, 50, 100]  
best_sparse_weights = None
best_sparse_metrics = None
best_lambda = None
highest_sharpe_ratio = float("-inf")

for l1_lambda in lambdas:
    try:
        sparse_weights = sparse_portfolio_optimization(
            grouped_data["Expected_Return"].values,
            cov_matrix,
            l1_lambda,
            target_return=TARGET_RETURN
        )
        
        sparse_portfolio_return = sparse_weights @ grouped_data["Expected_Return"].values
        sparse_portfolio_volatility = np.sqrt(sparse_weights @ (cov_matrix @ sparse_weights)) * ANNUALIZATION_FACTOR
        sparse_portfolio_sharpe_ratio = (sparse_cumulative_return - RISK_FREE_RATE) / sparse_portfolio_volatility


        sparse_cumulative_return = (1 + sparse_portfolio_return) ** (252 / len(grouped_data)) - 1
        
        if sparse_portfolio_sharpe_ratio > highest_sharpe_ratio:
            best_sparse_weights = sparse_weights
            best_sparse_metrics = {
                "Cumulative_Return": sparse_cumulative_return,
                "Portfolio_Return": sparse_portfolio_return,
                "Portfolio_Volatility": sparse_portfolio_volatility,
                "Portfolio_Sharpe_Ratio": sparse_portfolio_sharpe_ratio,
                "Portfolio_Alpha": sparse_portfolio_return - RISK_FREE_RATE,
            }
            best_lambda = l1_lambda
            highest_sharpe_ratio = sparse_portfolio_sharpe_ratio
    except ValueError as e:
        print(f"Optimization failed for λ={l1_lambda}: {str(e)}")

grouped_data["Sparse_Weights"] = best_sparse_weights
grouped_data["Sparse_Sharpe_Ratio"] = (grouped_data["Expected_Return"] - RISK_FREE_RATE) / grouped_data["Volatility"]

grouped_data.to_csv("optimization_outputs/sparse_mean_variance_optimized_portfolio.csv", index=False)
pd.DataFrame([best_sparse_metrics]).to_csv("optimization_outputs/sparse_mean_variance_portfolio_metrics.csv", index=False)

print(f"Best λ: {best_lambda}")
print("Sparse Portfolio Weights and Metrics:")
print(grouped_data[["Ticker", "Sparse_Weights", "Expected_Return", "Volatility", "Sparse_Sharpe_Ratio"]])
print("\nSparse Portfolio Metrics:")
print(best_sparse_metrics)
print("\nResults saved to 'sparse_mean_variance_optimized_portfolio.csv' and 'sparse_mean_variance_portfolio_metrics.csv'")