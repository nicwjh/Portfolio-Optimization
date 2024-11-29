import numpy as np
import pandas as pd
from scipy.optimize import minimize
import gc

# Define the risk-free rate
rF = 0.0443  # Current risk-free rate (4.43% - 1-year US T-bill)

methods = ["pcr", "rf", "wma", "gru"]

predicted_returns_dir = "preds"
covariance_matrix_file = f"{predicted_returns_dir}/covariance_matrix.csv"

avg_pred_returns = []

for method in methods:
    # Load predicted returns
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
from functools import reduce
combined_avg_returns = reduce(lambda left, right: pd.merge(left, right, on='Ticker', how='inner'), avg_pred_returns)

# Check if combined_avg_returns is empty
if combined_avg_returns.empty:
    raise ValueError("No overlapping tickers found among methods after merging.")

# Calculate Expected Return per Ticker as mean across methods
predicted_return_cols = [f'Predicted_Returns_{method.upper()}' for method in methods]
combined_avg_returns['Expected_Return'] = combined_avg_returns[predicted_return_cols].mean(axis=1)

# Prepare grouped_data for optimization
grouped_data = combined_avg_returns[['Ticker', 'Expected_Return']]

### Load and Filter Covariance Matrix ###
# Load covariance matrix without specifying dtype
cov_matrix_df = pd.read_csv(
    covariance_matrix_file,
    index_col=0
)

# Convert the data to float
cov_matrix_df = cov_matrix_df.astype(np.float32)

cov_matrix_df.index = cov_matrix_df.index.astype(str)
cov_matrix_df.columns = cov_matrix_df.columns.astype(str)

# Get unique tickers 
unique_tickers = grouped_data["Ticker"].values

# Ensure the covariance matrix includes all the unique tickers
missing_tickers = set(unique_tickers) - set(cov_matrix_df.index)
if missing_tickers:
    raise ValueError(f"Tickers {missing_tickers} are missing in the covariance matrix.")

# Filter the covariance matrix to include only the tickers in unique_tickers
cov_matrix_df = cov_matrix_df.loc[unique_tickers, unique_tickers]
cov_matrix = cov_matrix_df.values

# Ensure covariance matrix matches the number of assets
n_assets = len(unique_tickers)
if cov_matrix.shape != (n_assets, n_assets):
    raise ValueError("Covariance matrix size does not match the number of unique tickers.")

### Mean-Variance Optimization ###
def mean_variance_optimization(expected_returns, cov_matrix, target_return=None):
    """
    Perform mean-variance portfolio optimization.

    Args:
        expected_returns (np.ndarray): Array of expected returns.
        cov_matrix (np.ndarray): Covariance matrix.
        target_return (float, optional): Target return for the portfolio.

    Returns:
        np.ndarray: Optimal portfolio weights.
    """
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return weights @ (cov_matrix @ weights)

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]

    if target_return is not None:
        # Add target return constraint
        constraints.append({"type": "eq", "fun": lambda weights: weights @ expected_returns - target_return})

    # Bounds for weights (long-only portfolio)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess (equal distribution)
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize using SLSQP 
    result = minimize(
        portfolio_variance,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if not result.success:
        raise ValueError("Optimization failed: " + result.message)

    return result.x

# Optimize for the minimum variance portfolio
optimal_weights = mean_variance_optimization(grouped_data["Expected_Return"].values, cov_matrix)

### Calculate Portfolio Metrics ###
portfolio_return = optimal_weights @ grouped_data["Expected_Return"].values
portfolio_volatility = np.sqrt(optimal_weights @ (cov_matrix @ optimal_weights))
portfolio_sharpe_ratio = (portfolio_return - rF) / portfolio_volatility
portfolio_alpha = portfolio_return - rF

### Append Results to the DataFrame ###
grouped_data["Optimal_Weights"] = optimal_weights

# Get individual asset volatilities
asset_volatilities = np.sqrt(np.diag(cov_matrix))
grouped_data["Volatility"] = asset_volatilities

# Calculate Sharpe Ratio for each asset
grouped_data["Sharpe_Ratio"] = (grouped_data["Expected_Return"] - rF) / grouped_data["Volatility"]

### Portfolio Metrics ###
portfolio_metrics = {
    "Portfolio_Return": portfolio_return,
    "Portfolio_Volatility": portfolio_volatility,
    "Portfolio_Sharpe_Ratio": portfolio_sharpe_ratio,
    "Portfolio_Alpha": portfolio_alpha,
}

# Export
#grouped_data.to_csv("mean_variance_optimized_portfolio.csv", index=False)
#pd.DataFrame([portfolio_metrics]).to_csv("mean_variance_portfolio_metrics.csv", index=False)

# Display
#print("Optimal Portfolio Weights and Metrics:")
#print(grouped_data[["Ticker", "Optimal_Weights", "Expected_Return", "Volatility", "Sharpe_Ratio"]])
#print("\nPortfolio Metrics:")
#print(portfolio_metrics)
#print("\nResults saved to 'mean_variance_optimized_portfolio.csv' and 'mean_variance_portfolio_metrics.csv'")

### Sparse Portfolio Optimization ###
def sparse_portfolio_optimization(expected_returns, cov_matrix, l1_lambda):
    """
    Perform sparse portfolio optimization using L1 regularization.

    Args:
        expected_returns (np.ndarray): Array of expected returns.
        cov_matrix (np.ndarray): Covariance matrix.
        l1_lambda (float): L1 regularization strength.

    Returns:
        np.ndarray: Optimal sparse portfolio weights.
    """
    n_assets = len(expected_returns)

    def objective(weights):
        # Portfolio variance + L1 regularization penalty
        variance = weights @ (cov_matrix @ weights)
        penalty = l1_lambda * np.sum(np.abs(weights))
        return variance + penalty

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1}]

    # Bounds for weights (long-only portfolio)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess (equal distribution)
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize using SLSQP
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

# Hyperparameter tuning for sparsity
lambdas = [0.01, 0.05, 0.1, 0.5, 1.0]  # List of L1 regularization strengths
best_sparse_weights = None
best_sparse_metrics = None
best_lambda = None
highest_sharpe_ratio = float("-inf")

for l1_lambda in lambdas:
    try:
        sparse_weights = sparse_portfolio_optimization(
            grouped_data["Expected_Return"].values,
            cov_matrix,
            l1_lambda
        )

        # Calculate sparse portfolio metrics
        sparse_portfolio_return = sparse_weights @ grouped_data["Expected_Return"].values
        sparse_portfolio_volatility = np.sqrt(sparse_weights @ (cov_matrix @ sparse_weights))
        sparse_portfolio_sharpe_ratio = (sparse_portfolio_return - rF) / sparse_portfolio_volatility

        # Save results if Sharpe ratio improves
        if sparse_portfolio_sharpe_ratio > highest_sharpe_ratio:
            best_sparse_weights = sparse_weights
            best_sparse_metrics = {
                "Portfolio_Return": sparse_portfolio_return,
                "Portfolio_Volatility": sparse_portfolio_volatility,
                "Portfolio_Sharpe_Ratio": sparse_portfolio_sharpe_ratio,
                "Portfolio_Alpha": sparse_portfolio_return - rF,
            }
            best_lambda = l1_lambda
            highest_sharpe_ratio = sparse_portfolio_sharpe_ratio
    except ValueError as e:
        print(f"Optimization failed for λ={l1_lambda}: {str(e)}")

# Append Sparse Results to the DataFrame
grouped_data["Sparse_Weights"] = best_sparse_weights
grouped_data["Sparse_Sharpe_Ratio"] = (grouped_data["Expected_Return"] - rF) / grouped_data["Volatility"]

# Export sparse portfolio results
grouped_data.to_csv("optimization_outputs/sparse_mean_variance_optimized_portfolio.csv", index=False)
pd.DataFrame([best_sparse_metrics]).to_csv("optimization_outputs/sparse_mean_variance_portfolio_metrics.csv", index=False)

print(f"Best λ: {best_lambda}")
print("Sparse Portfolio Weights and Metrics:")
print(grouped_data[["Ticker", "Sparse_Weights", "Expected_Return", "Volatility", "Sparse_Sharpe_Ratio"]])
print("\nSparse Portfolio Metrics:")
print(best_sparse_metrics)
print("\nResults saved to 'sparse_mean_variance_optimized_portfolio.csv' and 'sparse_mean_variance_portfolio_metrics.csv'")

### Show top-10 holdings in portfolio ### 

non_zero_sparse_weights = grouped_data[grouped_data["Sparse_Weights"] > 0]

top_10_holdings = non_zero_sparse_weights.nlargest(10, "Sparse_Weights")

# Reduce the covariance matrix to include only the top 10 holdings
top_10_tickers = top_10_holdings["Ticker"].values
top_10_cov_matrix = cov_matrix_df.loc[top_10_tickers, top_10_tickers].values

# Calculate metrics for the top 10 holdings
top_10_sparse_metrics = {
    "Top_10_Sparse_Weights_Sum": top_10_holdings["Sparse_Weights"].sum(),
    "Top_10_Sparse_Return": (top_10_holdings["Sparse_Weights"] * top_10_holdings["Expected_Return"]).sum(),
    "Top_10_Sparse_Volatility": np.sqrt(
        top_10_holdings["Sparse_Weights"].values @ (top_10_cov_matrix @ top_10_holdings["Sparse_Weights"].values)
    ),
}

# Calculate Sharpe ratio for the top 10 sparse portfolio
top_10_sparse_metrics["Top_10_Sparse_Sharpe"] = (
    (top_10_sparse_metrics["Top_10_Sparse_Return"] - rF) /
    top_10_sparse_metrics["Top_10_Sparse_Volatility"]
)

# Export top-10 holdings
pd.DataFrame([top_10_sparse_metrics]).to_csv("optimization_outputs/top_10_sparse_portfolio_metrics.csv", index=False)

print("\nTop 10 Sparse Portfolio Metrics:")
print(top_10_sparse_metrics)