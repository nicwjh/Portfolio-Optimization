import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize

# Load PCR, RF, WMA results
pcr_results = pd.read_csv("preds/pcr_validation_results.csv")
rf_results = pd.read_csv("preds/rf_validation_results.csv")
wma_results = pd.read_csv("preds/wma_validation_results.csv")

# Combine results into a single DataFrame
results = pd.merge(
    pcr_results[["Ticker", "Normalized_MSE", "Predicted_Return"]],
    rf_results[["Ticker", "Normalized_MSE", "Predicted_Return"]],
    on="Ticker",
    suffixes=("_PCR", "_RF")
)
results = pd.merge(
    results,
    wma_results[["Ticker", "Normalized_MSE", "Predicted_Return"]],
    on="Ticker"
)
results.rename(columns={"Normalized_MSE": "Normalized_MSE_WMA", "Predicted_Return": "Predicted_Return_WMA"}, inplace=True)

# Define confidence levels (inverse of normalized MSE)
results["Confidence_PCR"] = 1 / results["Normalized_MSE_PCR"]
results["Confidence_RF"] = 1 / results["Normalized_MSE_RF"]
results["Confidence_WMA"] = 1 / results["Normalized_MSE_WMA"]

# Equilibrium returns (assume equal weights)
market_weights = np.ones(len(results)) / len(results)
cov_matrix = np.eye(len(results))  # Placeholder, replace with actual covariance matrix
tau = 0.025  # Scaling factor for uncertainty in equilibrium returns
pi = tau * cov_matrix.dot(market_weights)

# Views (predicted returns)
views = results[["Predicted_Return_PCR", "Predicted_Return_RF", "Predicted_Return_WMA"]].mean(axis=1).values
P = np.eye(len(results))  # One view per asset
Q = views

# Confidence matrix
Omega = np.diag(results[["Confidence_PCR", "Confidence_RF", "Confidence_WMA"]].mean(axis=1))

# Calculate adjusted returns using Black-Litterman formula
M_inverse = inv(inv(tau * cov_matrix) + P.T @ inv(Omega) @ P)
adjusted_returns = M_inverse.dot(inv(tau * cov_matrix).dot(pi) + P.T @ inv(Omega).dot(Q))

# Portfolio optimization
def portfolio_optimization(cov_matrix, expected_returns):
    n_assets = len(expected_returns)

    def objective(weights):
        return -((weights @ expected_returns) / np.sqrt(weights @ cov_matrix @ weights))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]  # Sum of weights = 1
    bounds = [(0, 1) for _ in range(n_assets)]  # Long-only portfolio

    result = minimize(
        objective,
        x0=np.ones(n_assets) / n_assets,
        bounds=bounds,
        constraints=constraints
    )
    return result.x

# Optimize portfolio
optimal_weights = portfolio_optimization(cov_matrix, adjusted_returns)

# Display results
results["Optimal_Weights"] = optimal_weights
print("Optimal Portfolio Weights:")
print(results[["Ticker", "Optimal_Weights"]])

# Save results
results.to_csv("black_litterman_optimized_portfolio.csv", index=False)
print("Portfolio optimization results saved to 'black_litterman_optimized_portfolio.csv'")