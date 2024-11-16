import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import minimize

# Load the data
methods = ["pcr", "rf", "wma", "gru"]
predicted_returns_data = []
validation_results_data = []

# Read all the predicted returns and validation results files
for method in methods:
    predicted_returns_data.append(pd.read_csv(f"preds/{method}_predicted_returns.csv"))
    validation_results_data.append(pd.read_csv(f"preds/{method}_validation_results.csv"))

# Ensure predicted returns column names are dynamically handled
for i, data in enumerate(predicted_returns_data):
    print(f"Columns in {methods[i]}_predicted_returns.csv:")
    print(data.columns)

# Dynamically check and assign the correct column name for predicted returns
predicted_returns_column_map = {}
for i, df in enumerate(predicted_returns_data):
    columns = df.columns
    if "Predicted_Returns" in columns:
        predicted_returns_column_map[methods[i]] = "Predicted_Returns"
    elif "Predicted_Adj_Close" in columns:
        predicted_returns_column_map[methods[i]] = "Predicted_Adj_Close"
    else:
        raise KeyError(f"Unable to find the predicted returns column in {methods[i]}_predicted_returns.csv")

# Merge all validation results into a single DataFrame
validation_merged = validation_results_data[0]
for i in range(1, len(validation_results_data)):
    validation_merged = pd.merge(
        validation_merged,
        validation_results_data[i][["Ticker", "Normalized_MSE"]],
        on="Ticker",
        suffixes=("", f"_{methods[i].upper()}")
    )

# Add the normalized MSE for the last method manually
validation_merged.rename(columns={"Normalized_MSE": f"Normalized_MSE_{methods[0].upper()}"}, inplace=True)

# Merge predicted returns with validation results
for i, method in enumerate(methods):
    predicted_column = predicted_returns_column_map[method]
    validation_merged = pd.merge(
        validation_merged,
        predicted_returns_data[i][["Ticker", predicted_column]],
        on="Ticker",
        suffixes=("", f"_{method.upper()}")
    )
validation_merged.rename(
    columns={f"{predicted_returns_column_map[methods[0]]}": f"Predicted_Returns_{methods[0].upper()}"},
    inplace=True
)

# Define confidence levels (inverse of normalized MSE)
for method in methods:
    validation_merged[f"Confidence_{method.upper()}"] = 1 / validation_merged[f"Normalized_MSE_{method.upper()}"]

# Black-Litterman parameters
market_weights = np.ones(len(validation_merged)) / len(validation_merged)  # Equal weights as equilibrium
cov_matrix = np.eye(len(validation_merged))  # Placeholder for covariance matrix
tau = 0.025  # Scaling factor for uncertainty in equilibrium returns
pi = tau * cov_matrix.dot(market_weights)

# Views (mean of predicted returns from the three methods)
validation_merged["Mean_Predicted_Return"] = validation_merged[
    [f"Predicted_Returns_{method.upper()}" for method in methods]
].mean(axis=1)
views = validation_merged["Mean_Predicted_Return"].values

# P matrix (identity matrix: one view per asset)
P = np.eye(len(validation_merged))
Q = views

# Confidence matrix (diagonal of mean confidence levels)
validation_merged["Mean_Confidence"] = validation_merged[
    [f"Confidence_{method.upper()}" for method in methods]
].mean(axis=1)
Omega = np.diag(validation_merged["Mean_Confidence"].values)

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

# Save and display results
validation_merged["Optimal_Weights"] = optimal_weights
print("Optimal Portfolio Weights:")
print(validation_merged[["Ticker", "Optimal_Weights"]])

# Save results
validation_merged.to_csv("black_litterman_optimized_portfolio.csv", index=False)
print("Portfolio optimization results saved to 'black_litterman_optimized_portfolio.csv'")