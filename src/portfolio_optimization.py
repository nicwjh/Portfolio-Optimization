import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Define the risk-free rate
rF = 0.0443  # Current risk-free rate (4.43%)

# Load the data
methods = ["pcr", "rf", "wma", "gru"]
predicted_returns_data = {}
validation_results_data = {}

# Read all the predicted returns and validation results files
for method in methods:
    predicted_returns_data[method] = pd.read_csv(f"preds/{method}_predicted_returns.csv")
    validation_results_data[method] = pd.read_csv(f"preds/{method}_validation_results.csv")

# Merge predicted returns and validation results
tickers = predicted_returns_data[methods[0]]['Ticker']
combined_data = pd.DataFrame({'Ticker': tickers})

for method in methods:
    preds = predicted_returns_data[method]
    val_res = validation_results_data[method]

    # Ensure the predicted returns column is correctly identified
    if 'Predicted_Returns' in preds.columns:
        pred_col = 'Predicted_Returns'
    elif 'Simple_Returns' in preds.columns:
        pred_col = 'Simple_Returns'
    else:
        raise KeyError(f"Unable to find the predicted returns column in {method}_predicted_returns.csv")

    preds = preds[['Ticker', pred_col]]
    preds.rename(columns={pred_col: f'Predicted_Returns_{method.upper()}'}, inplace=True)

    val_res = val_res[['Ticker', 'Normalized_MSE']]
    val_res.rename(columns={'Normalized_MSE': f'Normalized_MSE_{method.upper()}'}, inplace=True)

    combined_data = combined_data.merge(preds, on='Ticker')
    combined_data = combined_data.merge(val_res, on='Ticker')

# Calculate mean predicted returns
predicted_return_cols = [f'Predicted_Returns_{method.upper()}' for method in methods]
combined_data['Expected_Return'] = combined_data[predicted_return_cols].mean(axis=1)

# Load the covariance matrix
cov_matrix_df = pd.read_csv("preds/covariance_matrix.csv", index_col=0)
cov_matrix = cov_matrix_df.values

# Check if the covariance matrix size matches the number of assets
n_assets = len(combined_data)
if cov_matrix.shape != (n_assets, n_assets):
    raise ValueError("Covariance matrix size does not match the number of assets.")

# Calculate Volatility (standard deviation)
combined_data['Volatility'] = np.sqrt(np.diag(cov_matrix))

### Mean-Variance Optimization ###

def mean_variance_optimization(expected_returns, cov_matrix, target_return=None):
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}]  # Sum of weights equals 1

    if target_return is not None:
        # Add target return constraint
        constraints.append({'type': 'eq', 'fun': lambda weights: weights.T @ expected_returns - target_return})

    # Bounds for weights (long-only portfolio)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess (equal weights)
    initial_weights = np.ones(n_assets) / n_assets

    # Optimize
    result = minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result.x

# Set a target return (optional)
# You can set a target return, for example, the average of expected returns
# target_return = combined_data['Expected_Return'].mean()

# For this example, we'll optimize for the minimum variance portfolio without a target return
optimal_weights = mean_variance_optimization(combined_data['Expected_Return'].values, cov_matrix)

# Calculate portfolio metrics
portfolio_return = optimal_weights @ combined_data['Expected_Return'].values
portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
portfolio_sharpe_ratio = (portfolio_return - rF) / portfolio_volatility
portfolio_alpha = portfolio_return - rF

# Save results
combined_data['Optimal_Weights'] = optimal_weights
combined_data['Sharpe_Ratio'] = (combined_data['Expected_Return'] - rF) / combined_data['Volatility']

# Portfolio metrics
portfolio_metrics = {
    'Portfolio_Return': portfolio_return,
    'Portfolio_Volatility': portfolio_volatility,
    'Portfolio_Sharpe_Ratio': portfolio_sharpe_ratio,
    'Portfolio_Alpha': portfolio_alpha
}

# Save to CSV
combined_data.to_csv('mean_variance_optimized_portfolio.csv', index=False)
pd.DataFrame([portfolio_metrics]).to_csv('mean_variance_portfolio_metrics.csv', index=False)

# Display results
print("Optimal Portfolio Weights and Metrics:")
print(combined_data[['Ticker', 'Optimal_Weights', 'Expected_Return', 'Volatility', 'Sharpe_Ratio']])
print("\nPortfolio Metrics:")
print(portfolio_metrics)
print("\nResults saved to 'mean_variance_optimized_portfolio.csv' and 'mean_variance_portfolio_metrics.csv'")