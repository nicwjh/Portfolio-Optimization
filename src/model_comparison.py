import pandas as pd
import os
import matplotlib.pyplot as plt

results_dir = "preds"
wma_results_file = os.path.join(results_dir, "wma_validation_results.csv")
pcr_results_file = os.path.join(results_dir, "pcr_validation_results.csv")
rf_results_file = os.path.join(results_dir, "rf_validation_results.csv")
lstm_results_file = os.path.join(results_dir, "lstm_validation_results.csv")

wma_results = pd.read_csv(wma_results_file)
pcr_results = pd.read_csv(pcr_results_file)
rf_results = pd.read_csv(rf_results_file)
lstm_results = pd.read_csv(lstm_results_file)

# Merge results using Ticker as key 
comparison_df = pd.merge(
    wma_results[["Ticker", "Normalized_MSE"]].rename(columns={"Normalized_MSE": "WMA_Normalized_MSE"}),
    pcr_results[["Ticker", "Normalized_MSE"]].rename(columns={"Normalized_MSE": "PCR_Normalized_MSE"}),
    on="Ticker"
)
comparison_df = pd.merge(
    comparison_df,
    rf_results[["Ticker", "Normalized_MSE"]].rename(columns={"Normalized_MSE": "RF_Normalized_MSE"}),
    on="Ticker"
)
comparison_df = pd.merge(
    comparison_df,
    lstm_results[["Ticker", "Normalized_MSE"]].rename(columns={"Normalized_MSE": "LSTM_Normalized_MSE"}),
    on="Ticker"
)

# Calculate summary statistics
wma_mean_mse = comparison_df["WMA_Normalized_MSE"].mean()
pcr_mean_mse = comparison_df["PCR_Normalized_MSE"].mean()
rf_mean_mse = comparison_df["RF_Normalized_MSE"].mean()
lstm_mean_mse = comparison_df["LSTM_Normalized_MSE"].mean()

print(f"Overall Mean Normalized MSE:")
print(f"  WMA: {wma_mean_mse}")
print(f"  PCR: {pcr_mean_mse}")
print(f"  RF: {rf_mean_mse}")
print(f"  LSTM: {lstm_mean_mse}")

# Comparison plot
comparison_df.set_index("Ticker").plot(kind="bar", figsize=(12, 6), width=0.8)
plt.title("Normalized MSE Comparison by Ticker")
plt.ylabel("Normalized MSE")
plt.xlabel("Tickers")
plt.legend(["WMA", "PCR", "RF", "LSTM"])
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "mse_comparison_plot.png"))
plt.show()

# Export comparison DataFrame to CSV
comparison_df.to_csv(os.path.join(results_dir, "mse_comparison_results.csv"), index=False)
print(f"Comparison results saved to mse_comparison_results.csv")