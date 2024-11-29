import pandas as pd
import os
import matplotlib.pyplot as plt

results_dir = "preds"
wma_results_file = os.path.join(results_dir, "wma_validation_results.csv")
pcr_results_file = os.path.join(results_dir, "pcr_validation_results.csv")
rf_results_file = os.path.join(results_dir, "rf_validation_results.csv")
gru_results_file = os.path.join(results_dir, "gru_validation_results.csv")

wma_results = pd.read_csv(wma_results_file)
pcr_results = pd.read_csv(pcr_results_file)
rf_results = pd.read_csv(rf_results_file)
gru_results = pd.read_csv(gru_results_file)

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
    gru_results[["Ticker", "Normalized_MSE"]].rename(columns={"Normalized_MSE": "GRU_Normalized_MSE"}),
    on="Ticker"
)

# Summary statistics
wma_mean_mse = comparison_df["WMA_Normalized_MSE"].mean()
pcr_mean_mse = comparison_df["PCR_Normalized_MSE"].mean()
rf_mean_mse = comparison_df["RF_Normalized_MSE"].mean()
gru_mean_mse = comparison_df["GRU_Normalized_MSE"].mean()

print(f"Overall Mean Normalized MSE:")
print(f"  WMA: {wma_mean_mse}")
print(f"  PCR: {pcr_mean_mse}")
print(f"  RF: {rf_mean_mse}")
print(f"  GRU: {gru_mean_mse}")

output_dir = "figures"

# Comparison plot
comparison_df.set_index("Ticker").plot(kind="bar", figsize=(12, 6), width=0.8)
plt.title("Normalized MSE Comparison by Ticker")
plt.ylabel("Normalized MSE")
plt.xlabel("Tickers")
plt.legend(["WMA", "PCR", "RF", "GRU"])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "mse_comparison_plot.png"))
plt.show()

# Export 
comparison_df.to_csv(os.path.join(output_dir, "mse_comparison_results.csv"), index=False)
print(f"Comparison results saved to mse_comparison_results.csv")