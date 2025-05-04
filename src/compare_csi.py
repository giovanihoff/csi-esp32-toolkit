import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

MODEL_DIR = "model"
DATASET_PATH = "dataset/dataset.csv"
OUTPUT_DIR = "output"

def plot_comparison(processed_csv, target_user):
    """Generate comparison plots between the captured data and the dataset for a specific user."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the processed CSI data
    df_processed = pd.read_csv(processed_csv)

    # Load the dataset
    df_dataset = pd.read_csv(DATASET_PATH)

    # Filter the dataset for the target user
    df_user = df_dataset[df_dataset["user"] == target_user]

    # Load the selected features
    selected_features_path = os.path.join(MODEL_DIR, "selected_features.csv")
    if not os.path.exists(selected_features_path):
        raise FileNotFoundError(f"Selected features file not found: {selected_features_path}")
    selected_features = pd.read_csv(selected_features_path, header=None)[0].tolist()

    # Ensure the selected features exist in both datasets
    missing_features = [f for f in selected_features if f not in df_processed.columns or f not in df_user.columns]
    if missing_features:
        raise ValueError(f"Missing features in the datasets: {missing_features}")

    # Calculate the mean of the selected features for the user in the dataset
    user_mean = df_user[selected_features].mean()

    # Calculate the mean of the selected features for the captured data
    capture_mean = df_processed[selected_features].mean()

    # --- Bar Plot ---
    plt.figure(figsize=(12, 6))
    x = np.arange(len(selected_features))
    plt.bar(x - 0.2, user_mean, width=0.4, label=f"{target_user} (Dataset)", color="blue", alpha=0.7)
    plt.bar(x + 0.2, capture_mean, width=0.4, label="Captured Data", color="orange", alpha=0.7)
    plt.xticks(x, selected_features, rotation=90, fontsize=8)
    plt.ylabel("Mean Amplitude")
    plt.title(f"Comparison of Captured Data vs Dataset for User: {target_user} (Bar Plot)")
    plt.legend()
    plt.tight_layout()
    bar_plot_path = os.path.join(OUTPUT_DIR, f"comparison_bar_{target_user}.png")
    plt.savefig(bar_plot_path)
    plt.close()

    # --- Line Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(selected_features, user_mean, label=f"{target_user} (Dataset)", marker="o", color="blue", alpha=0.7)
    plt.plot(selected_features, capture_mean, label="Captured Data", marker="o", color="orange", alpha=0.7)
    plt.xticks(rotation=90, fontsize=8)
    plt.ylabel("Mean Amplitude")
    plt.title(f"Comparison of Captured Data vs Dataset for User: {target_user} (Line Plot)")
    plt.legend()
    plt.tight_layout()
    line_plot_path = os.path.join(OUTPUT_DIR, f"comparison_line_{target_user}.png")
    plt.savefig(line_plot_path)
    plt.close()

    # --- Scatter Plot ---
    plt.figure(figsize=(8, 8))
    plt.scatter(user_mean, capture_mean, alpha=0.7, color="purple")
    plt.plot([min(user_mean), max(user_mean)], [min(user_mean), max(user_mean)], color="red", linestyle="--", label="Ideal Match")
    plt.xlabel(f"{target_user} (Dataset Mean)")
    plt.ylabel("Captured Data Mean")
    plt.title(f"Scatter Plot of Captured Data vs Dataset for User: {target_user}")
    plt.legend()
    plt.tight_layout()
    scatter_plot_path = os.path.join(OUTPUT_DIR, f"comparison_scatter_{target_user}.png")
    plt.savefig(scatter_plot_path)
    plt.close()

    # --- Boxplot ---
    combined_data = pd.concat([df_user[selected_features].assign(Source="Dataset"),
                                df_processed[selected_features].assign(Source="Captured Data")])
    plt.figure(figsize=(12, 6))
    combined_data_melted = combined_data.melt(id_vars="Source", var_name="Feature", value_name="Amplitude")
    sns.boxplot(x="Feature", y="Amplitude", hue="Source", data=combined_data_melted)
    plt.xticks(rotation=90, fontsize=8)
    plt.title(f"Boxplot of Captured Data vs Dataset for User: {target_user}")
    plt.tight_layout()
    boxplot_path = os.path.join(OUTPUT_DIR, f"comparison_boxplot_{target_user}.png")
    plt.savefig(boxplot_path)
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"- Bar Plot: {bar_plot_path}")
    print(f"- Line Plot: {line_plot_path}")
    print(f"- Scatter Plot: {scatter_plot_path}")
    print(f"- Boxplot: {boxplot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate comparison plots between captured data and dataset for a specific user.")
    parser.add_argument("-i", "--input", default="data/processed_csi.csv", help="Input CSV file for captured data")
    parser.add_argument("-u", "--user", required=True, help="Target user for comparison")
    args = parser.parse_args()

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

    plot_comparison(args.input, args.user)