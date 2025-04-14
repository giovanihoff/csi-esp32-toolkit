import matplotlib
matplotlib.use('Agg')

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from modules.csi_logger_module import setup_logger
from modules.csi_visualizations_module import (
    plot_pca_2d,
    plot_pca_3d,
    plot_correlation_heatmap,
    plot_pca_loadings,
    plot_scree_plot,
    plot_pca_biplot,
    plot_feature_distribution
)

# === Logger ===
logger = setup_logger(__name__, "visualization.log")

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Visualize CSI features and PCA analysis with filtering support.")
parser.add_argument("--input", type=str, default="dataset/dataset.csv", help="Input feature CSV file")
parser.add_argument("--output_dir", type=str, default="metrics", help="Output directory for plots")
parser.add_argument("--filter_label", type=str, help="Optional: Filter by specific label")
parser.add_argument("--filter_environment", type=str, help="Optional: Filter by specific environment")
parser.add_argument("--filter_position", type=str, help="Optional: Filter by specific position")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# === Load and preprocess data ===
logger.info("📥 Loading data from: %s", args.input)
df = pd.read_csv(args.input)

if "label" not in df.columns:
    logger.error("❌ Missing 'label' column. Cannot continue visualization.")
    exit(1)

# === Apply optional filters ===
if args.filter_label:
    df = df[df["label"] == args.filter_label]
if args.filter_environment:
    df = df[df["environment"] == args.filter_environment]
if args.filter_position:
    df = df[df["position"] == args.filter_position]

logger.info("📊 Final dataset size after filtering: %d samples", len(df))

X = df.select_dtypes(include=[np.number]).copy()
y = df["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === PCA ===
logger.info("🔄 Performing PCA...")
pca = PCA(n_components=min(10, X.shape[1]))
X_pca = pca.fit_transform(X_scaled)

# === Generate plots ===
logger.info("🖼️ Generating visualizations...")

plot_pca_2d(X_pca[:, :2], y.values, os.path.join(args.output_dir, "pca_2d.png"))
plot_pca_3d(X_pca[:, :3], y.values, os.path.join(args.output_dir, "pca_3d.png"))
plot_correlation_heatmap(X_scaled, X.columns, os.path.join(args.output_dir, "correlation_heatmap.png"))
plot_pca_loadings(pca, X.columns, os.path.join(args.output_dir, "pca_loadings.png"))
plot_scree_plot(pca, os.path.join(args.output_dir, "scree_plot.png"))
plot_pca_biplot(X_pca, pca, X.columns, y.values, os.path.join(args.output_dir, "pca_biplot.png"))

# === Export PCA stats ===
pd.DataFrame(pca.components_, columns=X.columns).to_csv(os.path.join(args.output_dir, "pca_components.csv"), index=False)
pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]).to_csv(os.path.join(args.output_dir, "pca_projection.csv"), index=False)
pd.Series(pca.explained_variance_ratio_).to_csv(os.path.join(args.output_dir, "pca_variance.csv"), index_label="PC", header=["explained_variance"])

# === Feature distribution plot ===
plot_feature_distribution(df, os.path.join(args.output_dir, "feature_distributions"))

logger.info("✅ All enhanced visualizations and exports completed.")

