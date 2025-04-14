import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_pca_2d(X_pca_2d, y, output_path):
    plt.figure(figsize=(8,6))
    for label_name in np.unique(y):
        idx = y == label_name
        plt.scatter(X_pca_2d[idx, 0], X_pca_2d[idx, 1], label=label_name, alpha=0.7)
    plt.legend()
    plt.title("Sample Visualization (PCA 2D)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pca_3d(X_pca_3d, y, output_path):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    for label_name in np.unique(y):
        idx = y == label_name
        ax.scatter(X_pca_3d[idx, 0], X_pca_3d[idx, 1], X_pca_3d[idx, 2], label=label_name, alpha=0.6, s=15)
    ax.set_title("PCA 3D Visualization")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correlation_heatmap(X_scaled, X_columns, output_path):
    corr_df = pd.DataFrame(X_scaled, columns=X_columns)
    corr = corr_df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pca_loadings(pca, X_columns, output_path):
    loading_df = pd.DataFrame(pca.components_[:2], columns=X_columns, index=["PC1", "PC2"]).T
    loading_df.plot(kind='bar', figsize=(14,6))
    plt.title("PCA Loadings (PC1 and PC2)")
    plt.xlabel("Features")
    plt.ylabel("Weight (Loading)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scree_plot(pca, output_path):
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             pca.explained_variance_ratio_, marker='o', label='Explained variance per component')
    plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
             np.cumsum(pca.explained_variance_ratio_), marker='x', linestyle='--', label='Cumulative variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance')
    plt.title('Scree Plot')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_pca_biplot(X_pca, pca, X_columns, y, output_path):
    plt.figure(figsize=(8,6))
    for label_name in np.unique(y):
        idx = y == label_name
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label_name, alpha=0.5)
    for i, feature in enumerate(X_columns):
        plt.arrow(0, 0,
                  pca.components_[0, i]*5,
                  pca.components_[1, i]*5,
                  color='r', alpha=0.5)
        plt.text(pca.components_[0, i]*5.2,
                 pca.components_[1, i]*5.2,
                 feature, fontsize=6, ha='center', va='center')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA Biplot (with Loadings)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_feature_distribution(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    label_col = "label"
    feature_cols = df.select_dtypes(include=[np.number]).columns
    for col in feature_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=label_col, y=col)
        plt.title(f"Distribution of '{col}' by label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()

