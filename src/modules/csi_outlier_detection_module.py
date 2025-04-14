# csi_outlier_detection_module.py

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers(X_pca, y):
    """
    Detects outliers using Local Outlier Factor (LOF) no espaço PCA

    Returns a DataFrame with the column 'outlier' = True/False
    """
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    y_pred_outlier = lof.fit_predict(X_pca)

    df_outliers = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    df_outliers['label'] = y
    df_outliers['outlier'] = y_pred_outlier == -1

    return df_outliers

if __name__ == "__main__":
    # Standalone example (when running this script directly)
    from sklearn.datasets import make_classification
    X, _ = make_classification(n_samples=300, n_features=15)
    y_dummy = np.random.choice(['A', 'B'], size=300)
    df = detect_outliers(X, y_dummy)
    print(df.head())
