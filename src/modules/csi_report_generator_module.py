# csi_report_generator_module.py

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
import base64
import os
import logging

logger = logging.getLogger(__name__)

def embed_image(path):
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

def generate_html_report(metrics_dir="metrics", output="metrics/final_report.html"):
    env = Environment(loader=FileSystemLoader("modules/templates"))
    template = env.get_template("relatorio_template.html")

    images = {
        "pca_visualization": embed_image(f"{metrics_dir}/pca_visualization.png"),
        "pca_3d_visualization": embed_image(f"{metrics_dir}/pca_3d_visualization.png"),
        "correlation_heatmap": embed_image(f"{metrics_dir}/correlation_heatmap.png"),
        "pca_loadings": embed_image(f"{metrics_dir}/pca_loadings.png"),
        "scree_plot": embed_image(f"{metrics_dir}/scree_plot.png"),
        "pca_biplot": embed_image(f"{metrics_dir}/pca_biplot.png")
    }

    labels = ['aline', 'giovani', 'ines']
    classifiers = ['random_forest', 'k-nearest_neighbors', 'svm_(rbf_kernel)']

    metrics_data = {}

    for classifier in classifiers:
        metrics_data[classifier] = {}
        for label in labels:
            filename = f"{metrics_dir}/{classifier}_{label}_metrics.csv"
            if os.path.exists(filename):
                metrics_data[classifier][label] = pd.read_csv(filename)
                logger.info(f"Metrics loaded: {filename}")
            else:
                logger.warning(f"Metrics file not found: {filename}")
    
    # Feature Importance do Random Forest do primeiro label como exemplo
    feat_imp_path = f"{metrics_dir}/feature_importance.csv"
    feat_imp = pd.read_csv(feat_imp_path) if os.path.exists(feat_imp_path) else pd.DataFrame()

    html = template.render(
        images=images,
        metrics_data=metrics_data,
        feat_imp=feat_imp,
        labels=labels
    )

    with open(output, "w") as f:
        f.write(html)

    logger.info("📄 HTML report generated.")

if __name__ == "__main__":
    generate_html_report()

