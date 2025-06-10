import os
import pandas as pd
import numpy as np
import matplotlib
import argparse
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import logging
from module.logger import Logger

# Initialize the logger
Logger(log_dir="log", log_file="validation_manager.log")

class ValidationManager:

    def __init__(self, dataset_path, input_path, user, distance_threshold=6.0, output_dir="output"):
        self.dataset_path = dataset_path
        self.input_path = input_path
        self.user = user
        self.distance_threshold = distance_threshold
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def load_and_prepare_data(self):
        ignore_cols = ["user", "position", "environment", "weight", "capture_id", "local_timestamp"]
        if not os.path.exists(self.dataset_path):
            logging.error(f"Dataset file not found: {self.dataset_path}")
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        df_dataset = pd.read_csv(self.dataset_path)
        df_new = pd.read_csv(self.input_path)
        df_new["user"] = self.user

        feature_cols = [col for col in df_dataset.columns if col not in ignore_cols]
        df_combined = pd.concat(
            [df_dataset[feature_cols + ["user"]], df_new[feature_cols + ["user"]]],
            ignore_index=True
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_combined[feature_cols])
        y_labels = df_combined["user"].values

        n_components = min(10, X_scaled.shape[1], len(X_scaled))
        logging.info(f"ℹ️ PCA using {n_components} components for validation")
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        self.X_old = X_pca[:-1]
        self.X_new = X_pca[-1].reshape(1, -1)
        self.y_old = y_labels[:-1]

        df_pca = pd.DataFrame(self.X_old, columns=[f"PC{i+1}" for i in range(self.X_old.shape[1])])
        df_pca["user"] = self.y_old
        pca_cols = [col for col in df_pca.columns if col.startswith("PC")]
        self.centroids = df_pca.groupby("user")[pca_cols].mean()

    def check_environment_and_user(self):
        if "empty" not in self.centroids.index:
            logging.warning("Class 'empty' not found in dataset. Background analysis may be compromised.")
            distance_empty = float("inf")
        else:
            empty_centroid = self.centroids.loc["empty"].values.reshape(1, -1)
            distance_empty = np.linalg.norm(self.X_new - empty_centroid)

        distances_to_users = {}
        for label in self.centroids.index:
            if label != "empty":
                user_centroid = self.centroids.loc[label].values.reshape(1, -1)
                distance = np.linalg.norm(self.X_new - user_centroid)
                distances_to_users[label] = distance

        if distance_empty <= self.distance_threshold:
            result = "Environment is empty"
            logging.info(f"Environment check: {result} (distance to empty: {distance_empty:.2f})")
            return result

        if distances_to_users:
            closest_user = min(distances_to_users, key=distances_to_users.get)
            result = f"User detected: {closest_user} (distance: {distances_to_users[closest_user]:.2f})"
            logging.info(f"Environment check: {result}")
            return result

        result = "Unknown environment state"
        logging.warning(result)
        return result

    def plot_pca(self):
        df_pca = pd.DataFrame(self.X_old, columns=[f"PC{i+1}" for i in range(self.X_old.shape[1])])
        df_pca["user"] = self.y_old
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        pc1, pc2, pc3 = "PC1", "PC2", "PC3"
        for user in df_pca["user"].unique():
            subset = df_pca[df_pca["user"] == user]
            ax.scatter(subset[pc1], subset[pc2], subset[pc3], label=user)
        ax.scatter(self.X_new[0][0], self.X_new[0][1], self.X_new[0][2], c='red', label="new_sample", s=100, marker='X')
        for label, row in self.centroids.iterrows():
            ax.text(row[pc1], row[pc2], row[pc3], label, fontsize=10)
        ax.set_title("PCA 3D - Cluster Separability")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "pca_validation.png"))
        plt.close()

    def plot_confusion_matrix(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_old, self.y_old)
        y_pred = knn.predict(self.X_old)
        cm = confusion_matrix(self.y_old, y_pred, labels=self.centroids.index)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=self.centroids.index, yticklabels=self.centroids.index, cmap="Blues")
        plt.title("Confusion Matrix (KNN Simulation)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix_sim.png"))
        plt.close()

    def plot_distance_histogram(self):
        if self.user not in self.centroids.index:
            logging.warning(f"User '{self.user}' not found in dataset for histogram.")
            return
        distances_to_user = np.linalg.norm(self.X_old - self.centroids.loc[self.user].values, axis=1)
        distance_user = np.linalg.norm(self.X_new - self.centroids.loc[self.user].values)
        plt.figure(figsize=(8, 5))
        plt.hist(distances_to_user, bins=30, alpha=0.7, label=f"Distances to '{self.user}' centroid")
        plt.axvline(distance_user, color='red', linestyle='--', label="New sample distance")
        plt.axvline(self.distance_threshold, color='green', linestyle=':', label="Distance threshold")
        plt.title("Histogram of Distances to Target User Centroid")
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "distance_histogram.png"))
        plt.close()

    def generate_all_visualizations(self):
        self.load_and_prepare_data()
        result = self.check_environment_and_user()
        self.plot_pca()
        self.plot_confusion_matrix()
        self.plot_distance_histogram()
        logging.info("✅ All visualizations generated and saved.")
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CSI visualizations and environment check")
    parser.add_argument("-d", "--dataset", default="../dataset/dataset.csv", help="Path to dataset")
    parser.add_argument("-i", "--input", default="../data/processed_csi.csv", help="Path to input capture")
    parser.add_argument("-u", "--user", required=True, help="User label for input")
    parser.add_argument("-t", "--threshold", type=float, default=6.0, help="Distance threshold")
    parser.add_argument("-o", "--output", default="../output", help="Directory to save output visualizations")
    args = parser.parse_args()

    validator = ValidationManager(dataset_path=args.dataset, input_path=args.input, user=args.user, distance_threshold=args.threshold, output_dir=args.output)
    result = validator.generate_all_visualizations()
    logging.info(f"✅ Final validation result: {result}")
