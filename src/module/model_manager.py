import pandas as pd
import joblib
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import logging
from module.logger import Logger

# Initialize the logger
Logger(log_dir="log", log_file="model_manager.log")

try:
    from xgboost import XGBClassifier
    has_xgboost = True
except ImportError:
    has_xgboost = False

class ModelManager:

    def __init__(self, dataset_path, model_dir="model"):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            "LogisticRegression": LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
        }
        if has_xgboost:
            self.models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    @staticmethod
    def get_available_models():
        models = ["RandomForest", "SVM", "KNN", "MLP", "LogisticRegression"]
        try:
            import xgboost
            models.append("XGBoost")
        except ImportError:
            pass
        return models

    def train_and_save_models(self):
        df = pd.read_csv(self.dataset_path)
        X = df.drop(columns=["user", "position", "environment", "weight", "capture_id", "local_timestamp"], errors="ignore")
        y = df["user"]

        # Binary classification - one-vs-all per user (excluding 'empty' as a user)
        unique_users = [u for u in y.unique() if u != "empty"]
        for target_user in unique_users:
            y_binary = y.apply(lambda label: target_user if label == target_user else f"not_{target_user}")
            if len(y_binary.unique()) < 2:
                logging.warning(f"⚠️ Skipping binary training for '{target_user}': only one class present.")
                continue
            for name, model in self.models.items():
                n_components = min(10, X.shape[1], len(X))
                logging.info(f"ℹ️ PCA using {n_components} components for user '{target_user}'")
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=n_components)),
                    ("clf", model)
                ])
                pipeline.fit(X, y_binary)
                model_path = os.path.join(self.model_dir, f"{name}_binary_{target_user}_vs_all.joblib")
                joblib.dump(pipeline, model_path)
                logging.info(f"✅ {name} binary model ({target_user} vs. all) saved at: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and save binary and one-vs-all models from CSI dataset")
    parser.add_argument("-d", "--dataset", default="../dataset/dataset.csv", help="Path to the dataset CSV")
    parser.add_argument("-m", "--model_dir", default="../model", help="Directory to save trained models")
    args = parser.parse_args()

    manager = ModelManager(dataset_path=args.dataset, model_dir=args.model_dir)
    manager.train_and_save_models()
    logging.info("✅ All models (binary and one-vs-all) trained and saved successfully.")
