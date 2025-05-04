import pandas as pd
import numpy as np
import argparse
import joblib
import os
import logging
from scipy.spatial.distance import euclidean
from module.logger import Logger

# Setup logging
Logger(log_dir="log", log_file="auth_csi.log")

OUTPUT_DIR = "output"
MODEL_DIR = "model"

SAFE_ZONE_LIMIT = 3.0  # Maximum distance to the user's centroid to consider "authenticated"
CONFIDENCE_THRESHOLD = 0.6  # Minimum probability threshold to accept authentication
PRESENCE_THRESHOLD = 0.8  # Minimum probability threshold for human presence


class CSIAuthenticator:
    def __init__(self, processed_csv, target_user):
        self.processed_csv = processed_csv
        self.target_user = target_user
        self.df_new = None
        self.X_new_scaled = None
        self.presences = []

    def load_data(self):
        """Load and preprocess the input data."""
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR, exist_ok=True)

        self.df_new = pd.read_csv(self.processed_csv)
        selected_cols = pd.read_csv(os.path.join(MODEL_DIR, "selected_features.csv"), header=None)[0].tolist()
        X_new = self.df_new[selected_cols]

        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        self.X_new_scaled = pd.DataFrame(scaler.transform(X_new), columns=X_new.columns)

    def authenticate_multiclass(self):
        """Authenticate using the multiclass Random Forest model."""
        model_rf = joblib.load(os.path.join(MODEL_DIR, "random_forest_model.joblib"))
        predictions = model_rf.predict(self.X_new_scaled)
        probas = model_rf.predict_proba(self.X_new_scaled)
        class_labels = model_rf.classes_

        probas_dict = {label: np.mean(probas[:, i]) for i, label in enumerate(class_labels)}
        counts = pd.Series(predictions).value_counts().to_dict()

        logging.info("--- Multiclass Model (Random Forest) ---")
        logging.info(f"Prediction counts: {counts}")
        for label, val in probas_dict.items():
            logging.info(f"Average probability for class '{label}': {val:.4f}")

        if self.target_user in class_labels:
            logging.info(f"Average probability for user '{self.target_user}': {probas_dict[self.target_user]:.4f}")
        else:
            logging.warning(f"User '{self.target_user}' not found in the Random Forest model.")

    def authenticate_binary(self):
        """Authenticate using binary models (user vs others)."""
        model_types = ["logreg", "svm", "mlp"]
        for model_type in model_types:
            model_path = os.path.join(MODEL_DIR, f"model_{self.target_user}_vs_users_{model_type}.joblib")
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            proba = model.predict_proba(self.X_new_scaled)[:, 1]
            avg_proba = np.mean(proba)
            logging.info(f"--- Binary Model: {model_type.upper()} ({self.target_user} vs Other Users) ---")
            logging.info(f"Average probability for '{self.target_user}': {avg_proba:.4f}")

    def check_human_presence(self):
        """Check human presence using binary models."""
        model_types = ["logreg", "svm", "mlp"]
        logging.info("--- Checking Human Presence (vs. Empty Environment) ---")
        for model_type in model_types:
            presence_model_path = os.path.join(MODEL_DIR, f"model_presence_binary_{model_type}.joblib")
            if not os.path.exists(presence_model_path):
                continue
            model = joblib.load(presence_model_path)
            proba = model.predict_proba(self.X_new_scaled)[:, 1]
            avg_proba = np.mean(proba)
            self.presences.append(avg_proba)
            logging.info(f"[{model_type.upper()}] Average probability of human presence: {avg_proba:.4f}")

    def authenticate_user_vs_empty(self):
        """Authenticate user vs empty environment."""
        model_types = ["logreg", "svm", "mlp"]
        logging.info(f"--- Model {self.target_user} vs Empty (All Positions) ---")
        for model_type in model_types:
            model_path = os.path.join(MODEL_DIR, f"model_{self.target_user}_vs_empty_all_positions_{model_type}.joblib")
            scaler_path = os.path.join(MODEL_DIR, f"scaler_{self.target_user}_vs_empty_all_positions.joblib")
            features_path = os.path.join(MODEL_DIR, f"features_{self.target_user}_vs_empty_all_positions.csv")
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            scaler_user = joblib.load(scaler_path)
            selected_features = pd.read_csv(features_path, header=None)[0].tolist()
            X_spec = self.df_new[selected_features]
            X_spec_scaled = scaler_user.transform(X_spec)
            proba = model.predict_proba(X_spec_scaled)[:, 1]
            avg_proba = np.mean(proba)
            logging.info(f"[{model_type.upper()}] Average probability for '{self.target_user}' (all positions vs empty): {avg_proba:.4f}")

            # Safe zone verification
            dataset_path = os.path.join("dataset", "dataset.csv")
            if os.path.exists(dataset_path):
                df_ref = pd.read_csv(dataset_path)
                df_user = df_ref[df_ref["user"] == self.target_user]
                user_centroid = df_user[selected_features].mean().values
                capture_mean = X_spec.mean().values
                distance = euclidean(capture_mean, user_centroid)
                logging.info(f"Average distance to '{self.target_user}' centroid: {distance:.4f}")

                if avg_proba < CONFIDENCE_THRESHOLD or distance > SAFE_ZONE_LIMIT or max(self.presences) < PRESENCE_THRESHOLD:
                    logging.warning("⚠️ Pattern rejected. No reliable user presence detected within the safe authentication zone.")
                else:
                    logging.info("✅ User successfully authenticated within the safe zone.")

    def run(self):
        """Run the entire authentication pipeline."""
        self.load_data()
        self.authenticate_multiclass()
        self.authenticate_binary()
        self.check_human_presence()
        self.authenticate_user_vs_empty()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authenticate CSI for a specific user or environment")
    parser.add_argument("-i", "--input", default="data/processed_csi.csv", help="Input CSV file")
    parser.add_argument("-u", "--user", required=True, help="Target user for authentication (do not use 'empty')")
    args = parser.parse_args()

    if args.user == "empty":
        logging.error("The label 'empty' represents the empty environment and should not be used as a target user.")
        exit(1)

    authenticator = CSIAuthenticator(args.input, args.user)
    authenticator.run()
