import joblib
import pandas as pd
import numpy as np
import os
import csv
import logging
import argparse
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from module.logger import Logger
from module.validation_manager import ValidationManager
from module.model_manager import ModelManager

# Initialize logger
Logger(log_dir="log", log_file="auth_csi.log")


class PipelineHandler:

    @staticmethod
    def load_pipeline(pipeline_path):
        """Load a pre-trained pipeline from the specified path."""
        return joblib.load(pipeline_path)

    @staticmethod
    def retrain_pipeline(dataset_path, pipeline_path):
        """Retrain the pipeline using the updated dataset."""
        df = pd.read_csv(dataset_path)
        df['weight'] = df['weight'].fillna(1.0)
        sample_weight = df['weight'].values
        X = df.drop(columns=["user", "position", "environment", "weight", "capture_id", "local_timestamp"], errors="ignore")
        y = df["user"]
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        pipeline.fit(X, y, clf__sample_weight=sample_weight)
        joblib.dump(pipeline, pipeline_path)
        logging.info(f"✅ New pipeline saved at: {pipeline_path}")


class DatasetHandler:

    @staticmethod
    def append_to_dataset(dataset_path, new_data, user, weight=1.0):
        """Append new data to the dataset or create a new dataset if it doesn't exist."""
        import uuid
        new_data = new_data.copy()
        new_data["user"] = user
        new_data["weight"] = weight
        if "capture_id" not in new_data.columns or new_data["capture_id"].isnull().any():
            capture_id = str(uuid.uuid4())
            new_data["capture_id"] = capture_id

        if os.path.exists(dataset_path):
            existing = pd.read_csv(dataset_path)
            combined = pd.concat([existing, new_data], ignore_index=True)
            logging.info(f"✅ Existing dataset found. Appending new capture to: {dataset_path}")
        else:
            combined = new_data
            logging.info(f"⚠️ Dataset file not found. Creating a new dataset at: {dataset_path}")

        combined.to_csv(dataset_path, index=False)
        logging.info(f"✅ New capture added to dataset: {dataset_path}")

    @staticmethod
    def remove_oldest_capture(dataset_path, user):
        """Remove the oldest capture for a specific user from the dataset."""
        if os.path.exists(dataset_path):
            existing = pd.read_csv(dataset_path)
            if "capture_id" in existing.columns:
                user_captures = existing[existing["user"] == user]
                if not user_captures.empty:
                    oldest_capture_id = user_captures["capture_id"].iloc[0]
                    logging.info(f"🗑️ Removing oldest capture for user '{user}' with capture_id '{oldest_capture_id}'.")
                    existing = existing[~((existing["user"] == user) & (existing["capture_id"] == oldest_capture_id))]
                    existing.to_csv(dataset_path, index=False)
                    return True
        return False


class AuthCSI:

    def __init__(self, input_path="data/processed_csi.csv", user=None, threshold=0.75, output_path="output/auth_results.csv",
                 dataset_path="dataset/dataset.csv", model_dir="model"):
        """Initialize the authentication process with the required parameters."""
        self.input_path = input_path
        self.user = user
        self.threshold = threshold
        self.output_path = output_path
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        self.models = ModelManager.get_available_models()

    def log_authentication_result(self, user, accuracies, result_type):
        """Log the authentication result to a CSV file."""
        accuracy_str = ', '.join([f"{model}: {accuracy:.4f}" for model, accuracy in accuracies.items()])
        logging.info(f"Result: user={user}, accuracies={accuracy_str}, type={result_type}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(["timestamp", "user", "accuracies", "type"])
            writer.writerow([datetime.now().isoformat(), user, accuracy_str, result_type])

    def authenticate_with_modelmanager(self, input_df):
        """Authenticate the user using all available models."""
        feature_data = input_df.drop(columns=["user", "position", "environment", "weight", "capture_id", "local_timestamp"], errors="ignore")
        results = []

        for model_name in self.models:
            user_detected = "unknown"
            best_score = 0.0
            binary_models = [f for f in os.listdir(self.model_dir) if f.startswith(f"{model_name}_binary_") and "_vs_all.joblib" in f]

            for model_file in binary_models:
                user_label = model_file[len(f"{model_name}_binary_"):-len("_vs_all.joblib")]
                model_path = os.path.join(self.model_dir, model_file)
                model = joblib.load(model_path)
                proba = model.predict_proba(feature_data)

                classes = model.classes_
                proba_df = pd.DataFrame(proba, columns=[f"{model_name}_{cls}" for cls in classes])

                if proba_df.shape[1] == 2 and user_label in model.classes_:
                    idx = list(model.classes_).index(user_label)
                    score = proba_df.iloc[0, idx]

                    if score > best_score:
                        best_score = score
                        user_detected = user_label

            if best_score < self.threshold:
                user_detected = "unknown"

            results.append((model_name, user_detected, best_score))

        return results

    def run(self):
        """Run the authentication process."""
        # Check if the dataset exists
        if not os.path.exists(self.dataset_path):
            logging.warning(f"⚠️ Dataset file not found: {self.dataset_path}. Creating a new dataset...")
            df_input = pd.read_csv(self.input_path)
            DatasetHandler.append_to_dataset(self.dataset_path, df_input, self.user)
            logging.info(f"✅ New dataset created at: {self.dataset_path}")
            
            # Authenticate to calculate accuracies even for calibration data
            result_models = self.authenticate_with_modelmanager(df_input)
            accuracies = {model_name: avg_proba for model_name, _, avg_proba in result_models}
            self.log_authentication_result(self.user, accuracies, "calibrated")
            logging.info("ℹ️ Data added for calibration. Authentication skipped.")
            
            # Train all models after creating the dataset
            model_manager = ModelManager(dataset_path=self.dataset_path, model_dir=self.model_dir)
            model_manager.train_and_save_models()
            logging.info("✅ All models trained after dataset creation.")
            return

        # Check if models are trained
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
        if not model_files:
            logging.warning(f"⚠️ No trained models found in directory: {self.model_dir}. Training models...")
            model_manager = ModelManager(dataset_path=self.dataset_path, model_dir=self.model_dir)
            model_manager.train_and_save_models()
            logging.info("✅ All models trained successfully.")

        logging.info("🔍 [STEP 1] ValidationManager - PCA-based environment analysis:")
        validator = ValidationManager(self.dataset_path, self.input_path, self.user)
        validator.generate_all_visualizations()

        df_input = pd.read_csv(self.input_path)
        feature_data = df_input.drop(columns=["user", "position", "environment", "weight", "capture_id", "local_timestamp"], errors="ignore")

        logging.info("🔎 [STEP 2] Binary classification - one-vs-all user analysis:")

        binary_accuracies = {}
        authenticated_binary_models = []
        binary_models_found = False

        for model_name in self.models:
            model_file = os.path.join(self.model_dir, f"{model_name}_binary_{self.user}_vs_all.joblib")
            if os.path.exists(model_file):

                if self.user == "empty":
                    logging.warning("⚠️ Skipping binary authentication for 'empty' (not a valid user).")
                    break

                binary_models_found = True
                model = joblib.load(model_file)
                if self.user in model.classes_:
                    idx = list(model.classes_).index(self.user)
                    proba = model.predict_proba(feature_data)[0][idx]
                    binary_accuracies[f"Binary_{model_name}"] = proba
                    if proba >= self.threshold:
                        authenticated_binary_models.append(model_name)
                        logging.info(f"✅ {model_name} (binary) authenticated user '{self.user}' with probability {proba:.4f}")
                    else:
                        logging.info(f"❌ {model_name} (binary) rejected user '{self.user}' with probability {proba:.4f}")

        if not binary_models_found:
            logging.warning(f"⚠️ No binary models found matching expected pattern for user '{self.user}'. Skipping STEP 2.")
            logging.warning(f"🔎 Expected format: {{model}}_binary_{self.user}_vs_all.joblib in {self.model_dir}")

        logging.info("🔐 [STEP 3] User authentication - 'user' vs 'others':")
        result_models = self.authenticate_with_modelmanager(df_input)

        authenticated_models = []
        user_accuracies = {}
        for model_name, predicted_user, avg_proba in result_models:
            logging.info(f"🔎 Model: {model_name} | Predicted: {predicted_user} | Score: {avg_proba:.2%}")
            if predicted_user == self.user:
                authenticated_models.append(model_name)
                user_accuracies[model_name] = avg_proba

        effective_accuracies = {
            model: score for model, score in {**user_accuracies, **binary_accuracies}.items()
            if score >= self.threshold
        }

        if authenticated_models or authenticated_binary_models:
            combined = authenticated_models + authenticated_binary_models
            logging.info(f"✅ User '{self.user}' authenticated by: {', '.join(combined)}")
            self.log_authentication_result(self.user, effective_accuracies, "authenticated")
        else:
            logging.warning(f"❌ User '{self.user}' was NOT authenticated by any model.")
            self.log_authentication_result(self.user, effective_accuracies, "not_authenticated")

        confirm = input(f"❓ Add this capture to dataset for user '{self.user}'? (Y/n): ").strip().lower()
        if confirm in ["", "y", "yes"]:
            logging.info("📂 [STEP 4] Dataset update - user confirmation and model retraining")
            DatasetHandler.append_to_dataset(self.dataset_path, df_input, self.user)
            model_manager = ModelManager(dataset_path=self.dataset_path, model_dir=self.model_dir)
            model_manager.train_and_save_models()
            self.log_authentication_result(self.user, effective_accuracies, "calibrated")
            logging.info("✅ New capture added and models retrained.")
            self.log_dataset_statistics()
        else:
            self.log_authentication_result(self.user, effective_accuracies, "effective")
            logging.info("⚠️ Capture was not added to the dataset.")

        if authenticated_models or authenticated_binary_models:
            logging.info(f"✅ Final result: user '{self.user}' authenticated by: {', '.join(combined)}")
        else:
            logging.warning(f"❌ Final result: user '{self.user}' was not authenticated.")

    def log_dataset_statistics(self):
        """Log statistics about the dataset."""
        df_dataset = pd.read_csv(self.dataset_path)
        user_stats = df_dataset.groupby("user").agg(
            total_captures=("capture_id", "nunique"),
            total_samples=("capture_id", "count")
        )
        logging.info(f"📊 Dataset statistics:\n{user_stats}")


def main():
    """Main entry point for the authentication process."""
    parser = argparse.ArgumentParser(description="CSI Authentication with ValidationManager and ModelManager")
    parser.add_argument("-i", "--input", default="data/processed_csi.csv", help="Input capture CSV")
    parser.add_argument("-u", "--user", required=True, help="Expected user")
    parser.add_argument("-t", "--threshold", type=float, default=0.75, help="Acceptance threshold")
    parser.add_argument("-o", "--output", default="output/auth_results.csv", help="Output log file")
    parser.add_argument("-d", "--dataset", default="dataset/dataset.csv", help="Dataset CSV")
    parser.add_argument("-m", "--model_dir", default="model", help="Model directory")
    args = parser.parse_args()

    auth = AuthCSI(
        input_path=args.input,
        user=args.user,
        threshold=args.threshold,
        output_path=args.output,
        dataset_path=args.dataset,
        model_dir=args.model_dir
    )
    auth.run()


if __name__ == "__main__":
    main()
