import joblib
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from module.logger import Logger
import logging

# Initialize the logger
Logger(log_dir="log", log_file="auth_csi.log")


class PipelineHandler:
    """Handles loading and training of the pipeline."""

    @staticmethod
    def load_pipeline(pipeline_path):
        """Load the trained pipeline."""
        return joblib.load(pipeline_path)

    @staticmethod
    def retrain_pipeline(dataset_path, pipeline_path):
        """Retrain the pipeline with the updated dataset."""
        df = pd.read_csv(dataset_path)
        X = df.drop(columns=["user", "position", "environment"])
        y = df["user"]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        joblib.dump(pipeline, pipeline_path)
        logging.info(f"✅ New pipeline saved at: {pipeline_path}")


class DatasetHandler:
    """Handles dataset operations such as appending new data."""

    @staticmethod
    def append_to_dataset(dataset_path, new_data, user):
        """Add a new capture to the dataset."""
        new_data = new_data.copy()
        new_data["user"] = user
        new_data["position"] = "unknown"
        new_data["environment"] = "unknown"
        if os.path.exists(dataset_path):
            existing = pd.read_csv(dataset_path)
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
        combined.to_csv(dataset_path, index=False)
        logging.info(f"✅ New capture added to dataset: {dataset_path}")
        return combined


class Authenticator:
    """Handles the authentication process."""

    @staticmethod
    def authenticate_with_threshold(input_data, pipeline, threshold):
        """Authenticate using the pipeline and a threshold."""
        # Remove non-feature columns
        feature_data = input_data.drop(columns=["user", "position", "environment"], errors="ignore")

        # Predict probabilities using the pipeline
        proba = pipeline.predict_proba(feature_data)
        predicted_class = pipeline.classes_[np.argmax(proba, axis=1)]
        max_prob = np.max(proba, axis=1)
        result = ["unknown" if prob < threshold else cls for prob, cls in zip(max_prob, predicted_class)]
        return result, max_prob

    @staticmethod
    def majority_vote(predictions):
        """Determine the majority vote from predictions."""
        return pd.Series(predictions).value_counts().idxmax()


def log_authentication_result(user, status, avg_proba, result_type, output_path="output/auth_results.csv"):
    """Log the authentication result and save it to a CSV file."""
    # Log the result
    logging.info(f"Result: user={user}, status={status}, accuracy={avg_proba:.2%}, type={result_type}")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the result to the CSV file
    with open(output_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Add header if the file is empty
            writer.writerow(["timestamp", "user", "status", "accuracy", "type"])
        writer.writerow([datetime.now().isoformat(), user, status, f"{avg_proba:.2%}", result_type])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Interactive CSI Authentication")
    parser.add_argument("-i", "--input", nargs="?", default="data/processed_csi.csv", help="Path to the capture CSV (default: processed_csi.csv)")
    parser.add_argument("-u", "--user", required=True, help="Expected user for authentication")
    parser.add_argument("-p", "--pipeline", default="model/authentication_pipeline.joblib", help="Trained pipeline")
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Acceptance threshold")
    parser.add_argument("-o", "--output", default="output/auth_results.csv", help="Path to the output results CSV")

    args = parser.parse_args()

    # Fixed path for the labeled dataset
    dataset_path = "dataset/dataset.csv"

    # Confirm the user name
    confirm_user = input(f"❓ Is the provided user '{args.user}' correct? (Y/n): ").strip().lower()
    if confirm_user not in ["", "y"]:
        logging.info("🚫 Operation canceled by the user.")
        exit(0)

    # Try to load the pipeline
    try:
        pipeline = PipelineHandler.load_pipeline(args.pipeline)
    except FileNotFoundError:
        logging.warning(f"⚠️ Pipeline file '{args.pipeline}' not found.")
        choice = input("❓ Do you want to add this capture to the dataset and train a new pipeline? (Y/n): ").strip().lower()
        if choice in ["", "y"]:
            logging.info("➕ Adding capture to the dataset and training a new pipeline...")
            df = pd.read_csv(args.input)
            DatasetHandler.append_to_dataset(dataset_path, df, args.user)
            PipelineHandler.retrain_pipeline(dataset_path, args.pipeline)
            logging.info("✅ New pipeline trained successfully.")
            # Reload the pipeline after training
            pipeline = PipelineHandler.load_pipeline(args.pipeline)
            log_authentication_result(args.user, "Not authenticated", 0.0, "calibrated", args.output)
        else:
            logging.info("🗑️ Capture discarded. Exiting.")
            exit(0)

    # Load input data
    df = pd.read_csv(args.input)
    df["position"] = "unknown"
    df["environment"] = "unknown"

    # Authenticate
    logging.info("🔍 Authenticating user...")
    predictions, probs = Authenticator.authenticate_with_threshold(df, pipeline, args.threshold)
    avg_proba = np.mean(probs)
    majority = Authenticator.majority_vote(predictions)

    if majority == args.user and avg_proba >= args.threshold:
        logging.info("✅ Authentication successful!")
        log_authentication_result(args.user, "Authenticated", avg_proba, "effective", args.output)
    else:
        logging.info("❌ Authentication failed.")
        choice = input("❓ Do you want to add this capture to the dataset for calibration? (Y/n): ").strip().lower()
        if choice in ["", "y"]:
            logging.info("➕ Adding capture to the dataset for calibration...")
            DatasetHandler.append_to_dataset(dataset_path, df, args.user)
            PipelineHandler.retrain_pipeline(dataset_path, args.pipeline)
            logging.info("✅ Pipeline updated successfully.")
            log_authentication_result(args.user, "Not authenticated", avg_proba, "calibrated", args.output)
        else:
            logging.info("🗑️ Capture discarded. Authentication marked as effective.")
            log_authentication_result(args.user, "Not authenticated", avg_proba, "effective", args.output)


if __name__ == "__main__":
    main()
