# csi_auth.py

import os
import subprocess
import argparse
import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.ensemble import IsolationForest
from modules.csi_logger_module import setup_logger

logger = setup_logger(__name__, "auth.log")

def run_capture():
    try:
        subprocess.run(["python", "csi_capture.py"], check=True)
    except subprocess.CalledProcessError:
        logger.error("❌ CSI capture script failed. Aborting authentication.")
        exit(1)

def run_processing():
    try:
        subprocess.run(["python", "csi_processing.py"], check=True)
        logger.info("✅ CSI data processed successfully.")
    except subprocess.CalledProcessError:
        logger.error("❌ CSI processing script failed. Aborting authentication.")
        exit(1)

def validate_pca_distance(df_dataset, df_new, fixed_threshold=None):
    df_dataset["label"] = df_dataset["label"].astype(str)
    df_dataset["__source__"] = "dataset"
    df_new["label"] = "unknown"
    df_new["__source"] = "capture"
    df_combined = pd.concat([df_dataset, df_new], ignore_index=True)

    X_all = df_combined.select_dtypes(include=[np.number])
    y_all = df_combined["label"]
    X_scaled_all = StandardScaler().fit_transform(X_all)

    # Fit PCA to explain 95% of variance
    pca_full = PCA(n_components=0.95)
    X_pca_full = pca_full.fit_transform(X_scaled_all)

    df_pca = pd.DataFrame(X_pca_full)
    df_pca["label"] = y_all.values
    df_pca["__source"] = df_combined["__source"]

    centroids = {
        label: df_pca[df_pca["label"] == label].iloc[:, :-2].mean().values
        for label in df_dataset["label"].unique()
    }

    unknown_points = df_pca[df_pca["__source"] == "capture"].iloc[:, :-2].values
    avg_distances = {
        label: np.mean(pairwise_distances(unknown_points, [centroid]))
        for label, centroid in centroids.items()
    }

    # Compute adaptive threshold
    adaptive_thresholds = {}
    for label in df_dataset["label"].unique():
        cluster_points = df_pca[df_pca["label"] == label].iloc[:, :-2].values
        centroid = centroids[label]
        dists = pairwise_distances(cluster_points, [centroid]).flatten()
        adaptive_thresholds[label] = np.percentile(dists, 95)

    closest_label, closest_dist = min(avg_distances.items(), key=lambda x: x[1])
    threshold = adaptive_thresholds.get(closest_label, fixed_threshold or 3.5)

    if closest_dist > threshold:
        logger.warning("❗ Closest cluster '%s' is too far (%.2f > %.2f). Marking as UNKNOWN.", closest_label, closest_dist, threshold)
        return {"label": "unknown", "distance": avg_distances}
    else:
        logger.info("🖐️ Closest cluster: %s (%.2f)", closest_label, closest_dist)
        return {"label": closest_label, "distance": avg_distances}

def apply_isolation_forest(label, X_scaled, models_dir):
    iso_path = os.path.join(models_dir, f"isoforest_{label}.joblib")
    if not os.path.exists(iso_path):
        logger.warning("⚠️ IsolationForest model for '%s' not found. Skipping outlier validation.", label)
        return True

    isoforest = joblib.load(iso_path)
    predictions = isoforest.predict(X_scaled)
    inliers = np.sum(predictions == 1)
    total = len(predictions)
    inlier_ratio = inliers / total
    logger.info("🌲 IsolationForest for '%s': %d/%d inliers (%.2f%%)", label, inliers, total, inlier_ratio * 100)

    if inlier_ratio < 0.8:
        logger.warning("❗ Too many outliers for label '%s'. Marking as UNKNOWN.", label)
        return False
    return True

def print_final_summary(results):
    print("\n================================")
    print("🔐 FINAL AUTHENTICATION SUMMARY")
    print("================================")

    binary = []
    multi = []
    for model, result in results.items():
        if model in ["PCA_Distance", "OutlierCheck"]:
            continue
        if result['type'] == 'binary':
            binary.append((model, result['label'], result['score'], result['accepted']))
        else:
            counts = result['distribution']
            total = sum(counts.values())
            top_label = result['label']
            top_count = counts[top_label]
            multi.append((model, top_label, f"{top_count}/{total}"))

    if binary:
        print("\n[BINARY MODELS]")
        print(f"{'Model':<35} {'Target':<10} {'% Positive':<12} {'Auth'}")
        for m, l, s, a in binary:
            print(f"{m:<35} {l:<10} {s*100:>10.2f}%    {'✅' if a else '❌'}")

    if multi:
        print("\n[MULTICLASS MODELS]")
        print(f"{'Model':<35} {'Top Prediction':<15} {'Confidence'}")
        for m, l, c in multi:
            print(f"{m:<35} {l:<15} {c}")

    if "PCA_Distance" in results:
        dist_data = results["PCA_Distance"]
        print("\n[PCA VALIDATION]")
        print(f"Closest cluster: {dist_data['label']}")
        print("Distance to clusters:")
        for label, dist in dist_data['distance'].items():
            print(f"- {label:<10}: {dist:.2f}")
        if dist_data['label'] == 'unknown':
            print("❗ Distance too high. Identity marked as UNKNOWN.")
        else:
            print("✅ Distance acceptable. Identity confirmed.")

    if "OutlierCheck" in results:
        print("\n[OUTLIER DETECTION]")
        if results["OutlierCheck"] == "unknown":
            print("❗ Too many outliers detected. Identity rejected.")
        else:
            print("✅ IsolationForest check passed.")

def authenticate(models_dir="model", scaler_path="model/scaler_csi.joblib", features_path="data/csi_features.csv", dataset_path="dataset/dataset.csv", distance_threshold=3.5, binary_threshold=0.5):
    logger.info("🔄 Loading models and scaler...")
    scaler = joblib.load(scaler_path)
    df_new = pd.read_csv(features_path)
    X_new = df_new.select_dtypes(include=[np.number]).copy()
    X_new_scaled = scaler.transform(X_new)

    model_files = [
        f for f in os.listdir(models_dir)
        if f.startswith("model_") and f.endswith(".joblib")
    ]

    if not model_files:
        logger.error("❌ No model files found in directory: %s", models_dir)
        exit(1)

    results = {}

    for model_file in sorted(model_files):
        model_path = os.path.join(models_dir, model_file)
        model_name = os.path.splitext(model_file)[0]
        logger.info("🔍 Loading model: %s", model_file)
        clf = joblib.load(model_path)

        if hasattr(clf, 'n_features_in_') and clf.n_features_in_ != X_new_scaled.shape[1]:
            logger.warning("⚠️ Skipping model '%s' due to feature mismatch: model expects %d, got %d",
                           model_name, clf.n_features_in_, X_new_scaled.shape[1])
            continue

        y_pred = clf.predict(X_new_scaled)

        if set(y_pred) <= {0, 1}:  # Binary model
            label = model_name.split("_")[-2]
            score = np.mean(y_pred)
            logger.info("📊 [BINARY] %s - %% predicted as positive (label %s): %.2f", model_name, label, score)
            accepted = score >= binary_threshold
            results[model_name] = {"type": "binary", "label": label, "score": score, "accepted": accepted}
            logger.info("✅ [BINARY] Authenticated: %s", "YES" if accepted else "NO")
        else:
            prediction_counts = Counter(y_pred)
            most_common_label, count = prediction_counts.most_common(1)[0]
            logger.info("📊 [MULTICLASS] %s - Prediction summary:", model_name)
            for label, cnt in prediction_counts.items():
                logger.info("- %s: %d", label, cnt)
            logger.info("✅ [MULTICLASS] Most likely identity: %s", most_common_label)
            results[model_name] = {
                "type": "multiclass",
                "label": most_common_label,
                "distribution": dict(prediction_counts)
            }

    logger.info("🔎 Performing hybrid PCA-based distance validation...")
    df_dataset = pd.read_csv(dataset_path)
    df_new_for_pca = pd.read_csv(features_path)
    pca_result = validate_pca_distance(df_dataset, df_new_for_pca, distance_threshold)
    results["PCA_Distance"] = pca_result

    predicted_label = pca_result["label"]
    if predicted_label != "unknown":
        valid = apply_isolation_forest(predicted_label, X_new_scaled, models_dir)
        if not valid:
            results["OutlierCheck"] = "unknown"
        else:
            results["OutlierCheck"] = predicted_label

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Authenticate CSI capture against trained models")
    parser.add_argument("--models_dir", type=str, default="model", help="Directory containing trained models")
    parser.add_argument("--scaler", type=str, default="model/scaler_csi.joblib", help="Path to trained scaler")
    parser.add_argument("--features", type=str, default="data/csi_features.csv", help="Path to input feature CSV")
    parser.add_argument("--dataset", type=str, default="dataset/dataset.csv", help="Path to original labeled dataset")
    parser.add_argument("--distance_threshold", type=float, default=3.5, help="PCA distance threshold for cluster validation")
    parser.add_argument("--binary_threshold", type=float, default=0.5, help="Threshold to classify binary prediction as authenticated")
    parser.add_argument("--capture", action="store_true", help="Run CSI capture and processing before authentication")
    args = parser.parse_args()

    if args.capture:
        run_capture()
        run_processing()

    results = authenticate(
        models_dir=args.models_dir,
        scaler_path=args.scaler,
        features_path=args.features,
        dataset_path=args.dataset,
        distance_threshold=args.distance_threshold,
        binary_threshold=args.binary_threshold
    )

    print_final_summary(results)

