# csi_train_models.py

import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

from modules.csi_logger_module import setup_logger

logger = setup_logger(__name__, "training.log")

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Train CSI user authentication model.")
parser.add_argument("--input", type=str, default="dataset/dataset.csv", help="Labeled dataset for training")
parser.add_argument("--output_dir", type=str, default="model", help="Directory to store trained models and scaler")
args = parser.parse_args()

# === Prepare output directory ===
os.makedirs(args.output_dir, exist_ok=True)

# === Load dataset ===
logger.info("📥 Loading dataset from: %s", args.input)
df = pd.read_csv(args.input)

if "label" not in df.columns:
    logger.error("❌ Missing 'label' column in dataset. Aborting.")
    exit(1)

X = df.select_dtypes(include=[np.number]).copy()
y_original = df["label"].astype(str)

logger.info("🔍 Features: %d", X.shape[1])
logger.info("🔢 Samples: %d", len(X))
logger.info("🏷️ Labels: %s", sorted(y_original.unique()))

# === Normalize features ===
logger.info("⚙️ Normalizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
scaler_path = os.path.join(args.output_dir, "scaler_csi.joblib")
joblib.dump(scaler, scaler_path)
logger.info("💾 Scaler saved to: %s", scaler_path)

# === Define classifiers ===
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN_5": KNeighborsClassifier(n_neighbors=5)
}

# === Multi-class training ===
logger.info("🔄 Starting multi-class training...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_original, test_size=0.3, stratify=y_original, random_state=42)

for name, model in models.items():
    logger.info("🚀 [MULTI] Training %s classifier...", name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logger.info("📊 [MULTI] %s - Classification Report:\n%s", name, classification_report(y_test, y_pred))
    logger.info("🟪 [MULTI] %s - Confusion Matrix:\n%s", name, confusion_matrix(y_test, y_pred))
    model_path = os.path.join(args.output_dir, f"model_multiclass_{name}.joblib")
    joblib.dump(model, model_path)
    logger.info("✅ [MULTI] %s model saved to: %s", name, model_path)

# === Binary training (one-vs-all for each label) ===
for target_label in sorted(y_original.unique()):
    logger.info("🔁 Starting binary training for label: %s", target_label)
    y_binary = y_original.apply(lambda lbl: 1 if lbl == target_label else 0)
    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_scaled, y_binary, test_size=0.3, stratify=y_binary, random_state=42)

    for name, model in models.items():
        logger.info("🚀 [BINARY - %s] Training %s classifier...", target_label, name)
        model.fit(X_train_bin, y_train_bin)
        y_pred_bin = model.predict(X_test_bin)
        logger.info("📊 [BINARY - %s] %s - Classification Report:\n%s", target_label, name, classification_report(y_test_bin, y_pred_bin))
        logger.info("🟪 [BINARY - %s] %s - Confusion Matrix:\n%s", target_label, name, confusion_matrix(y_test_bin, y_pred_bin))
        model_path = os.path.join(args.output_dir, f"model_binary_{target_label}_{name}.joblib")
        joblib.dump(model, model_path)
        logger.info("✅ [BINARY - %s] %s model saved to: %s", target_label, name, model_path)

# === IsolationForest for each label ===
logger.info("🌲 Starting IsolationForest training for each label...")
for label in sorted(y_original.unique()):
    logger.info("🔍 Training IsolationForest for label: %s", label)
    X_label = X_scaled[y_original == label]
    isoforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    isoforest.fit(X_label)
    iso_path = os.path.join(args.output_dir, f"isoforest_{label}.joblib")
    joblib.dump(isoforest, iso_path)
    logger.info("✅ IsolationForest for %s saved to: %s", label, iso_path)

logger.info("🎉 All models (multi, binary, and IsolationForest) trained and saved successfully.")

