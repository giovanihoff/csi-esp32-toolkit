import logging
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report
from module.logger import Logger

# Setup logging
Logger(log_dir="log", log_file="train_csi.log")

TOP_SUBCARRIERS = 52

class CSITrainer:
    def __init__(self, train_path, model_dir, features_path, top_subcarriers=TOP_SUBCARRIERS):
        self.train_path = train_path
        self.model_dir = model_dir
        self.features_path = features_path
        self.top_subcarriers = top_subcarriers
        self.df = None
        self.amplitude_cols = None
        self.users = None

    def ensure_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load_data(self):
        self.df = pd.read_csv(self.train_path)
        self.users = [u for u in self.df["user"].unique() if u != "empty"]
        self.amplitude_cols = [col for col in self.df.columns if col.startswith("amplitude_")]

    def select_discriminative_features(self, label_col, df=None):
        if df is None:
            df = self.df
        unique_users = df[label_col].unique()
        diffs = {}
        for col in self.amplitude_cols:
            user_means = [df[df[label_col] == user][col].mean() for user in unique_users]
            diffs[col] = max(user_means) - min(user_means)
        sorted_cols = sorted(diffs, key=diffs.get, reverse=True)
        return sorted_cols[:self.top_subcarriers]

    def save_features(self, selected_cols, filename):
        pd.DataFrame(selected_cols).to_csv(filename, index=False, header=False)

    def train_model(self, X, y, model_name):
        if len(np.unique(y)) < 2:
            logging.warning(f"Insufficient data to train '{model_name}' (only one class present). Training skipped.")
            return
        clf_log = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf_log.fit(X, y)
        joblib.dump(clf_log, os.path.join(self.model_dir, f"{model_name}_logreg.joblib"))
        logging.info(f"Logistic Regression model trained and saved for {model_name}.")
        svm_params = {"C": [0.1, 1, 10], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf"]}
        grid_svm = GridSearchCV(SVC(probability=True, class_weight="balanced", random_state=42),
                                svm_params, scoring=make_scorer(f1_score), cv=3, n_jobs=-1)
        grid_svm.fit(X, y)
        joblib.dump(grid_svm.best_estimator_, os.path.join(self.model_dir, f"{model_name}_svm.joblib"))
        logging.info(f"Best SVM parameters for {model_name}: {grid_svm.best_params_}")
        mlp_params = {"hidden_layer_sizes": [(64,), (64, 32)], "alpha": [0.0001, 0.001],
                      "early_stopping": [True], "max_iter": [1000]}
        grid_mlp = GridSearchCV(MLPClassifier(random_state=42), mlp_params,
                                scoring=make_scorer(f1_score), cv=3, n_jobs=-1)
        grid_mlp.fit(X, y)
        joblib.dump(grid_mlp.best_estimator_, os.path.join(self.model_dir, f"{model_name}_mlp.joblib"))
        logging.info(f"Best MLP parameters for {model_name}: {grid_mlp.best_params_}")

    def train_multiclass_model(self, X_scaled, y):
        clf_rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        clf_rf.fit(X_scaled, y)
        joblib.dump(clf_rf, os.path.join(self.model_dir, "random_forest_model.joblib"))
        logging.info("Random Forest model trained and saved.")
        logging.info("Classification report for Random Forest:\n" + classification_report(y, clf_rf.predict(X_scaled)))

    def train_human_presence_model(self, X_scaled):
        logging.info("Training human presence model vs empty")
        self.df["presence"] = (self.df["user"] != "empty").astype(int)
        self.train_model(X_scaled, self.df["presence"], "model_presence_binary")

    def train_binary_models(self, X_scaled):
        for user in self.users:
            logging.info(f"Training model: {user} vs empty (all positions)")
            df_user = self.df[self.df["user"] == user]
            df_empty = self.df[self.df["user"] == "empty"].sample(n=len(df_user), random_state=42)
            df_combo = pd.concat([df_user, df_empty]).sample(frac=1, random_state=42).reset_index(drop=True)
            df_combo["label"] = (df_combo["user"] == user).astype(int)
            selected_user_cols = self.select_discriminative_features("label", df_combo)
            self.save_features(selected_user_cols, os.path.join(self.model_dir, f"features_{user}_vs_empty_all_positions.csv"))
            X_user = df_combo[selected_user_cols]
            y_user = df_combo["label"]
            scaler_user = StandardScaler()
            X_user_scaled = scaler_user.fit_transform(X_user)
            joblib.dump(scaler_user, os.path.join(self.model_dir, f"scaler_{user}_vs_empty_all_positions.joblib"))
            joblib.dump(X_user.mean().values, os.path.join(self.model_dir, f"centroid_{user}_vs_empty_all_positions.joblib"))
            self.train_model(X_user_scaled, y_user, f"model_{user}_vs_empty_all_positions")

    def run(self):
        self.ensure_dir(self.model_dir)
        self.load_data()
        selected_cols = self.select_discriminative_features("user")
        self.save_features(selected_cols, self.features_path)
        X = self.df[selected_cols]
        y = self.df["user"].values
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        joblib.dump(scaler, os.path.join(self.model_dir, "scaler.joblib"))
        self.train_multiclass_model(X_scaled, y)
        self.train_human_presence_model(X_scaled)
        self.train_binary_models(X_scaled)
        logging.info("Models trained and saved successfully.")

if __name__ == "__main__":
    trainer = CSITrainer(
        train_path="dataset/dataset.csv",
        model_dir="model",
        features_path="model/selected_features.csv",
        top_subcarriers=TOP_SUBCARRIERS
    )
    trainer.run()
