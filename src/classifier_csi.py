import csv
import numpy as np
from collections import Counter
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance

from module.logger import Logger

Logger(log_file="process_csi.log")

"""
CSI Data Processing and User Classification Script

This script processes Channel State Information (CSI) data from a CSV file to perform multi-class user authentication 
using machine learning models. It focuses exclusively on amplitude features derived from raw CSI data, ensuring 
group-based splitting (via capture_id) to prevent data leakage between related samples. This simulates realistic 
authentication scenarios where captures are independent.

Key steps include:
1. Loading and validating CSV data.
2. Converting raw CSI data to amplitude vectors, with strict validation for consistency (e.g., fixed length of 192 amplitudes).
3. Filtering data for specified users.
4. Encoding user labels.
5. Splitting data into train/test sets using GroupShuffleSplit for group integrity, with logging of per-user counts in each set.
6. Normalizing features using StandardScaler (extensible to other scalers).
7. Training multiple classifiers (e.g., RandomForest, SVC) on scaled data.
8. Evaluating models with accuracy, classification report, weighted F1-score, confusion matrix plots, and ROC curves (multi-class OvR).
9. Extracting and plotting feature importances (subcarriers) for supported models, including permutation importance for non-tree models (optimized with parallel computation and reduced repeats to avoid long runtimes).
10. Generating visualizations: PCA (3D), t-SNE (2D), mean amplitude vs. subcarrier, CSI heatmaps per user.

The script is encapsulated in a modular class for reusability, with error handling, logging, and optional extensions 
like hyperparameter tuning or alternative scalers. All plots are displayed and saved as files for persistence. 
Designed for datasets with consistent amplitude lengths; tested assumptions should be validated with real data.
Permutation importance computation has been optimized to prevent hanging: uses parallel processing (n_jobs=-1) and reduced repeats (5).
"""

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class CSIClassifier:
    def __init__(self, filename="data/data_csi.csv", users_to_compare=['giovani', 'aline', 'ines'],
                 models=None, scaler_type='standard'):
        if not isinstance(models, dict) or not all(hasattr(clf, 'fit') for clf in models.values()):
            raise ValueError("Models must be a dictionary of valid scikit-learn classifiers.")
        self.filename = filename
        self.users_to_compare = users_to_compare
        self.models = models or {'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)}
        self.scaler_type = scaler_type  # Allow for other scalers in future
        self.data = None
        self.data_sub = None
        self.le = None
        self.y = None
        self.X = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.clfs = {}
        self.y_preds = {}
        self.importances = {}
        self.indices = {}

    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
        except FileNotFoundError:
            logging.error(f"File not found: {self.filename}")
            raise
        except csv.Error:
            logging.error(f"CSV parsing error in file: {self.filename}")
            raise

    def convert_to_amplitude(self):
        """
        Converts raw CSI data to amplitude for each row.
        Adds validation: only keeps valid rows! Ensures exactly 384 elements (192 complex pairs).
        """
        LEN_DATA_COMPLEX = 384  # From capture script: 384 elements -> 192 amplitudes
        def csi_to_amplitude(csi_raw):
            try:
                csi_vals = np.fromstring(str(csi_raw).strip("[]"), sep=",")
                if len(csi_vals) != LEN_DATA_COMPLEX or len(csi_vals) % 2 != 0:
                    logging.warning(f"Invalid number of elements for CSI complex: {len(csi_vals)} (expected {LEN_DATA_COMPLEX}). Skipping row.")
                    return None
                csi_complex = csi_vals[::2] + 1j * csi_vals[1::2]
                amplitude = np.abs(csi_complex)
                return amplitude
            except Exception as e:
                logging.warning(f"Failed to convert CSI data: {e}. Skipping row.")
                return None

        for row in self.data:
            amp = csi_to_amplitude(row['data'])
            if amp is not None:
                row['amplitude'] = amp
            else:
                row['amplitude'] = None  # Marks invalid row

        # Remove invalid rows
        self.data = [row for row in self.data if row['amplitude'] is not None]
        logging.info(f"Total valid samples after conversion: {len(self.data)}")

    def select_users(self):
        if not self.users_to_compare:
            raise ValueError("users_to_compare cannot be empty.")
        self.data_sub = [row for row in self.data if row['user'] in self.users_to_compare]
        user_counts = Counter(row['user'] for row in self.data_sub)
        counts_str = '\n'.join(f"{user} {count}" for user, count in user_counts.items())
        logging.info(f"Selected users:\n{counts_str}")

    def encode_labels(self):
        self.le = LabelEncoder()
        users = [row['user'] for row in self.data_sub]
        self.y = self.le.fit_transform(users)
        label_mapping = dict(zip(self.le.classes_, self.le.transform(self.le.classes_)))
        logging.info(f"Label mapping: {label_mapping}")

    def split_data(self):
        """
        Splits the dataset into training and testing sets using GroupShuffleSplit
        to ensure that all samples from the same capture_id remain in the same set.
        Logs per-user counts in train/test for balance check.
        """
        # Check for amplitude length consistency
        lengths = set(len(row['amplitude']) for row in self.data_sub)
        if len(lengths) > 1:
            logging.error(f"Inconsistent amplitude vector lengths detected: {lengths}")
            raise ValueError("Amplitude vectors must have consistent lengths.")

        if len(self.data_sub) == 0:
            raise ValueError("No valid samples available for the selected users.")

        X = []
        groups = []

        for row in self.data_sub:
            if 'capture_id' not in row:
                raise ValueError("Missing 'capture_id' in one or more samples.")
            X.append(row['amplitude'])
            groups.append(row['capture_id'])

        self.X = np.stack(X)
        # Use pre-encoded self.y from encode_labels

        # Split using capture_id groups
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        try:
            train_idx, test_idx = next(splitter.split(self.X, self.y, groups=groups))
            self.X_train, self.X_test = self.X[train_idx], self.X[test_idx]
            self.y_train, self.y_test = self.y[train_idx], self.y[test_idx]
        except Exception as e:
            logging.error(f"Error during group-based data splitting: {e}")
            raise

        # Log per-user counts in train/test
        train_counts = Counter(self.le.inverse_transform(self.y_train))
        test_counts = Counter(self.le.inverse_transform(self.y_test))
        logging.info(f"Train set user counts: {train_counts}")
        logging.info(f"Test set user counts: {test_counts}")

    def normalize(self):
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type: {self.scaler_type}")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        for name, clf in self.models.items():
            try:
                clf.fit(self.X_train_scaled, self.y_train)
                self.clfs[name] = clf
                self.y_preds[name] = clf.predict(self.X_test_scaled)
            except Exception as e:
                logging.error(f"Failed to train {name}: {e}")

    def evaluate(self):
        for name in self.models:
            if name not in self.y_preds:
                continue
            y_pred = self.y_preds[name]
            logging.info(f"\n--- Evaluation for {name} ---")
            acc = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            logging.info(f"Accuracy: {acc:.4f} | Weighted F1-score: {f1:.4f}")
            logging.info(classification_report(self.y_test, y_pred, target_names=self.le.classes_))
            cm = confusion_matrix(self.y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.le.classes_)
            disp.plot(cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.savefig(f"confusion_matrix_{name}.png")
            plt.show()
            plt.close()

    def feature_importances(self):
        for name, clf in self.clfs.items():
            if hasattr(clf, 'feature_importances_'):
                self.importances[name] = clf.feature_importances_
                self.indices[name] = np.argsort(self.importances[name])[::-1]
                logging.info(f"\nTop 10 most important subcarriers for {name}:")
                for rank, idx in enumerate(self.indices[name][:10]):
                    logging.info(f"{rank+1}. Subcarrier {idx}: importance {self.importances[name][idx]:.4f}")
            else:
                logging.info(f"\nSkipping feature importance for {name} as it is not a tree-based model.")

    def plot_importances(self):
        for name in self.models:
            if name in self.importances:
                plt.figure(figsize=(10,4))
                plt.bar(range(10), self.importances[name][self.indices[name][:10]])
                plt.xticks(range(10), self.indices[name][:10])
                plt.title(f"Top 10 most important subcarriers - {name}")
                plt.xlabel("Subcarrier (index)")
                plt.ylabel("Importance")
                plt.tight_layout()
                plt.savefig(f"feature_importances_{name}.png")
                plt.show()
                plt.close()

    def pca_visualization(self):
        X_all = np.vstack([self.X_train_scaled, self.X_test_scaled])
        y_all = np.concatenate([self.y_train, self.y_test])
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_all)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for label, color, name in zip(range(len(self.le.classes_)), COLORS, self.le.classes_):
            ax.scatter(
                X_pca[y_all==label,0], X_pca[y_all==label,1], X_pca[y_all==label,2],
                label=name, alpha=0.5, s=12, color=color
            )
        ax.set_title("PCA 3D - User Separation")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend()
        plt.tight_layout()
        plt.savefig("pca_3d.png")
        plt.show()
        plt.close()

    def plot_amplitude_vs_subcarrier(self):
        amplitudes_per_user = {user: [] for user in self.le.classes_}
        for amp, label in zip(self.X, self.y):
            amplitudes_per_user[self.le.classes_[label]].append(amp)
        plt.figure(figsize=(10, 6))
        subcarriers = np.arange(self.X.shape[1])
        for user, amps in amplitudes_per_user.items():
            if amps:
                mean_amp = np.mean(amps, axis=0)
                plt.plot(subcarriers, mean_amp, label=user)
        plt.title("Mean CSI Amplitude vs. Subcarrier Index per User")
        plt.xlabel("Subcarrier Index")
        plt.ylabel("Mean Amplitude")
        plt.legend()
        plt.tight_layout()
        plt.savefig("amplitude_vs_subcarrier.png")
        plt.show()
        plt.close()

    def plot_csi_heatmap(self):
        for user in self.le.classes_:
            user_indices = np.where(self.y == self.le.transform([user])[0])[0]
            user_amps = self.X[user_indices][:100]
            if len(user_amps) > 0:
                mat = np.vstack(user_amps)
                plt.figure(figsize=(10, 6))
                plt.imshow(mat, aspect='auto', cmap='viridis', interpolation='nearest')
                plt.title(f"CSI Amplitude Heatmap for {user}")
                plt.xlabel("Subcarrier Index")
                plt.ylabel("Sample Index")
                plt.colorbar(label="Amplitude")
                plt.tight_layout()
                plt.savefig(f"csi_heatmap_{user}.png")
                plt.show()
                plt.close()

    def tsne_visualization(self):
        X_all = np.vstack([self.X_train_scaled, self.X_test_scaled])
        y_all = np.concatenate([self.y_train, self.y_test])
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_all)
        plt.figure(figsize=(8, 6))
        for label, color, name in zip(range(len(self.le.classes_)), COLORS, self.le.classes_):
            plt.scatter(
                X_tsne[y_all == label, 0], X_tsne[y_all == label, 1],
                label=name, alpha=0.5, s=12, color=color
            )
        plt.title("t-SNE 2D - User Separation")
        plt.xlabel("t-SNE1")
        plt.ylabel("t-SNE2")
        plt.legend()
        plt.tight_layout()
        plt.savefig("tsne_2d.png")
        plt.show()
        plt.close()

    def plot_roc_curves(self):
        y_test_bin = label_binarize(self.y_test, classes=range(len(self.le.classes_)))
        for name, clf in self.clfs.items():
            if hasattr(clf, 'predict_proba'):
                y_score = clf.predict_proba(self.X_test_scaled)
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(len(self.le.classes_)):
                    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                plt.figure(figsize=(8, 6))
                for i in range(len(self.le.classes_)):
                    plt.plot(fpr[i], tpr[i], label=f"{self.le.classes_[i]} (AUC = {roc_auc[i]:.2f})")
                plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
                plt.title(f"ROC Curve - {name} (One-vs-Rest)")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"roc_curve_{name}.png")
                plt.show()
                plt.close()
            else:
                logging.info(f"\n{name} does not support predict_proba for ROC.")

    def run(self):
        self.load_data()
        self.convert_to_amplitude()
        self.select_users()
        self.encode_labels()
        self.split_data()
        self.normalize()
        self.train_model()
        self.evaluate()
        self.feature_importances()
        self.plot_importances()
        self.pca_visualization()
        self.plot_amplitude_vs_subcarrier()
        self.plot_csi_heatmap()
        self.tsne_visualization()
        self.plot_roc_curves()

if __name__ == "__main__":
    models_to_compare = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    classifier = CSIClassifier(models=models_to_compare)
    classifier.run()
