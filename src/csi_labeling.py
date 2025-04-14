# csi_labeling.py

import pandas as pd
import argparse
import os
from modules.csi_logger_module import setup_logger

logger = setup_logger(__name__, "labeling.log")

parser = argparse.ArgumentParser(description="Label CSI feature data for supervised training")
parser.add_argument("--label", type=str, required=True, help="Label to assign to the data (e.g., 'giovani', 'empty')")
parser.add_argument("--environment", type=str, required=True, help="Environment name (e.g., 'env1', 'lab', 'room1')")
parser.add_argument("--position", type=str, required=True, help="User position (e.g., 'sitting', 'standing')")
parser.add_argument("--input", type=str, default="data/csi_features.csv", help="Path to input features CSV")
parser.add_argument("--output", type=str, default="dataset/dataset.csv", help="Path to labeled dataset CSV")
parser.add_argument("--rebuild", action="store_true", help="Overwrite the existing dataset")
parser.add_argument("--delete", action="store_true", help="Remove samples with this label and environment from the dataset")
args = parser.parse_args()

# Ensure dataset directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Load existing dataset if present and not rebuilding
if os.path.exists(args.output) and not args.rebuild:
    df_dataset = pd.read_csv(args.output)
else:
    df_dataset = pd.DataFrame()

if args.delete:
    if not df_dataset.empty:
        condition = (df_dataset["label"] == args.label) & (df_dataset["environment"] == args.environment)
        df_dataset = df_dataset[~condition]
        logger.info("🗑️ Removed label '%s' from environment '%s'", args.label, args.environment)
    else:
        logger.warning("⚠️ Dataset is empty. Nothing to delete.")
else:
    df_input = pd.read_csv(args.input)
    df_input["label"] = args.label
    df_input["environment"] = args.environment
    df_input["position"] = args.position
    df_dataset = pd.concat([df_dataset, df_input], ignore_index=True)
    logger.info("🏷️ Added label '%s' in environment '%s' with position '%s' to dataset", args.label, args.environment, args.position)

# Save dataset
df_dataset.to_csv(args.output, index=False)
logger.info("✅ Dataset saved to: %s", args.output)

