# csi_processing.py

import pandas as pd
import numpy as np
import cmath
import argparse
import tempfile
import os
from modules.csi_logger_module import setup_logger
from modules.csi_feature_extraction_module import extract_statistical_features
from filters.hampel_filter import hampel_filter
from filters.low_pass_filter import low_pass_filter

logger = setup_logger(__name__, "processing.log")

# === Arguments ===
parser = argparse.ArgumentParser(description='CSI processor with statistical feature extraction (windowed + filters)')
parser.add_argument('--input', type=str, default='data/csi_data.csv', help='Input CSV file')
parser.add_argument('--output', type=str, default='data/csi_features.csv', help='Output CSV file')
parser.add_argument('--label', type=str, default='neutral', help='Label for the capture')
parser.add_argument('--env', type=str, default='neutral', help='Environment identifier')
parser.add_argument('--position', type=str, default='neutral', help='Position identifier')
args = parser.parse_args()

# === Step 1: Clean malformed lines (odd quotes) ===
logger.info("🧹 Cleaning malformed lines (odd quotes)...")
with open(args.input, 'r', encoding='utf-8') as fin:
    valid_lines = [line for line in fin if line.count('"') % 2 == 0]

with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv') as temp_file:
    temp_file.writelines(valid_lines)
    temp_path = temp_file.name

# === Step 2: Load and process CSI ===
df = pd.read_csv(temp_path)

cols_to_remove = [
    'type', 'seq', 'timestamp', 'taget_seq', 'taget', 'mac', 'rate', 'sig_mode',
    'mcs', 'cwb', 'smoothing', 'not_sounding', 'aggregation', 'stbc', 'fec_coding',
    'sgi', 'noise_floor', 'ampdu_cnt', 'channel', 'channel_primary', 'channel_secondary',
    'local_timestamp', 'ant', 'sig_len', 'rx_state', 'len', 'first_word_invalid'
]
df = df.drop(columns=cols_to_remove, errors='ignore')

def process_csi(row):
    try:
        numbers = list(map(int, row[1:-1].split(',')))[2:]
        amps = []
        for i in range(0, len(numbers), 2):
            c = complex(numbers[i], numbers[i + 1])
            amp, _ = cmath.polar(c)
            amps.append(amp)
        return amps
    except:
        return np.nan

csi_processed = df['data'].apply(process_csi)
csi_valid = csi_processed[csi_processed.apply(lambda x: isinstance(x, list))]
logger.info("🔍 Extracting features from %d valid CSI samples (windowed)...", len(csi_valid))

# === Step 3: Windowing and Feature Extraction ===
features_list = []
window_size = 50
samples = csi_valid.tolist()

for i in range(0, len(samples) - window_size + 1, window_size):
    try:
        window = samples[i:i + window_size]
        amp_matrix = np.vstack(window)
        mean_amplitudes = amp_matrix.mean(axis=0)

        # Convert to DataFrame for compatibility com filtros
        df_amp = pd.DataFrame({'amplitude': mean_amplitudes})

        # === Apply Hampel Filter ===
        df_amp['amplitude'] = hampel_filter(df_amp['amplitude'])

        # === Apply Low-Pass Filter ===
        df_amp['amplitude'] = low_pass_filter(df_amp['amplitude'])

        filtered = df_amp['amplitude'].values

        features = extract_statistical_features(filtered)
        features["label"] = args.label
        features["environment"] = args.env
        features["position"] = args.position
        features_list.append(features)
    except Exception as e:
        logger.warning("⚠️ Feature extraction failed for window %d: %s", i, str(e))

# === Step 4: Save final features ===
final_df = pd.DataFrame(features_list)
final_df.to_csv(args.output, index=False)
logger.info(f"✅ Processing completed and saved to: {args.output}")

# === Cleanup ===
os.remove(temp_path)

