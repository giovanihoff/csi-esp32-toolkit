import pandas as pd
import numpy as np
import json
import argparse
import logging
import uuid 
from scipy.signal import butter, filtfilt, medfilt
from scipy.stats import skew, kurtosis

from module.logger import Logger

# Setup logging
Logger(log_file="process_csi.log")


class CSIVerifier:
    """Handles verification of CSI data consistency."""

    @staticmethod
    def verify_channel(df):
        unique_channels = df['channel'].unique()
        if len(unique_channels) > 1:
            raise ValueError(f"❌ Error: More than one channel found in the captures: {unique_channels}")
        logging.info(f"✅ Channel verification OK. Channel used: {unique_channels[0]}")

    @staticmethod
    def verify_rssi(df, min_rssi=-80, max_rssi=-10):
        invalid_rssi = df[(df['rssi'] < min_rssi) | (df['rssi'] > max_rssi)]
        if not invalid_rssi.empty:
            raise ValueError(f"❌ Error: RSSI dBm range out of acceptable range in rows: {invalid_rssi.index.tolist()}")
        logging.info("✅ RSSI dBm range verification OK.")


class CSIProcessor:
    """Handles processing of CSI data."""

    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv

    @staticmethod
    def iq_to_amplitude_phase(iq_data):
        iq_data = iq_data[4:108]  # Remove header/meta fields and limit to 52 subcarriers (104 IQ values)
        iq_complex = np.array(iq_data[::2]) + 1j * np.array(iq_data[1::2])
        amplitude = np.abs(iq_complex)
        phase = np.unwrap(np.angle(iq_complex))
        return amplitude, phase

    @staticmethod
    def normalize(data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (data - mean) / std

    @staticmethod
    def low_pass_filter(data, cutoff=0.1, fs=80, order=4):
        b, a = butter(order, cutoff, btype='low', fs=fs)
        return filtfilt(b, a, data, axis=0)

    @staticmethod
    def hampel_filter(data, window_size=5, n_sigmas=3):
        filtered_data = data.copy()
        for i in range(data.shape[1]):
            series = data[:, i]
            med = medfilt(series, kernel_size=window_size)
            diff = np.abs(series - med)
            mad = np.median(diff)
            threshold = n_sigmas * 1.4826 * mad
            outlier_idx = diff > threshold
            series[outlier_idx] = med[outlier_idx]
            filtered_data[:, i] = series
        return filtered_data

    def process(self):
        df_raw = pd.read_csv(self.input_csv)

        # Verify channel and RSSI
        logging.info("🔍 Verifying channel and RSSI...")
        CSIVerifier.verify_channel(df_raw)
        CSIVerifier.verify_rssi(df_raw)

        processed_rows = []

        # Valid subcarrier positions (excluding pilots: 6, 20, 25, 39)
        valid_positions = [i for i in range(52) if i not in [6, 20, 25, 39]]

        for _, row in df_raw.iterrows():
            raw_iq = json.loads(row['data'])
            amplitude, phase = self.iq_to_amplitude_phase(raw_iq)

            processed_row = {}
            for i in valid_positions:
                processed_row[f'amplitude_{i+1}'] = amplitude[i]
                processed_row[f'phase_{i+1}'] = phase[i]

            # Include additional columns from the raw CSV
            if 'local_timestamp' in row:
                processed_row['local_timestamp'] = row['local_timestamp']
            if 'user' in row:
                processed_row['user'] = row['user']
            if 'position' in row:
                processed_row['position'] = row['position']
            if 'environment' in row:
                processed_row['environment'] = row['environment']

            processed_rows.append(processed_row)

        df_processed = pd.DataFrame(processed_rows)

        # Set weight to 1.0 for all rows
        df_processed['weight'] = 1.0

        # Generate ONE unique capture_id for ALL rows
        capture_id = str(uuid.uuid4())
        df_processed['capture_id'] = capture_id

        amplitude_cols = [col for col in df_processed.columns if col.startswith("amplitude_")]
        phase_cols = [col for col in df_processed.columns if col.startswith("phase_")]

        # Normalize
        logging.info("⚙️ Normalizing data...")
        df_processed[amplitude_cols] = self.normalize(df_processed[amplitude_cols].values)
        df_processed[phase_cols] = self.normalize(df_processed[phase_cols].values)

        # Hampel filter (remove outliers)
        logging.info("🛠️ Applying Hampel filter to remove outliers...")
        df_processed[amplitude_cols] = self.hampel_filter(df_processed[amplitude_cols].values)
        df_processed[phase_cols] = self.hampel_filter(df_processed[phase_cols].values)

        # Low-pass filter
        logging.info("🔽 Applying low-pass filter...")
        df_processed[amplitude_cols] = self.low_pass_filter(df_processed[amplitude_cols].values)
        df_processed[phase_cols] = self.low_pass_filter(df_processed[phase_cols].values)

        # Compute additional statistical features
        for col_prefix in ['amplitude', 'phase']:
            cols = [col for col in df_processed.columns if col.startswith(col_prefix)]
            df_processed[f'{col_prefix}_var'] = np.var(df_processed[cols].values, axis=1)
            df_processed[f'{col_prefix}_skew'] = skew(df_processed[cols].values, axis=1)
            df_processed[f'{col_prefix}_kurtosis'] = kurtosis(df_processed[cols].values, axis=1)

        # Save processed data to CSV
        df_processed.to_csv(self.output_csv, index=False)
        logging.info(f"✅ Processed CSI data saved to: {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Process CSI Data for Authentication')
    parser.add_argument('-i', '--input', default='data/data_csi.csv', help='Raw input CSV file')
    parser.add_argument('-o', '--output', default='data/processed_csi.csv', help='Processed output CSV file')

    args = parser.parse_args()

    processor = CSIProcessor(args.input, args.output)
    processor.process()


if __name__ == "__main__":
    main()
