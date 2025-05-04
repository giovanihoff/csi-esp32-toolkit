import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
import logging
from scipy.signal import butter, filtfilt

from module.logger import Logger

# Setup logging
Logger(log_file="process_csi.log")


class CSIVerifier:
    """Handles verification of CSI data consistency."""

    @staticmethod
    def verify_channel(df):
        unique_channels = df['channel'].unique()
        if len(unique_channels) > 1:
            raise ValueError(f"Error: More than one channel found in the captures: {unique_channels}")
        logging.info(f"Channel verification OK. Channel used: {unique_channels[0]}")

    @staticmethod
    def verify_rssi(df, min_rssi=-80, max_rssi=-10):
        invalid_rssi = df[(df['rssi'] < min_rssi) | (df['rssi'] > max_rssi)]
        if not invalid_rssi.empty:
            raise ValueError(f"Error: RSSI dBm range out of acceptable range in rows: {invalid_rssi.index.tolist()}")
        logging.info("RSSI dBm range verification OK.")


class CSIProcessor:
    """Handles processing of CSI data."""

    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv

    @staticmethod
    def iq_to_amplitude_phase(iq_data):
        iq_complex = np.array(iq_data[::2]) + 1j * np.array(iq_data[1::2])
        amplitude = np.abs(iq_complex)
        phase = np.unwrap(np.angle(iq_complex))
        return amplitude, phase

    @staticmethod
    def normalize(data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # Avoid division by zero by replacing zero standard deviations with 1
        return (data - mean) / std

    @staticmethod
    def low_pass_filter(data, cutoff=0.1, fs=80, order=4):
        b, a = butter(order, cutoff, btype='low', fs=fs)
        return filtfilt(b, a, data, axis=0)

    def process(self):
        df_raw = pd.read_csv(self.input_csv)

        # Verify channel and RSSI
        CSIVerifier.verify_channel(df_raw)
        CSIVerifier.verify_rssi(df_raw)

        processed_rows = []

        for _, row in df_raw.iterrows():
            raw_iq = json.loads(row['data'])
            amplitude, phase = self.iq_to_amplitude_phase(raw_iq)

            processed_row = {}

            # Amplitude and phase per subcarrier
            for idx, (amp, ph) in enumerate(zip(amplitude, phase), start=1):
                processed_row[f'amplitude_{idx}'] = amp
                processed_row[f'phase_{idx}'] = ph

            processed_rows.append(processed_row)

        df_processed = pd.DataFrame(processed_rows)

        # Add new columns with default value "neutral"
        df_processed['user'] = 'neutral'
        df_processed['position'] = 'neutral'
        df_processed['environment'] = 'neutral'

        # Normalization
        amplitude_cols = [col for col in df_processed.columns if 'amplitude_' in col]
        phase_cols = [col for col in df_processed.columns if 'phase_' in col]

        df_processed[amplitude_cols] = self.normalize(df_processed[amplitude_cols].values)
        df_processed[phase_cols] = self.normalize(df_processed[phase_cols].values)

        # Low-pass filter to remove noise
        df_processed[amplitude_cols] = self.low_pass_filter(df_processed[amplitude_cols].values)
        df_processed[phase_cols] = self.low_pass_filter(df_processed[phase_cols].values)

        df_processed.to_csv(self.output_csv, index=False)
        logging.info(f"Processed CSI data saved to: {self.output_csv}")


def main():
    parser = argparse.ArgumentParser(description='Process CSI Data for Authentication')
    parser.add_argument('-i', '--input', default='data/data_csi.csv', help='Raw input CSV file')
    parser.add_argument('-o', '--output', default='data/processed_csi.csv', help='Processed output CSV file')

    args = parser.parse_args()

    processor = CSIProcessor(args.input, args.output)
    processor.process()


if __name__ == "__main__":
    main()
