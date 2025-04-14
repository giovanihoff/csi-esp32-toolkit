# csi_utils_module.py

import numpy as np
import csv
import os
import logging

def validate_csi(data, csi_subcarriers):
    return len(data) == (csi_subcarriers * 2) + 4

def calculate_magnitude(data):
    complex_data = np.array(data[::2]) + 1j * np.array(data[1::2])
    return np.abs(complex_data)

def save_csv_data(data, output_file):
    """
    Saves captured CSI data to CSV in a format compatible with the CSI processing pipeline.
    Each row will be stored as a stringified list in a single 'data' column.
    """
    if data:
        with open(output_file, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["data"])  # single column header for processing compatibility
            for row in data:
                csi_string = "[" + ",".join(map(str, row[1:])) + "]"
                writer.writerow([csi_string])
    else:
        raise RuntimeError("No CSI data captured.")

