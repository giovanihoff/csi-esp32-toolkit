# csi_config_module.py

import argparse
import os

parser = argparse.ArgumentParser(description="CSI capture via ESP32")
parser.add_argument("--duration", type=int, default=15, help="Capture duration in seconds (default: 10)")
parser.add_argument("--no-traffic", action="store_true", help="Passive mode (no ping)")
args = parser.parse_args()

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = "921600"
ESP_IDF_PATH = "/home/hoff/esp/esp-idf"
USE_TRAFFIC = not args.no_traffic
CAPTURE_DURATION = args.duration
CSI_SUBCARRIERS = 52

os.makedirs("data", exist_ok=True)
OUTPUT_FILE = os.path.join("data", "csi_data.csv")
