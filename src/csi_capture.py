# csi_capture.py

import os
import sys
import re
import time
import signal
import logging
import numpy as np
import subprocess

from modules.csi_config_module import (
    SERIAL_PORT, BAUD_RATE, OUTPUT_FILE, ESP_IDF_PATH,
    USE_TRAFFIC, CAPTURE_DURATION, CSI_SUBCARRIERS, args
)
from modules.csi_monitor_module import start_serial_monitor, start_ping
from modules.csi_utils_module import validate_csi, calculate_magnitude, save_csv_data

# Logging setup
from modules.csi_logger_module import setup_logger
logger = setup_logger(__name__, "capture.log")

def capture_csi():
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    process = start_serial_monitor(ESP_IDF_PATH, SERIAL_PORT, BAUD_RATE)
    ip_address, captured_data = None, []
    rssi_values = []
    start_time = None
    ping_process = None

    os.system("reset")
    logger.info("🔍 Waiting for ESP32 IP...")

    try:
        while True:
            line = process.stdout.readline().strip()

            if ip_address is None:
                match_ip = re.search(r'got ip:([\d.]+)', line)
                if match_ip:
                    ip_address = match_ip.group(1)
                    logger.info(f"📡 IP detected: {ip_address}")
                continue

            if line.startswith("CSI_DATA"):
                if start_time is None:
                    start_time = time.time()
                    logger.info("🚀 CSI_DATA detected. Starting capture...")
                    if USE_TRAFFIC:
                        ping_process = start_ping(ip_address, CAPTURE_DURATION)
                        logger.info(f"🔁 Ping started in parallel to {ip_address}")
                    else:
                        logger.info("📴 No traffic (passive mode).")

                fields = line.split(",", 24)
                if len(fields) > 24:
                    try:
                        rssi = int(fields[3])
                        local_timestamp = fields[18]
                        raw_csi = fields[24].strip('[]"').split(",")

                        if all(re.match(r'^-?\d+$', x.strip()) for x in raw_csi):
                            csi_data = [int(x.strip()) for x in raw_csi]

                            if -80 <= rssi <= -30:
                                rssi_values.append(rssi)
                                magnitude_csi = calculate_magnitude(csi_data[4:])
                                if np.mean(magnitude_csi) > 0:
                                    captured_data.append([local_timestamp] + csi_data)
                        else:
                            logger.warning("⚠️ Corrupted or invalid CSI data discarded.")
                    except ValueError as e:
                        logger.warning(f"⚠️ Error converting CSI to integer: {e}")
                        continue

                elapsed_time = time.time() - start_time
                if elapsed_time >= CAPTURE_DURATION:
                    os.system("reset")
                    break
                else:
                    print(f"⏳ Capturing... {int(elapsed_time)}s / {CAPTURE_DURATION}s", end="\r")

        if ping_process:
            stdout, _ = ping_process.communicate()
            match_loss = re.search(r'([\d.]+)% packet loss', stdout)
            if match_loss:
                perda = float(match_loss.group(1))
                logger.info(f"📉 Packet loss rate: {perda:.2f}%")
                if perda > 0.0:
                    logger.error("❌ High packet loss, aborting execution.")
                    sys.exit(1)
            else:
                logger.warning("⚠️ Failed to determine packet loss rate.")
                sys.exit(1)

        if captured_data:
            rssi_medio = np.mean(rssi_values) if rssi_values else 0
            logger.info(f"📡 Average RSSI: {rssi_medio:.2f} dBm")
            sample = captured_data[0]
            logger.info(f"🧩 CSI fields per sample (excluding timestamp): {len(sample) - 1}")
        else:
            logger.warning("⚠️ No valid sample was captured.")

        save_csv_data(captured_data, OUTPUT_FILE)
        duracao_total = time.time() - start_time
        logger.info(f"✅ Capture finished in {duracao_total:.2f} segundos.")
        logger.info(f"📊 Total captured samples: {len(captured_data)}")
        logger.info(f"📁 Data saved to '{OUTPUT_FILE}'.")

    except KeyboardInterrupt:
        os.system("reset")
        logger.info("🛑 Capture manually interrupted.")
        sys.exit(1)
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)  # espera até 10 segundos após o fim da captura
        except subprocess.TimeoutExpired:
            process.terminate()


if __name__ == "__main__":
    capture_csi()

