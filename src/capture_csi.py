import csv
import os
import sys
import argparse
import time
import logging
import serial
import uuid

from module.logger import Logger

"""
CSI Capture Script

This script captures Channel State Information (CSI) data from an ESP32 device connected via UART.

Key functionalities:
- Validates captured data integrity by ensuring sequential IDs.
- Logs raw data received directly from UART into a CSV file.
- Includes metadata such as user, position, and environment for each capture.
- Controls capture duration or maximum number of lines captured.

The output is a CSV file containing raw CSI data, ready for subsequent processing.
"""

# Setup logging
Logger(log_file="capture_csi.log")

SAMPLE_FILE = "data/sample_csi.csv"
LEN_DATA = 384
INCREMENT = 1
SLEEP = 6
BAUDRATE_UART = 921600
TIMEOUT_UART = 1
HEADER = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
          "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
          "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
          "ant", "sig_len", "rx_state", "len", "first_word", "data"]


class SerialPortManager:
    """Manages the serial port connection."""

    def __init__(self, port, baudrate=BAUDRATE_UART, timeout=TIMEOUT_UART):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

    def check_port(self):
        """Check if the serial port is accessible."""
        try:
            with serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout) as ser:
                pass
            return True
        except serial.SerialException:
            return False

    def open_port(self):
        """Open the serial port."""
        return serial.Serial(port=self.port, baudrate=self.baudrate, timeout=self.timeout, bytesize=8, parity='N', stopbits=1)


class CSICapture:
    """Handles CSI data capture."""

    def __init__(self, serial_port, output_file, max_lines=None, max_time=None, user="unknown", position="unknown", environment="unknown"):
        self.serial_port = serial_port
        self.output_file = output_file
        self.max_lines = max_lines
        self.max_time = max_time
        self.user = user
        self.position = position
        self.environment = environment
        self.captured_data_count = 0
        self.discarded_data_count = 0
        self.previous_id = None
        self.reset_attempted = False
        self.first_id = None
        self.start_time = None
        self.end_time = None
        self.capture_id = str(uuid.uuid4()) # Generate ONE unique capture_id for ALL rows

    def process_line(self, line):
        """Process a single line of CSI data."""
        fields = line.split(",", 24)
        if len(fields) != 25:
            logging.error("❌ Incomplete CSI data received. Skipping line...")
            self.discarded_data_count += 1
            return

        current_id = int(fields[1])

        if self.first_id is None:
            self.first_id = current_id
        elif current_id != self.previous_id + INCREMENT:
            sys.stdout.write("\n")
            sys.stdout.flush()
            logging.warning(f"⚠️  Non-sequential ID detected at ID {current_id}. Discarding previous samples and restarting capture window.")
            self.sample_file.seek(0)
            self.sample_file.truncate()
            self.sample_writer.writerow(HEADER + ["user", "position", "environment", "capture_id"])
            self.captured_data_count = 0
            self.start_time = time.time()
            self.end_time = self.start_time + self.max_time if self.max_time else None

        self.previous_id = current_id

        raw_data = fields[24].strip('[]"').replace(' ', '')
        if len(raw_data.split(",")) != LEN_DATA:
            self.discarded_data_count += 1
            return

        formatted_data = f'[{raw_data}]'
        self.sample_writer.writerow(["CSI_DATA"] + fields[1:24] + [formatted_data, self.user, self.position, self.environment, self.capture_id])        
        self.captured_data_count += 1

    def start_capture(self):

        cumulative_path = self.output_file
        is_new_file = not os.path.exists(cumulative_path)

        self.sample_file = open(SAMPLE_FILE, mode='w', newline='')
        self.sample_writer = csv.writer(self.sample_file)
        self.sample_writer.writerow(HEADER + ["user", "position", "environment", "capture_id"])

        self.cumulative_file = open(cumulative_path, mode='a', newline='')
        self.cumulative_writer = csv.writer(self.cumulative_file)
        if is_new_file:
            self.cumulative_writer.writerow(HEADER + ["user", "position", "environment", "capture_id"])
        """Start capturing CSI data."""
        with self.serial_port.open_port() as ser:
            logging.info("⏳ Waiting to initialize...")
            time.sleep(SLEEP)

            logging.info(f"📡 Starting CSI data capture on port {self.serial_port.port}. Saving to: {self.output_file}")
            logging.info("▶️  Capture started. Press Ctrl+C to stop manually.")

            self.start_time = time.time()
            self.end_time = self.start_time + self.max_time if self.max_time else None

            try:
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("CSI_DATA"):
                        self.process_line(line)

                    if self.max_lines and self.captured_data_count >= self.max_lines:
                        break
                    if not self.max_lines and self.end_time and time.time() >= self.end_time:
                        break

                    elapsed_time = int(time.time() - self.start_time)
                    sys.stdout.write(f"\r⏱️ Elapsed time: {elapsed_time}s | Captured data: {self.captured_data_count} | Discarded data: {self.discarded_data_count}")
                    sys.stdout.flush()

            except KeyboardInterrupt:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.warning("⏹️ Capture interrupted by the user.")
                # Discard current capture: do not write anything to cumulative file
                self.sample_file.close()
                self.cumulative_file.close()
                # Optionally, remove the sample file to avoid confusion
                if os.path.exists(SAMPLE_FILE):
                    os.remove(SAMPLE_FILE)
                return
            except ValueError as e:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.error(f"❌ {str(e)}")
                # Do not remove the cumulative file!
                self.sample_file.close()
                self.cumulative_file.close()
                if os.path.exists(SAMPLE_FILE):
                    os.remove(SAMPLE_FILE)
                sys.exit(1)

            # Log the summary after the loop ends
            sys.stdout.write("\n")  # Ensure the terminal line is cleared
            sys.stdout.flush()
            elapsed_time = int(time.time() - self.start_time)
            if self.captured_data_count > 0:
                self.sample_file.close()
                # Ask the operator if they want to include the capture in the cumulative file
                try:
                    user_input = input("\nDo you want to include this capture in the cumulative file (data/data_csi.csv)? [Y/n]: ").strip().lower()
                except EOFError:
                    user_input = ''  # Default to include if input is not possible
                if user_input in ('', 'y', 'yes'):
                    with open("data/sample_csi.csv", "r", newline="") as sample:
                        reader = csv.reader(sample)
                        next(reader)
                        for row in reader:
                            self.cumulative_writer.writerow(row)
                        logging.info("📥 Current capture data added to the cumulative file.")
                else:
                    logging.info("ℹ️ Capture NOT added to the cumulative file. Data is available in data/sample_csi.csv.")
            logging.info(f"✅ Capture completed. Total elapsed time: {elapsed_time}s | Total captured data: {self.captured_data_count}")
            self.sample_file.close()
            self.cumulative_file.close()


def main():
    parser = argparse.ArgumentParser(description="Capture CSI data.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("-t", "--time", type=int, default=10, help="Capture duration in seconds (default: 10 seconds)")
    parser.add_argument("-o", "--output", default="data/data_csi.csv", help="Output CSV file name (default: data/data_csi.csv)")
    parser.add_argument("-l", "--lines", type=int, help="Number of valid lines to capture (overrides time if provided)")
    parser.add_argument("-u", "--user", required=True, help="User associated with the capture")
    parser.add_argument("-p", "--position", default="unknown", help="Position associated with the capture (default: unknown)")
    parser.add_argument("-e", "--environment", default="unknown", help="Environment associated with the capture (default: unknown)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        logging.info(f"📂 Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    serial_manager = SerialPortManager(args.port, BAUDRATE_UART, TIMEOUT_UART)
    if not serial_manager.check_port():
        logging.error(f"❌ The port {args.port} is not accessible or no device is connected.")
        sys.exit(1)

    capture = CSICapture(serial_manager, args.output, max_lines=args.lines, max_time=args.time, user=args.user, position=args.position, environment=args.environment)
    capture.start_capture()


if __name__ == "__main__":
    main()
    