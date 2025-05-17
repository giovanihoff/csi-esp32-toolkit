import csv
import os
import sys
import argparse
import time
import logging
import serial

from module.logger import Logger

# Setup logging
Logger(log_file="capture_csi.log")

SLEEP = 6
BAUDRATE_UART = 921600
TIMEOUT_UART = 1
HEADER = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
          "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
          "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp",
          "ant", "sig_len", "rx_state", "len", "first_word", "data"]


class SerialPortManager:
    """Manages the serial port connection."""

    def __init__(self, port, baudrate, timeout):
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

    def __init__(self, serial_port, output_file, max_lines=None, max_time=None):
        self.serial_port = serial_port
        self.output_file = output_file
        self.max_lines = max_lines
        self.max_time = max_time
        self.captured_data_count = 0
        self.discarded_data_count = 0
        self.previous_id = None
        self.reset_attempted = False

    def validate_id(self, current_id):
        """Validate sequential ID."""
        if self.previous_id is not None and current_id != self.previous_id + 2:
            if self.reset_attempted:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.error(f"❌ Second non-sequential ID detected. Previous ID: {self.previous_id}, Current ID: {current_id}. Terminating capture...")
                raise ValueError("Second non-sequential ID detected.")
            else:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.warning(f"⚠️ Non-sequential ID detected. Previous ID: {self.previous_id}, Current ID: {current_id}. Resetting capture...")
                self.reset_attempted = True
                self.reset_capture()
                return False
        self.previous_id = current_id
        return True

    def reset_capture(self):
        """Reset the capture process."""
        self.captured_data_count = 0
        self.discarded_data_count = 0
        self.previous_id = None

    def process_line(self, line, csv_writer):
        """Process a single line of CSI data."""
        fields = line.split(",", 24)
        if len(fields) != 25:
            logging.error("❌ Incomplete CSI data received. Skipping line...")
            self.discarded_data_count += 1
            return

        current_id = int(fields[1])
        if not self.validate_id(current_id):
            return

        raw_data = fields[24].strip('[]"').replace(' ', '')
        if len(raw_data.split(",")) != 128:
            self.discarded_data_count += 1
            return

        formatted_data = f'[{raw_data}]'
        csv_writer.writerow(["CSI_DATA"] + fields[1:24] + [formatted_data])
        self.captured_data_count += 1

    def start_capture(self):
        """Start capturing CSI data."""
        with self.serial_port.open_port() as ser, open(self.output_file, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(HEADER)

            logging.info("⏳ Waiting to initialize...")
            time.sleep(SLEEP)

            logging.info(f"📡 Starting CSI data capture on port {self.serial_port.port}. Saving to: {self.output_file}")
            logging.info("▶️ Capture started. Press Ctrl+C to stop manually.")

            start_time = time.time()
            end_time = start_time + self.max_time if self.max_time else None

            try:
                while True:
                    line = ser.readline().decode('utf-8').strip()
                    if line.startswith("CSI_DATA"):
                        self.process_line(line, csv_writer)

                    if self.max_lines and self.captured_data_count >= self.max_lines:
                        break
                    if not self.max_lines and end_time and time.time() >= end_time:
                        break

                    elapsed_time = int(time.time() - start_time)
                    sys.stdout.write(f"\r⏱️ Elapsed time: {elapsed_time}s | Captured data: {self.captured_data_count} | Discarded data: {self.discarded_data_count}")
                    sys.stdout.flush()

            except KeyboardInterrupt:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.warning("⏹️ Capture interrupted by the user.")
            except ValueError as e:
                sys.stdout.write("\n")
                sys.stdout.flush()
                logging.error(f"❌ {str(e)}")
                csvfile.close()
                os.remove(self.output_file)
                sys.exit(1)

            # Log the summary after the loop ends
            sys.stdout.write("\n")  # Ensure the terminal line is cleared
            sys.stdout.flush()
            elapsed_time = int(time.time() - start_time)
            logging.info(f"✅ Capture completed. Total elapsed time: {elapsed_time}s | Total captured data: {self.captured_data_count}")


def main():
    parser = argparse.ArgumentParser(description="Capture CSI data.")
    parser.add_argument("-p", "--port", default="/dev/ttyUSB0", help="Serial port (default: /dev/ttyUSB0)")
    parser.add_argument("-t", "--time", type=int, default=10, help="Capture duration in seconds (default: 10 seconds)")
    parser.add_argument("-o", "--output", default="data/data_csi.csv", help="Output CSV file name (default: data/data_csi.csv)")
    parser.add_argument("-l", "--lines", type=int, help="Number of valid lines to capture (overrides time if provided)")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        logging.info(f"📂 Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    serial_manager = SerialPortManager(args.port, BAUDRATE_UART, TIMEOUT_UART)
    if not serial_manager.check_port():
        logging.error(f"❌ The port {args.port} is not accessible or no device is connected.")
        sys.exit(1)

    capture = CSICapture(serial_manager, args.output, max_lines=args.lines, max_time=args.time)
    capture.start_capture()


if __name__ == "__main__":
    main()
