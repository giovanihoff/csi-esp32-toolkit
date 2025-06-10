import os
import logging
from capture_csi import CSICapture, SerialPortManager
from process_csi import CSIProcessor
from auth_csi import AuthCSI
import pandas as pd
from auth_csi import DatasetHandler

class CSIPipeline:
    """Manages the CSI capture, processing, and authentication pipeline."""

    def __init__(self, port="/dev/ttyUSB0", lines=None, time=None, output_raw="data/data_csi.csv", user=None,
                 position="unknown", environment="unknown", output_processed="data/processed_csi.csv",
                 threshold=0.6, output_auth="output/auth_results.csv", dataset_path="dataset/dataset.csv"):
        self.port = port
        self.lines = lines
        self.time = time
        self.output_raw = output_raw
        self.user = user
        self.position = position
        self.environment = environment
        self.output_processed = output_processed
        self.threshold = threshold
        self.output_auth = output_auth
        self.dataset_path = dataset_path

    def capture_data(self):
        """Capture CSI data using CSICapture."""
        if not self.user:
            raise ValueError("❌ User parameter is required for capturing data.")

        logging.info("📡 [CAPTURE_CSI] Initializing CSI data capture...")
        try:
            serial_manager = SerialPortManager(self.port)

            if not serial_manager.check_port():
                logging.error(f"❌ [CAPTURE_CSI] The port {self.port} is not accessible or no device is connected.")
                raise ConnectionError(f"Port {self.port} is not accessible.")

            capture = CSICapture(
                serial_port=serial_manager,
                output_file=self.output_raw,
                max_lines=self.lines,
                max_time=self.time,
                user=self.user,
                position=self.position,
                environment=self.environment
            )
            capture.start_capture()
            logging.info(f"✅ [CAPTURE_CSI] CSI data capture completed. Data saved to: {self.output_raw}")
        except Exception as e:
            logging.error(f"❌ [CAPTURE_CSI] An error occurred during CSI data capture: {e}")
            raise  # Re-raise the exception to stop the pipeline execution

    def process_data(self):
        """Process CSI data using CSIProcessor."""
        logging.info("🔄 [PROCESS_CSI] Starting CSI data processing...")
        try:
            processor = CSIProcessor(input_csv=self.output_raw, output_csv=self.output_processed)
            processor.process()
            logging.info(f"✅ [PROCESS_CSI] CSI data processing completed. Data saved to: {self.output_processed}")
        except Exception as e:
            logging.error(f"❌ [PROCESS_CSI] An error occurred during CSI data processing: {e}")
            raise  # Re-raise the exception to stop the pipeline execution

    def authenticate_user(self):
        """Authenticate the user using AuthCSI."""
        logging.info("🔒 [AUTH_CSI] Starting CSI authentication...")

        auth_csi = AuthCSI(
            input_path=self.output_processed,
            user=self.user,
            threshold=self.threshold,
            output_path=self.output_auth,
            dataset_path=self.dataset_path
        )
        try:
            auth_csi.run()
            logging.info(f"✅ [AUTH_CSI] CSI authentication completed. Results saved to: {self.output_auth}")
        except Exception as e:
            logging.error(f"❌ [AUTH_CSI] An error occurred during CSI authentication: {e}")
            raise  # Re-raise the exception to stop the pipeline execution

    def run_pipeline(self):
        """Run the full pipeline: capture, process, and authenticate CSI data."""
        logging.info("🚀 Starting CSI Pipeline...")
        try:
            self.capture_data()
            self.process_data()
            self.authenticate_user()
            logging.info("🎉 CSI Pipeline completed successfully!")
        except Exception as e:
            logging.error(f"❌ [PIPELINE] Pipeline execution stopped due to an error: {e}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CSI Capture, Processing, and Authentication Pipeline")
    parser.add_argument("-l", "--lines", type=int, help="Number of valid lines to capture (overrides time if provided)")
    parser.add_argument("-t", "--time", type=int, default=10, help="Capture duration in seconds (default: 10 seconds)")
    parser.add_argument("-u", "--user", required=True, help="User associated with the capture")
    parser.add_argument("-p", "--position", default="unknown", help="Position associated with the capture (default: unknown)")
    parser.add_argument("-e", "--environment", default="unknown", help="Environment associated with the capture (default: unknown)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    pipeline = CSIPipeline(
        lines=args.lines,
        time=args.time,
        user=args.user,
        position=args.position,
        environment=args.environment
    )
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()