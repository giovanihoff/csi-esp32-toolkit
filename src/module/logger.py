import logging
import os
import sys

class Logger:
    """Handles logging configuration for the application."""

    def __init__(self, log_dir="log", log_file="csi.log", level=logging.INFO):
        self.log_dir = log_dir
        self.log_file = log_file
        self.level = level
        self._setup_logger()

    def _setup_logger(self):
        """Setup logging to log to both file and console."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        log_file_path = os.path.join(self.log_dir, self.log_file)
        logging.basicConfig(
            level=self.level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='w'), # Log to file (overwrite on each execution)
                logging.StreamHandler(sys.stdout) # Log to console
            ]
        )
