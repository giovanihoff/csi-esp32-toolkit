# modules/csi_logger_module.py

import os
import logging

def setup_logger(name: str, log_file: str) -> logging.Logger:
    os.makedirs("log", exist_ok=True)
    log_path = os.path.join("log", log_file)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Evitar múltiplos handlers se já estiver configurado
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(message)s")

        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger

