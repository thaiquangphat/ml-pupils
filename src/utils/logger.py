import logging
import os
from datetime import datetime

def get_logger(model_name):
    os.makedirs(f"results/log/{model_name}", exist_ok=True)

    timestamp = datetime.now().strftime("%d%m%Y_%H-%M-%S")
    log_filename = f"results/log/{model_name}/{model_name}-{timestamp}.txt"

    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO) 

    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger