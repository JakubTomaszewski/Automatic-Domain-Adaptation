import logging


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Stdout logging handler
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(log_formatter)

    # File logging handler
    file_handler = logging.FileHandler('debug_logs.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    return logger


logger = get_logger()
