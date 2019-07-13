import logging


def get_logger(filename):
    logger = logging.getLogger(__name__)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=filename, mode="w")
    logger.setLevel(logging.DEBUG)
    handler1.setLevel(logging.DEBUG)
    handler2.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


