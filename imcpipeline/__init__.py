import sys
import logging


def setup_logger(level=logging.DEBUG):
    logger = logging.getLogger("imcpipeline")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()

from imcpipeline.data_models import Project, IMCSample, ROI


# TODO: solve passing files with spaces to ilastik predict
