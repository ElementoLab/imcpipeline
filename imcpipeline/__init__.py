#! /usr/bin/env python

import sys
import logging
import matplotlib.pyplot as plt


plt.rcParams['svg.fonttype'] = 'none'


def setup_logger(level=logging.INFO):
    logger = logging.getLogger("imcpipeline")
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


LOGGER = setup_logger()


from imcpipeline.data_models import Project, IMCSample, ROI
