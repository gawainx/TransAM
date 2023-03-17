import os
from typing import Callable
from utils import time_suffix
from logging import Logger as pylogger
from logger import Logger


class StorageMixIn:
    def __init__(self, home_dir, name):
        proj_dir = os.path.join(home_dir, 'output', name)
        if not os.path.exists(proj_dir):
            os.makedirs(proj_dir)
        time_dir = os.path.join(proj_dir, time_suffix())
        if not os.path.exists(time_dir):
            os.makedirs(time_dir)
        self.proj_dir = time_dir
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
        self.ckpt_dir = os.path.join(self.proj_dir, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)


class LoggerMixIn:
    info: Callable
    warning: Callable
    warn: Callable
    debug: Callable
    critical: Callable

    def __init__(self, file: str):
        self.logger = Logger.shared(file)
        self.register_logger(self.logger)

    def register_logger(self, _logger: pylogger):
        self.info = _logger.info
        self.warn = _logger.warning
        self.warning = _logger.warning
        self.debug = _logger.debug
        self.critical = _logger.critical
