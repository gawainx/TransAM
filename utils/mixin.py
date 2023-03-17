import os
from datetime import datetime
from os import path as op
import torch
import logging
from rich.logging import RichHandler
from rich.console import Console
from typing import Callable
from torch.utils.tensorboard import SummaryWriter
from logging import Logger as InternalLogger
import threading
import numpy as np
from metrics import KGLPMetrics

NELL_FILE = '/dev/null'

inner_logger = Console().log


def logger(filename: str = NELL_FILE):
    FORMAT = "[line %(lineno)d] %(asctime)s %(levelname)s: %(message)s"
    inner_logger(f"Will Log in {filename}")
    logging.basicConfig(
            level="NOTSET",
            format=FORMAT,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=filename
    )
    log = logging.getLogger("rich")
    log.addHandler(RichHandler())
    return log


class Logger:
    _shared_lock_ = threading.Lock()

    def __init__(self, filename: str = NELL_FILE):
        FORMAT = "[%(filename)s] %(asctime)s %(levelname)s: %(message)s"
        inner_logger(f"Will Log in {filename}")
        logging.basicConfig(
                level="NOTSET",
                format=FORMAT,
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=NELL_FILE
        )
        log = logging.getLogger("rich")
        log.addHandler(RichHandler())
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(logging.Formatter(FORMAT))
        file_handler.setLevel(logging.WARN)
        log.addHandler(file_handler)
        self.logger = log

    @classmethod
    def shared(cls, *args, **kwargs):
        if not hasattr(Logger, "__shared__"):
            with Logger._shared_lock_:
                if not hasattr(Logger, "__shared__"):
                    Logger.__shared__ = Logger(*args, **kwargs)
        return Logger.__shared__.logger


class LoggerMixIn:
    info: Callable
    warning: Callable
    warn: Callable
    debug: Callable
    critical: Callable

    def __init__(self, file: str):
        self.logger = Logger.shared(file)
        self.register_logger(self.logger)

    def register_logger(self, _logger: InternalLogger):
        self.info = _logger.info
        self.warn = _logger.warning
        self.warning = _logger.warning
        self.debug = _logger.debug
        self.critical = _logger.critical


def time_suffix(last: bool = False):
    if last:
        return str(datetime.now()).split(' ')[-1].replace(':', '').replace('.', '_')
    return str(datetime.now()).split(' ')[0]


class StorageMixIn:
    """Log + Tensorboard + saved-load"""
    info: Callable
    warning: Callable
    warn: Callable
    debug: Callable
    critical: Callable

    def __init__(self, home_dir, name, on_train: bool = True, debug: bool = False):
        proj_dir = os.path.join(home_dir, 'output', name)
        if not os.path.exists(proj_dir):
            os.makedirs(proj_dir)
        time_dir = os.path.join(proj_dir, time_suffix())
        if not os.path.exists(time_dir):
            os.makedirs(time_dir)
        self.proj_dir = time_dir
        if not os.path.exists(self.proj_dir):
            os.makedirs(self.proj_dir)
        self.ckpt_dir = os.path.join(self.proj_dir, 'save_models')
        self.writer = SummaryWriter(self.proj_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.cur_epoch = -1
        self.model = None
        self.optimizer = None
        self.saved_models = {}
        if debug:
            self.logger = Logger.shared('/dev/null')
        elif on_train:
            self.logger = Logger.shared(op.join(self.proj_dir, 'training.log'))
        else:
            self.logger = Logger.shared(op.join(self.proj_dir, 'test.log'))
        self.register_logger(self.logger)

    def register_logger(self, _logger: InternalLogger):
        self.info = _logger.info
        self.warn = _logger.warning
        self.warning = _logger.warning
        self.debug = _logger.debug
        self.critical = _logger.critical

    def save_model(self, suffix: str = 'normal'):
        fn = os.path.join(self.ckpt_dir,
                          f"EPO{self.cur_epoch}_{suffix}.ckpt")
        result = {
            'state_dict': self.model.state_dict(),
            'optim':      self.optimizer.state_dict()
        }
        if os.path.exists(fn):
            fn = os.path.join(self.ckpt_dir,
                              f"EPO{self.cur_epoch}_{suffix}_{time_suffix(True)}.ckpt")
        torch.save(result, fn)
        return fn

    def save_latest(self):
        return self.save_model('latest')

    def load_in_fit(self, epo):
        if epo not in self.saved_models.keys():
            self.warn(f"NO RELATED CKPT FOR {epo}")
            return
        d = torch.load(self.saved_models[epo]['ckpt'])
        self.model.load_state_dict(d['state_dict'])

    def load_from_path(self, fn):
        if os.path.exists(fn):
            d = torch.load(fn)
            self.model.load_state_dict(d['state_dict'])
        else:
            raise FileNotFoundError(f"{fn} not exists !")

    def log_test_metrics(self, metrics: KGLPMetrics, epoch: int):
        self.writer.add_scalar('T-MRR', metrics.mrr, epoch)
        self.writer.add_scalar('T-HITS@10', metrics.hits_at_10, epoch)
        self.writer.add_scalar('T-HITS@1', metrics.hits_at_1, epoch)
        self.writer.add_scalar('T-HITS@5', metrics.hits_at_5, epoch)

    def log_valid_metrics(self, metrics: KGLPMetrics, epoch: int):
        self.writer.add_scalar('V-MRR', metrics.mrr, epoch)
        self.writer.add_scalar('V-HITS@10', metrics.hits_at_10, epoch)
        self.writer.add_scalar('V-HITS@1', metrics.hits_at_1, epoch)
        self.writer.add_scalar('V-HITS@5', metrics.hits_at_5, epoch)

    def log_losses(self, losses: list):
        self.writer.add_scalar('Loss', np.mean(losses))


def debug_sm():
    obj = StorageMixIn('/home/liangyi/pycharm_v100/tmp/you-are-not-alone', name='test')


if __name__ == '__main__':
    debug_sm()
