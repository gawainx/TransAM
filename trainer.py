import datetime
import random
import collections
import os
from typing import Literal

from utils.mixin import StorageMixIn
from utils import DictMixIn
from model import Model, ModelConfig
from data import Dataset, DATASET, PRETRAINED
from torch.optim import Adam
import utils.optimkit as optimkit
from utils import partition
from rich.progress import track
import tqdm
import torch
from metrics import KGLPMetrics, EvalLogItem
import numpy as np
import toml
from utils.tbhelper import TBHelper
from utils import time_suffix
import typer
import pickle

app = typer.Typer(pretty_exceptions_enable=False)

SPLIT = '*' * 32

PATH = 'toml'


class Config(DictMixIn):
    candidate_size: int = -1
    checkpoint: str = 'path/to/checkpoint'
    embed_fn: str = 'embeddings.ckpt'
    model = ModelConfig()
    parent = ''
    eval_size: int = 2000
    debug = False
    dataset: DATASET = 'nell'
    pretrain: PRETRAINED = 'ComplEx'
    name: str = 'TransAM'
    max_epochs: int = 500000
    log_every: int = 50
    eval_every: int = 10000
    warmup_epochs: int = 10000
    weight_decay = 0.0
    lr: int = 5e-5
    query_size: int = 100
    gpu: int = 2
    seed = 1234
    margin = 2.0
    neg_rate: int = 3
    mode = 'append'
    grad_clip: float = 5.0
    test_in_fit: bool = True
    max_eval_count: int = 5
    test = False

    @classmethod
    def from_dict(cls, kwargs):
        obj = cls()
        for k, v in kwargs.items():
            if isinstance(v, dict):
                cls.__setattr__(obj, k, ModelConfig.from_dict(v))
            else:
                cls.__setattr__(obj, k, v)

        return obj


class Trainer(StorageMixIn):
    def __init__(self, args: Config):
        _name = f"{args.model.encoder}{args.name}.{args.model.shot}SHOT-" \
                f"{args.dataset}-{args.pretrain}"
        StorageMixIn.__init__(self, args.parent, _name, debug=args.debug,
                              on_train=not args.test)
        self.name = _name
        self.args = args
        self.neg_rate = args.neg_rate
        self.shot = self.K = args.model.shot
        self.dataset = Dataset(logger_=self.logger, parent=self.args.parent, dataset=args.dataset,
                               few=args.model.shot, openke_fn=args.embed_fn,
                               pretrained=args.pretrain, max_neighbors=args.model.max_neighbors)
        self.device = self.configure_device()
        self.model = Model(self.dataset.symbol2vec, args.model,
                           pad_idx=self.dataset.pad_idx, device=self.device)
        self.parameters = filter(lambda p: p.requires_grad,
                                 self.model.parameters())
        self.optimizer = Adam(params=self.parameters, lr=self.args.lr,
                              weight_decay=args.weight_decay)
        self.cur_epoch = 0
        self.lr = args.lr
        self.max_epochs = args.max_epochs
        self.warmup_epochs = args.warmup_epochs
        self.log_every = args.log_every
        self.eval_every = args.eval_every
        self.test_results = collections.defaultdict(KGLPMetrics)
        self.valid_results = collections.defaultdict(KGLPMetrics)
        self.saved_models = dict()
        self.eval_logs: list[EvalLogItem] = list()
        self.best_test = KGLPMetrics()
        self.best_valid = KGLPMetrics()
        self.best_test_epo = -1
        self.best_val_epo = -1
        self.bce_loss = torch.nn.BCELoss()
        self.criterion = torch.nn.BCELoss()
        self.labels = (0., 1.)
        optimkit.seed_everything(args.seed)
        self.critical(f"Training Config:\n{self.args.to_dict()}")
        self.writer = TBHelper(self.proj_dir)
        ckpt_fn = os.path.join(self.args.parent,
                               'output',
                               self.args.checkpoint)
        if os.path.exists(ckpt_fn):
            self.critical(f"Loading from {ckpt_fn}")
            self.load_from_path(ckpt_fn)

    def train_generator(self):
        task_pool = list(self.dataset.train_tasks.keys())
        rel_idx = 0
        while True:
            if rel_idx % self.args.log_every == 0:
                random.shuffle(task_pool)
            rel = task_pool[rel_idx % self.args.log_every]
            rel_idx += 1
            if self.dataset.filter_rel(rel):
                continue
            sup_seq, query_seq, label = self.dataset.build_train_sequences(rel,
                                                                           query_size=self.args.query_size,
                                                                           label=self.labels)
            query_meta = self.dataset.graph_mixin.get_meta_v2(query_seq, device=self.device)
            support_meta = self.dataset.graph_mixin.get_meta_v2(sup_seq, device=self.device)
            labels = torch.tensor(label, device=self.device)
            yield support_meta, query_meta, labels

    def configure_device(self):
        if torch.cuda.is_available():
            if self.args.gpu >= 0:
                return torch.device(f"cuda:{self.args.gpu}")
            else:
                self.warn("GPU is available but not used !")
        return torch.device('cpu')

    def step_forward(self, dataloader):
        support_meta, query_meta, labels = next(dataloader)
        if self.cur_epoch == 0:
            self.info(f"{labels.shape=}")
        scores = self.model.forward(query_meta)
        scores = self.calc_scores(scores)
        bce_loss = self.bce_loss.forward(scores, labels)
        bce_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_clip)
        return bce_loss

    def fit(self):
        self.info("Start fitting...")
        max_epochs = self.args.max_epochs
        dataloader = self.train_generator()
        mills = []
        if max_epochs <= self.log_every:
            total_time = None
            while self.cur_epoch <= max_epochs:
                support_meta, query_meta, labels = next(dataloader)
                t0 = datetime.datetime.now()
                scores = self.model.forward(query_meta)
                scores = self.calc_scores(scores)
                loss = self.bce_loss.forward(scores, labels)
                loss.backward()
                t1 = datetime.datetime.now()
                mills.append((t1 - t0).microseconds)
                if total_time is None:
                    total_time = t1 - t0
                else:
                    total_time += (t1 - t0)
                torch.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_clip)

                lr = self.step_()
                self.info(f"EPO {self.cur_epoch}: Loss {loss.cpu().item():.6f}\t lr: {lr} with "
                          f"time {(t1 - t0).microseconds:.6f}")
                self.cur_epoch += 1
            mills.remove(np.max(mills))
            mills.remove(np.min(mills))
            self.info(f"Avg time per epoch {np.mean(mills):.6f}")
        else:
            losses = collections.deque([], maxlen=self.log_every)
            try:
                mills = []
                total_time = datetime.timedelta()
                cnt = 0
                while self.cur_epoch <= max_epochs:
                    if not self.model.training:
                        self.model.train()
                    t0 = datetime.datetime.now()
                    loss = self.step_forward(dataloader)
                    t1 = datetime.datetime.now()
                    total_time += (t1 - t0)
                    cnt += 1
                    if cnt == 10000:
                        self.critical(f"10000 epoches total {total_time.total_seconds()} seconds.")
                    losses.append(loss.cpu().item())
                    mills.append((t1 - t0).microseconds)
                    lr = self.step_()
                    if self.cur_epoch % self.log_every == 0 and self.cur_epoch != 0:
                        self.info(f"EPO {self.cur_epoch}: Avg. Loss {np.mean(losses):.6f}\t lr: "
                                  f"{lr:.6f} Avg mills {np.mean(mills):.2f}")
                        mills.clear()
                    if self.cur_epoch % self.eval_every == 0 and self.cur_epoch != 0:
                        self.critical(f"Evaluate after {self.cur_epoch} epochs...")
                        eval_metrics = self.eval_(cand_size=self.args.candidate_size)
                        self.critical(f"EPOCH {self.cur_epoch} valid result \n    {eval_metrics}")
                        self.writer.log_valid_metrics(eval_metrics, self.cur_epoch)
                        if eval_metrics > self.best_valid:
                            self.best_valid = eval_metrics.copy()
                            self.best_val_epo = self.cur_epoch
                            self.valid_results[self.best_val_epo] = eval_metrics
                            if not self.args.debug:
                                fn = self.save_model()
                                self.eval_logs.append(EvalLogItem(eval_metrics.copy(),
                                                                  self.cur_epoch,
                                                                  fn))
                                self.saved_models[self.cur_epoch] = \
                                    {"ckpt":  fn,
                                     'valid': eval_metrics.copy()}
                                self.critical(f"Model saved in {fn}")

                    self.cur_epoch += 1
                # latest_valid = self.eval_()
                latest_test = self.eval_('test', cand_size=self.args.candidate_size)
                self.critical(f"LATEST TEST {latest_test}")
                self.save_latest()
                self.load_in_fit(self.best_val_epo)
                best_test = self.eval_('test', cand_size=self.args.candidate_size)
                self.critical(f"BEST TEST {best_test}")
            except KeyboardInterrupt:
                self.eval_('test')
                self.critical(f"BEST Valid {self.best_valid} ON EPOCH {self.best_val_epo}")
                self.load_in_fit(self.best_val_epo)
                best_test = self.eval_('test', cand_size=self.args.candidate_size)
                self.critical(f"BEST TEST {best_test}")

    def step_(self):
        lr = optimkit.adjust_learning_rate(self.optimizer, self.cur_epoch,
                                           self.lr, warm_up_step=self.warmup_epochs,
                                           max_update_step=self.max_epochs)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return lr

    def calc_scores(self, scores: torch.Tensor):
        return scores.squeeze().sigmoid()

    def eval_(self, stage: Literal['test', 'valid'] = 'valid', cand_size: int = -1):
        self.model.eval()
        with torch.no_grad():
            self.critical(f'Evaluating on {stage.upper()} Dataset.')
            total_metrics = KGLPMetrics()
            tasks = self.dataset.dev_tasks if stage == 'valid' else self.dataset.test_tasks
            relations = list(tasks)
            results = dict()
            for idx, rel in enumerate(relations):
                self.critical(f"Eval for Rel {rel}")
                rel_metrics = KGLPMetrics()
                references = tasks[rel]
                supports = references[:self.K]
                others = references[self.K:]
                num_cands = []
                for pair in track(others, description='Pair'):
                    all_socres_ = []
                    sup_idx_, eval_queries_ = self.dataset.build_eval_queries(supports,
                                                                              pair,
                                                                              candidate_size=cand_size)
                    if self.args.eval_size == -1:
                        query_meta = self.dataset.graph_mixin.get_meta_v2(eval_queries_,
                                                                          device=self.device)
                        scores = self.calc_scores(self.model.forward(query_meta))
                        rel_metrics.perform_metrics(scores.detach().cpu())
                        num_cands.append(len(scores))
                    else:
                        part_queries = partition(eval_queries_, self.args.eval_size)
                        for val_query in part_queries:
                            query_meta = self.dataset.graph_mixin.get_meta_v2(val_query,
                                                                              device=self.device)
                            scores = self.calc_scores(self.model.forward(query_meta))
                            all_socres_.extend(scores.detach().cpu().tolist())
                        num_cands.append(len(all_socres_))
                        rel_metrics.perform_metrics(all_socres_)
                self.critical(f"#{idx + 1} Rel {rel}: Avg. Cands: {np.mean(num_cands):.2f}"
                              f" #Samp. {len(others)}"
                              f" Metrics:")
                self.critical(rel_metrics)
                results[idx + 1] = rel_metrics

                total_metrics += rel_metrics
            self.critical(f"TOTAL METRICS {total_metrics}")
            self.model.train()
        return total_metrics

    def save_model(self, suffix: str = 'best'):
        fn = os.path.join(self.ckpt_dir,
                          f"{self.name}_EPO{self.cur_epoch}_{suffix}.ckpt")
        result = {
            'state_dict': self.model.state_dict(),
            'optim':      self.optimizer.state_dict(),
            'epoch':      self.cur_epoch
        }
        if os.path.exists(fn):
            fn = os.path.join(self.ckpt_dir,
                              f"{self.name}_EPO{self.cur_epoch}_{suffix}_{time_suffix(True)}.ckpt")
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
            d = torch.load(fn, map_location=self.device)
            self.model.load_state_dict(d['state_dict'])
        else:
            raise FileNotFoundError(f"{fn} not exists !")


@app.command('fit')
def main(fn: str, shot: int = 1):
    toml_fn = os.path.join(PATH, f"{fn}.toml")
    if os.path.exists(toml_fn):
        cfgd = toml.load(open(toml_fn))
        cfg = Config.from_dict(cfgd)
        if shot != -1:
            cfg.model.shot = shot
        trainer = Trainer(cfg)
        trainer.fit()
    else:
        default = Config().to_dict('from_dict')
        toml.dump(default, open(toml_fn, 'w'))
        print(f"FILE {toml_fn} CREATED !")


@app.command('test')
def evaluate(fn: str):
    toml_fn = os.path.join(PATH, f"{fn}.toml")
    if os.path.exists(toml_fn):
        cfgd = toml.load(open(toml_fn))
        cfg = Config.from_dict(cfgd)
        trainer = Trainer(cfg)
        trainer.eval_('test', cand_size=cfg.candidate_size, dump_results=True)
        if cfg.candidate_size != -1:
            trainer.critical(f"TEST on FULL TEST SET")
            trainer.eval_('test', cand_size=-1)
    else:
        default = Config().to_dict('from_dict')
        toml.dump(default, open(toml_fn, 'w'))
        print(f"FILE {toml_fn} CREATED !")


if __name__ == '__main__':
    app()
