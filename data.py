import os
from collections import defaultdict
from logging import Logger
import numpy as np
import random
from rich.progress import track as tqdm
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader
from typing import NamedTuple, Literal
from utils import json
import os.path as op
import itertools

DATASET = Literal['wiki', 'nell']
PRETRAINED = Literal['ComplEx', 'TransE', 'RESCAL', 'DistMult']

SERVER_HOME = '/home/liangyi/pycharm_v100/tmp/transmetrics'
LOCAL_HOME = '/Users/yat/code/PyCharmsProjects/TransMetrics'


class ConnectionMeta(NamedTuple):
    indices: torch.Tensor  # [N, K, 2]
    left_connections: torch.Tensor
    left_degrees: torch.Tensor
    right_connections: torch.Tensor
    right_degrees: torch.Tensor


class ConnectionMetaV2(NamedTuple):
    indices: torch.Tensor  # [N, K, 2]
    connections: torch.Tensor
    degrees: torch.Tensor


def deploy_meta_to(meta: ConnectionMetaV2, device):
    return ConnectionMetaV2(
            indices=meta.indices.to(device),
            connections=meta.connections.to(device),
            degrees=meta.degrees.to(device)
    )


class GraphMixIn:

    def __init__(self, num_ents: int, pad_id: int, path_graph: str, symbol_id: dict, max_=50):
        self.num_ents = num_ents
        self.pad_id = pad_id
        self.fn = path_graph
        self.symbol2id = symbol_id
        self.entities = set()
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        self.connections = None
        self.build_graph(max_)

    def build_graph(self, max_=50):
        self.connections = (np.ones((len(self.symbol2id), max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)

        with open(self.fn) as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1]))
                self.entities.add(e1)
                self.entities.add(e2)

        degrees = {}
        ent2id = {e: self.symbol2id[e] for e in self.entities}
        for ent, id_ in ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]
                self.connections[id_, idx, 1] = _[1]

        return degrees

    def get_neighbors(self, entities):
        return torch.tensor(np.stack([self.connections[_, :, :] for _ in entities]))

    def get_seq_meta(self, pair_seq, device=torch.device('cpu')):
        left_connections = torch.tensor(np.stack([[self.connections[_[0], :, :] for _ in pair]
                                                  for pair in pair_seq],
                                                 axis=0),
                                        dtype=torch.long,
                                        device=device)
        left_degrees = torch.tensor([[self.e1_degrees[_[0]] for _ in pair] for pair in pair_seq],
                                    dtype=torch.float,
                                    device=device)
        right_connections = torch.tensor(np.stack([[self.connections[_[1], :, :] for _ in pair]
                                                   for pair in pair_seq],
                                                  axis=0),
                                         dtype=torch.long,
                                         device=device)
        right_degrees = torch.tensor([[self.e1_degrees[_[0]] for _ in pair] for pair in pair_seq],
                                     dtype=torch.float,
                                     device=device)

        return ConnectionMeta(
                indices=torch.tensor(pair_seq, device=device),
                left_connections=left_connections,
                left_degrees=left_degrees,
                right_connections=right_connections,
                right_degrees=right_degrees)

    def get_meta(self, pair, device=torch.device('cpu')) -> ConnectionMeta:
        """

        :param pair: [[left_idx, right_idx]]
        :param device:
        :return: _connections with shape [K, max_neighbors, 2]; _degrees with shape [K]
        """
        left_connections = torch.tensor(np.stack([self.connections[_[0], :, :] for _ in pair],
                                                 axis=0),
                                        dtype=torch.long,
                                        device=device)
        left_degrees = torch.tensor([self.e1_degrees[_[0]] for _ in pair], dtype=torch.float,
                                    device=device)
        right_connections = torch.tensor(np.stack([self.connections[_[1], :, :] for _ in pair],
                                                  axis=0), dtype=torch.long,
                                         device=device)
        right_degrees = torch.tensor([self.e1_degrees[_[1]] for _ in pair], dtype=torch.float,
                                     device=device)
        return ConnectionMeta(
                indices=torch.tensor(pair, device=device),
                left_connections=left_connections,
                left_degrees=left_degrees,
                right_connections=right_connections,
                right_degrees=right_degrees)

    def get_meta_v2(self, pairs, device=torch.device('cpu')) -> ConnectionMetaV2:
        """

        :param pairs: [[left_idx, right_idx]]*N
        :param device:
        :return: _connections with shape [K, max_neighbors, 2]; _degrees with shape [K]
        """
        N = len(pairs)
        pair_seq = np.array(pairs).reshape((N, -1)).tolist()
        connections = torch.tensor(np.stack([[self.connections[_, :, :] for _ in seq]
                                             for seq in pair_seq],
                                            axis=0),
                                   dtype=torch.long,
                                   device=device)
        degrees = torch.tensor([[self.e1_degrees[_] for _ in seq]
                                for seq in pair_seq],
                               dtype=torch.float,
                               device=device)

        return ConnectionMetaV2(
                indices=torch.tensor(pairs, device=device),
                connections=connections,
                degrees=degrees)


class DataPath:
    PAD_SYM = '<PAD>'

    def __init__(self, parent: str, dataset: DATASET = 'nell', spl: str = '\t'):
        self.split = spl
        self.parent = op.join(parent, 'datasets', dataset)
        self._dataset = dataset
        self.num_ents = -1
        self.pad_idx = -1

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_tasks(self):
        return op.join(self.parent, 'train_tasks.json')

    @property
    def dev_tasks(self):
        return op.join(self.parent, 'dev_tasks.json')

    @property
    def case_tasks(self):
        return op.join(self.parent, 'case_tasks.json')

    @property
    def test_tasks(self):
        return op.join(self.parent, 'test_tasks.json')

    @property
    def ent2ids(self):
        return op.join(self.parent, 'ent2ids')

    @property
    def rel2ids(self):
        return op.join(self.parent, 'relation2ids')

    @property
    def path_graph(self):
        return op.join(self.parent, 'path_graph')

    @property
    def e1rel_e2(self):
        return op.join(self.parent, 'e1rel_e2.json')

    @property
    def rel2cand(self):
        return op.join(self.parent, 'rel2candidates.json')

    def provide(self, fn):
        return op.join(self.parent, fn)

    def symbol2vec(self, model: PRETRAINED = 'TransE'):
        return op.join(self.parent, 'embed', f"symbol2vec.{model}")

    def ent2vec(self, model: PRETRAINED = 'TransE'):
        return op.join(self.parent, 'embed', f"entity2vec.{model}")

    def rel2vec(self, model: PRETRAINED = 'TransE'):
        return op.join(self.parent, 'embed', f"relation2vec.{model}")

    def __str__(self):
        return f"[HOME] {self.parent}\n[DATASET] {self.dataset}"

    def load_openke(self, embed_model: str, fn: str):
        symbol_id = {}
        rel2id = json.load(self.rel2ids)
        ent2id = json.load(self.ent2ids)
        assert os.path.exists(self.provide(fn)), f"Please provide correct {self.provide(fn)}"
        ckpt = torch.load(self.provide(fn), map_location=torch.device('cpu'))
        if embed_model == 'ComplEx':
            ent_re = ckpt['ent_re_embeddings.weight']
            ent_im = ckpt['ent_im_embeddings.weight']
            rel_re = ckpt['rel_re_embeddings.weight']
            rel_im = ckpt['rel_im_embeddings.weight']
            ent_embed = torch.cat([ent_re, ent_im], dim=-1)
            rel_embed = torch.cat([rel_re, rel_im], dim=-1)

            # normalize the complex embeddings
            ent_mean = torch.mean(ent_embed, dim=1, keepdim=True)
            ent_std = torch.std(ent_embed, dim=1, keepdim=True)
            rel_mean = torch.mean(rel_embed, dim=1, keepdim=True)
            rel_std = torch.std(rel_embed, dim=1, keepdim=True)
            eps = 1e-3
            ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
            rel_embed = (rel_embed - rel_mean) / (rel_std + eps)
        else:
            ent_embed = ckpt['ent_embeddings.weight']
            rel_embed = ckpt['rel_embeddings.weight']
        assert ent_embed.shape[0] == len(ent2id.keys())
        assert rel_embed.shape[0] == len(rel2id.keys())
        self.num_ents = len(ent2id.keys())

        i = 0
        embeddings = []
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(rel_embed[rel2id[key], :])

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1
                embeddings.append(ent_embed[ent2id[key], :])
        symbol_id[self.PAD_SYM] = i
        embeddings.append(torch.zeros((rel_embed.shape[1],)))
        self.pad_idx = i
        i += 1
        embed_th = torch.stack(embeddings, dim=0).to(dtype=torch.float)
        return symbol_id, embed_th


class Dataset:

    def __init__(self, logger_: Logger, parent: str, openke_fn: str, dataset: DATASET = 'nell', few: int = 3,
                 pretrained: Literal['DistMult', 'TransE', 'ComplEx', 'RESCAL'] = 'ComplEx',
                 max_neighbors=50, neg_rate: int = 1):
        self.data_path = DataPath(parent, dataset)
        self.logger = logger_
        self.logger.info("Prepare Dataset...")
        self.logger.info('Loading rel2candidates...')
        self.rel2cand = json.load(self.data_path.rel2cand)
        self.e1rel_e2 = json.load(self.data_path.e1rel_e2)
        self.few = few
        self.logger.info("Loading pretrain...")
        self.symbol2id, self.symbol2vec = self.data_path.load_openke(pretrained, fn=openke_fn)
        self.pad_idx = self.data_path.pad_idx
        self.id2symbol = {v: k for k, v in self.symbol2id.items()}
        self.logger.info(f"Total {len(self.symbol2id)} symbols")
        self.train_tasks = json.load(self.data_path.train_tasks)
        self.dev_tasks = json.load(self.data_path.dev_tasks)
        self.test_tasks = json.load(self.data_path.test_tasks)
        if op.exists(self.data_path.case_tasks):
            self.case_tasks = json.load(self.data_path.case_tasks)
        else:
            self.case_tasks = None
        self.graph_mixin = GraphMixIn(self.num_ents, self.symbol2id['<PAD>'],
                                      self.data_path.path_graph, self.symbol2id,
                                      max_neighbors)
        self.pretrained = pretrained
        self.dataset = dataset
        self.neg_rate = neg_rate

    def _mix_sup_query(self, sup, query, mode='replace'):
        result = list()
        if mode == 'replace':
            _sup = sup[:-1]
        else:
            _sup = sup
        for h, _, t in _sup:
            result.append([self.symbol2id[h], self.symbol2id[t]])
        result.append([self.symbol2id[query[0]], self.symbol2id[query[-1]]])
        return result

    def build_support_sequence(self, sup):
        result = list()
        for h, _, t in sup:
            result.append([self.symbol2id[h], self.symbol2id[t]])
        return result

    def filter_rel(self, rel):
        if len(self.rel2cand[rel]) <= 20:
            return True
        if len(self.train_tasks[rel]) <= self.few:
            return True

    def build_train_sequences(self, rel: str,
                              query_size: int = 100,
                              label=(0., 1.)):
        references = self.train_tasks[rel]
        candidates = self.rel2cand[rel]
        random.shuffle(references)
        supports = references[:self.few]
        others = references[self.few:]
        result = list()
        labels = list()
        sup_idx = []
        for (_h, _r, _t) in supports:
            sup_idx.append([self.symbol2id[_h], self.symbol2id[_t]])

        if len(others) >= query_size:
            queries = random.sample(others, k=query_size)
            for q in queries:
                qh_idx = self.symbol2id[q[0]]
                result.append(sup_idx + [[qh_idx, self.symbol2id[q[-1]]]])
                labels.append(label[1])
                h, r, t = q
                num_negs = 0
                neg_ents = set()
                while num_negs < self.neg_rate:
                    false = random.choice(candidates)
                    if false != t and (false not in self.e1rel_e2[h + r]):
                        false_q = sup_idx + [[qh_idx, self.symbol2id[false]]]
                        result.append(false_q)
                        labels.append(label[0])
                        num_negs += 1
                        neg_ents.add(self.symbol2id[false])
                    else:
                        continue
        else:
            count = 0
            while count < query_size:
                pair = random.choice(others)
                h, r, t = pair
                # need to modify line below to remove bias
                query_ = [[self.symbol2id[h], self.symbol2id[t]]]
                result.append(sup_idx + query_)
                labels.append(label[1])
                num_negs = 0
                neg_ents = set()
                while num_negs < self.neg_rate:
                    false = random.choice(candidates)
                    if false != t and (false not in self.e1rel_e2[h + r]):
                        false_q = [[self.symbol2id[h], self.symbol2id[false]]]
                        result.append(sup_idx + false_q)
                        labels.append(label[0])
                        num_negs += 1
                        neg_ents.add(self.symbol2id[false])
                    else:
                        continue
                count += 1
        assert len(result) == len(labels)
        return [sup_idx], result, labels

    def split_supports(self, references: list, query_size: int = 100,
                       return_index: bool = False):
        random.shuffle(references)
        pos_queries = list()
        supports = references[:self.few]
        others = references[self.few:]
        if len(others) >= query_size:
            if return_index:
                supports = self.map_symbol_idx(supports)
                queries = random.sample(others, k=query_size)
                queries = self.map_symbol_idx(queries)
                return supports, queries
            return supports, random.sample(others, k=query_size)
        else:
            while True:
                pair = random.choice(others)
                pos_queries.append(pair)
                if len(pos_queries) == query_size:
                    break
            if return_index:
                supports = self.map_symbol_idx(supports)
                pos_queries = self.map_symbol_idx(pos_queries)
                return supports, pos_queries
            return supports, pos_queries

    def map_symbol_idx(self, seq: list[list[str]]):
        result = list()
        for (h, _, t) in seq:
            result.append([self.symbol2id[h], self.symbol2id[t]])
        return result

    def build_false_samples(self, queries: list, rel: str):
        candidates = self.rel2cand[rel]
        false_samples = list()
        for (h, r, t) in queries:
            while True:
                false = random.choice(candidates)
                if false != t and (false not in self.e1rel_e2[h + r]):
                    false_samples.append([h, r, false])
                    break
        assert len(false_samples) == len(queries)
        return false_samples

    def build_eval_queries(self, sym_support, golden_triplet,
                           candidate_size: int = -1):
        sup_idx = []

        for (_h, _r, _t) in sym_support:
            sup_idx.append([self.symbol2id[_h], self.symbol2id[_t]])

        h, r, t = golden_triplet
        candidates = self.rel2cand[r]
        queries = [sup_idx + [[self.symbol2id[h], self.symbol2id[t]]]]
        for cand in candidates:
            if cand != t and cand not in self.e1rel_e2[h + r]:
                queries.append(sup_idx + [[self.symbol2id[h], self.symbol2id[cand]]])
        if candidate_size == -1 or len(queries) <= candidate_size:
            return [sup_idx], queries
        else:
            return [sup_idx], queries[:candidate_size]

    def map_seq_idx(self, seq: list[list[str]]):
        result = list()
        for (h, _, t) in seq:
            result.append([self.symbol2id[h], self.symbol2id[t]])
        return result

    @property
    def num_ents(self):
        return self.data_path.num_ents
