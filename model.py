from typing import Literal

import rich
import torch
from torch import nn, Tensor

from layers.transformer import RoFormerSinusoidalPositionalEmbedding
from utils import DictMixIn
from layers.gnn import HetroGNN
from layers.transformer import layer_norm, TransformerSeqEncoderLayer
from data import ConnectionMetaV2
import logging
from snoop import snoop


class RotatTransformerEncoder(nn.Module):

    def __init__(
            self,
            num_encoder_layers: int = 3,
            embedding_dim: int = 100,
            ffn_embedding_dim: int = 800,
            num_attention_heads: int = 4,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            shot: int = 5,
            encoder_normalize_before: bool = False,
            embedding_normalize: bool = False,
            activation_fn: str = "relu",
            embed_scale: float = None,
            rotary_value: bool = True,
            split_rotray_qk=True,
            enable_pos: bool = True,
            device=torch.device('cpu')) -> None:

        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.model_input_dim = embedding_dim
        self.transformer_dim = num_attention_heads * embedding_dim
        _role_embd_model = RoFormerSinusoidalPositionalEmbedding(3, self.embedding_dim)
        # MARK: 2 means head and tail, shape [3, DIM]
        self.role_vector = _role_embd_model.forward(torch.empty(1, 3).shape,
                                                    past_key_values_length=0)
        self.role_vector = self.role_vector.to(device)

        role_indices = [0]
        for _ in range(shot):
            role_indices.append(1)
            role_indices.append(2)
        self.role_indices = torch.tensor(role_indices, dtype=torch.long,
                                         device=device)
        self.shot = shot
        self.embed_scale = embed_scale
        self.cls_repr = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                     requires_grad=True)
        self.attn_scale_factor = 2
        self.num_attention_heads = num_attention_heads
        self.pair_pos_embedding = nn.Embedding(shot + 2, self.model_input_dim)
        self.pos_q_linear = nn.Linear(self.transformer_dim,
                                      self.transformer_dim)
        self.pos_k_linear = nn.Linear(self.transformer_dim,
                                      self.transformer_dim)
        self.pos_scaling = float(
                self.model_input_dim * self.attn_scale_factor) ** -0.5
        self.pos_ln = layer_norm(self.model_input_dim)
        self.enable_pos = enable_pos
        self.layers = nn.ModuleList(
                [
                    TransformerSeqEncoderLayer(
                            embedding_dim=self.model_input_dim * num_attention_heads,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=num_attention_heads,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            activation_fn=activation_fn,
                            attn_scale_factor=self.attn_scale_factor,
                            encoder_normalize_before=encoder_normalize_before,
                            split_rotray_qk=split_rotray_qk
                    )
                    for _ in range(num_encoder_layers)
                ]
        )
        self.rotary_value = rotary_value

        if embedding_normalize:
            self.emb_layer_norm = layer_norm(self.model_input_dim)
        else:
            self.emb_layer_norm = None

        if encoder_normalize_before:
            self.emb_out_layer_norm = layer_norm(self.model_input_dim)
        else:
            self.emb_out_layer_norm = None

    def repeat_dim(self, emb: torch.Tensor):
        return emb.repeat(1, 1, self.num_attention_heads)

    def obtain_weights(self,
                       entity_seq: torch.Tensor,
                       last_state_only: bool = False,
                       aggr_heads: Literal['keep', 'mean', 'sum', 'max'] = 'keep',
                       unify_cls: bool = True):
        N, T, DIM = entity_seq.shape

        cls_repr_ = self.cls_repr.repeat(N, 1, 1)  # [N, 1, DIM]

        x_ = torch.cat([cls_repr_, entity_seq], dim=1)  # [N, 2*K+1, DIM]
        if self.rotary_value:
            cls_role = self.role_vector[0, :].repeat(N, 1, 1)  # [N, 1, DIM]
            head_tail_roles = self.role_vector[1:, :].repeat(N, self.shot, 1)  # [N, 2*shot, DIM]
            all_roles = torch.cat([cls_role, head_tail_roles], dim=1)  # [N, L, DIM]
        else:
            all_roles = None
        seq_len = x_.size(1)

        x = x_.transpose(0, 1)
        weight = self.pos_ln(self.pair_pos_embedding.weight).repeat(1, self.num_attention_heads)
        pos_len = weight.shape[0]
        pos_q = self.pos_q_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        # MARK: unify [CLS]
        # p_0 \dot p_0 is cls to others
        cls_2_other = abs_pos_bias[:, 0, 0]
        # p_1 \dot p_1 is others to cls
        other_2_cls = abs_pos_bias[:, 1, 1]
        # offset
        abs_pos_bias: torch.Tensor = abs_pos_bias[:, 1:, 1:]  # [B, K+1, K+1]
        pos_len_ = abs_pos_bias.shape[-1]
        # [B, K+1, K+1] -> [B, K+1, 1, K+1, 1] -> [B, K+1, 2, K+1, 2] -> [B, 2K+2, 2K+2]
        abs_pos_bias = abs_pos_bias.unsqueeze(
                dim=-1).unsqueeze(dim=2).repeat(1, 1, 2, 1, 2).reshape(-1, 2 * pos_len_,
                                                                       2 * pos_len_)
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]  # [B, 2K+2, 2K+2] -> [B, 2K+1, 2K+1]
        if unify_cls:
            abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
            abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)

        if self.enable_pos:
            abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1),
                                                            -1, -1, -1).reshape(-1,
                                                                                seq_len, seq_len)
        else:
            abs_pos_bias = None

        x = self.repeat_dim(x)
        # MARK: make local_attn_mask
        local_mask_ = torch.ones(abs_pos_bias.size(),
                                 device=entity_seq.device)  # [B, 2K+1, 2K+1]
        for i in range(1, seq_len, 2):
            local_mask_[:, i, i] = 0
            local_mask_[:, i, i + 1] = 0
            local_mask_[:, i + 1, i] = 0
            local_mask_[:, i + 1, i + 1] = 0
        # MARK: Unify CLS
        # MARK: cls to others
        local_mask_[:, 0, :] = 0
        # local_mask_[:, :, 0] = 0

        inner_states = []
        global_weights, local_weights = [], []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            assert isinstance(layer, TransformerSeqEncoderLayer)
            x, global_weight, local_weight = layer.obtain_weights(
                    x,
                    self_attn_bias=abs_pos_bias,
                    sinusoidal_pos=all_roles,
                    rotatry_values=self.rotary_value,
                    local_attn_mask=local_mask_.bool()
            )
            global_weights.append(global_weight)
            local_weights.append(local_weight)
            if not last_state_only:
                inner_states.append(x)

        if self.emb_out_layer_norm is not None:
            x = self.emb_out_layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]  # cls repr, [B, H*DIM]

        if last_state_only:
            inner_states = [x]

        cls_repr = sentence_rep.view(N, self.num_attention_heads, -1)
        if aggr_heads == 'sum':
            cls_repr = cls_repr.sum(dim=1)
        elif aggr_heads == 'mean':
            cls_repr = cls_repr.mean(dim=1)
        elif aggr_heads == 'max':
            cls_repr, _ = cls_repr.max(dim=1)
        else:
            cls_repr = sentence_rep

        return cls_repr, global_weights, local_weights

    # @pysnooper.snoop(custom_repr=(torch.Tensor, debug_tensor))
    # @snoop
    def forward(
            self,
            entity_seq: torch.Tensor,
            last_state_only: bool = True,
            aggr_heads: Literal['keep', 'mean', 'sum', 'max'] = 'keep',
            unify_cls: bool = True,
    ):
        N, T, DIM = entity_seq.shape

        cls_repr_ = self.cls_repr.repeat(N, 1, 1)
        x_ = torch.cat([cls_repr_, entity_seq], dim=1)
        if self.rotary_value:
            cls_role = self.role_vector[0, :].repeat(N, 1, 1)
            head_tail_roles = self.role_vector[1:, :].repeat(N, self.shot, 1)
            all_roles = torch.cat([cls_role, head_tail_roles], dim=1)
        else:
            all_roles = None
        seq_len = x_.size(1)

        # B x T x C -> T x B x C
        x = x_.transpose(0, 1)

        # 0 is for other-to-cls 1 is for cls-to-other
        # Assume the input is ordered.
        # If your input token is permuted, you may need to update this accordingly
        # [4, 50]
        weight = self.pos_ln.forward(self.pair_pos_embedding.weight).repeat(1, self.num_attention_heads)
        pos_len = weight.shape[0]
        pos_q = self.pos_q_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))
        # MARK: unify [CLS]
        # p_0 \dot p_0 is cls to others
        cls_2_other = abs_pos_bias[:, 0, 0]
        # p_1 \dot p_1 is others to cls
        other_2_cls = abs_pos_bias[:, 1, 1]
        # offset
        abs_pos_bias: torch.Tensor = abs_pos_bias[:, 1:, 1:]  # [B, K+1, K+1]
        pos_len_ = abs_pos_bias.shape[-1]
        # [B, K+1, K+1] -> [B, K+1, 1, K+1, 1] -> [B, K+1, 2, K+1, 2] -> [B, 2K+2, 2K+2]
        abs_pos_bias = abs_pos_bias.unsqueeze(
                dim=-1).unsqueeze(dim=2).repeat(1, 1, 2, 1, 2).reshape(-1, 2 * pos_len_,
                                                                       2 * pos_len_)
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]  # [B, 2K+2, 2K+2] -> [B, 2K+1, 2K+1]
        if unify_cls:
            abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
            abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)

        if self.enable_pos:
            abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1),
                                                            -1, -1, -1).reshape(-1,
                                                                                seq_len, seq_len)
        else:
            abs_pos_bias = None

        x = self.repeat_dim(x)
        # MARK: make local_attn_mask
        local_mask_ = torch.ones(abs_pos_bias.size(),
                                 device=entity_seq.device)  # [B, 2K+1, 2K+1]
        for i in range(1, seq_len, 2):
            local_mask_[:, i, i] = 0
            local_mask_[:, i, i + 1] = 0
            local_mask_[:, i + 1, i] = 0
            local_mask_[:, i + 1, i + 1] = 0
        # MARK: Unify CLS
        # MARK: cls to others
        local_mask_[:, 0, :] = 0
        # local_mask_[:, :, 0] = 0

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(x,
                      self_attn_bias=abs_pos_bias,
                      sinusoidal_pos=all_roles,
                      rotatry_values=self.rotary_value,
                      local_attn_mask=local_mask_.bool())
            if not last_state_only:
                inner_states.append(x)

        if self.emb_out_layer_norm is not None:
            x = self.emb_out_layer_norm(x)

        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        cls_repr = sentence_rep.view(N, self.num_attention_heads, -1)
        if aggr_heads == 'sum':
            cls_repr = cls_repr.sum(dim=1)
        elif aggr_heads == 'mean':
            cls_repr = cls_repr.mean(dim=1)
        elif aggr_heads == 'max':
            cls_repr, _ = cls_repr.max(dim=1)
        else:
            cls_repr = sentence_rep

        return inner_states, cls_repr


class ModelConfig(DictMixIn):
    activation_fn = 'gelu'
    embed_dim: int = 100
    encoder: Literal['bare', 'avg', 'attn'] = 'attn'
    num_heads: int = 4
    num_layers: int = 3
    fine_tune: bool = False
    shot: int = 5
    dropout_tr = 0.3
    dropout_embed = 0.3
    max_neighbors: int = 50
    enable_segment = False
    enable_pos = True
    enable_rotary = True
    aggr_heads: Literal['keep', 'mean', 'sum', 'max'] = 'keep'
    mask_pad: bool = False
    matcher: Literal['single', 'resffn'] = 'single'
    unify_cls: bool = True
    split_rotray_qk = True


class LinearScorer(nn.Module):
    def __init__(self, dim: int):
        super(LinearScorer, self).__init__()
        self.lin1 = nn.Linear(dim, dim, False)
        self.lin2 = nn.Linear(dim, 1, False)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.xavier_normal_(self.lin1.weight)

    def forward(self, x):
        x_ = self.lin1(x)
        x_ = x_ + x
        return self.lin2(x_)


class Model(nn.Module):
    def __init__(self, embedding_vec: Tensor, args: ModelConfig,
                 pad_idx: int = -1, device=torch.device('cpu')):
        super(Model, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embedding_vec, freeze=not args.fine_tune)
        self.transformer = RotatTransformerEncoder(embedding_dim=args.embed_dim,
                                                   num_attention_heads=args.num_heads,
                                                   shot=args.shot + 1,
                                                   attention_dropout=args.dropout_tr,
                                                   activation_fn=args.activation_fn,
                                                   dropout=args.dropout_tr,
                                                   activation_dropout=args.dropout_tr,
                                                   split_rotray_qk=args.split_rotray_qk,
                                                   rotary_value=args.enable_rotary,
                                                   enable_pos=args.enable_pos,
                                                   device=device)
        self.args = args
        self.gnn_encoder = HetroGNN(args.embed_dim, args.dropout_embed,
                                    max_neighbors=args.max_neighbors)
        self.embed_dropout = nn.Dropout(args.dropout_embed)
        self.aggr = args.aggr_heads
        if args.aggr_heads == 'keep':
            self.mapper = self.make_mapper(args.matcher, self.transformer.transformer_dim)
        else:
            self.mapper = self.make_mapper(args.matcher, self.transformer.model_input_dim)
        self.pad_idx = pad_idx
        self.to(device=device)

    @staticmethod
    def make_mapper(kind: Literal['single', 'resffn'], dim: int):
        if kind == 'single':
            layer = nn.Linear(in_features=dim, out_features=1, bias=False)
            nn.init.xavier_normal_(layer.weight)
        else:
            layer = LinearScorer(dim)
        return layer

    def transformer_params(self):
        for name, param in self.named_parameters():
            if name.startswith('mapper') or name.startswith('transformer'):
                yield param
            else:
                continue

    def obtain_gnn_weights(self, indices: torch.Tensor):
        embeddings = self.embeddings.forward(indices)
        rel_idx = indices[:, :, 0]
        mask_ = torch.eq(rel_idx, self.pad_idx)
        rel_embeds = embeddings[:, :, 0, :]
        ent_embeds = embeddings[:, :, 1, :]
        out, weights = self.gnn_encoder.forward(rel_embeds, ent_embeds,
                                                return_weights=True,
                                                mask=mask_)
        return weights

    def neighbor_encoder(self, connection):
        N, L, max_neighbor, _ = connection.shape
        rel_idx = connection[:, :, :, 0].squeeze().reshape(-1, max_neighbor)
        mask_ = None
        if self.args.mask_pad:
            mask_ = torch.eq(rel_idx, self.pad_idx)

        rel_embed = self.embeddings.forward(connection[:, :, :, 0])
        ent_embed = self.embeddings.forward(connection[:, :, :, 1])
        ent_embed = self.embed_dropout.forward(ent_embed).reshape(-1, max_neighbor,
                                                                  self.args.embed_dim)
        rel_embed = self.embed_dropout.forward(rel_embed).reshape(-1, max_neighbor,
                                                                  self.args.embed_dim)
        out = self.gnn_encoder.forward(rel_embed, ent_embed, mask_)
        return out.reshape(N, L, -1)

    def debug(self, query_meta: ConnectionMetaV2):
        if self.args.encoder == 'bare':
            return self.bare_forward(query_meta)
        else:
            connection = query_meta.connections
            N, L, max_neighbor, _ = connection.shape
            rel_idx = connection[:, :, :, 0].squeeze().reshape(-1, max_neighbor)  # [NL, MAX]
            mask_ = torch.eq(rel_idx, self.pad_idx)
            rel_embed = self.embeddings.forward(connection[:, :, :, 0])
            ent_embed = self.embeddings.forward(connection[:, :, :, 1])
            ent_embed = self.embed_dropout.forward(ent_embed).reshape(-1, max_neighbor,
                                                                      self.args.embed_dim)
            rel_embed = self.embed_dropout.forward(rel_embed).reshape(-1, max_neighbor,
                                                                      self.args.embed_dim)
            out, weights = self.gnn_encoder.forward(rel_embed, ent_embed, mask_,
                                                    return_weights=True)
            query_embed = self.neighbor_encoder(query_meta.connections)
            _, query_cls = self.transformer.forward(query_embed)
            return self.mapper.forward(query_cls)

    def bare_forward(self, query_meta: ConnectionMetaV2):

        N, K, _ = query_meta.indices.shape
        query_embed: Tensor = self.embed_dropout(
                self.embeddings(query_meta.indices))  # [N, K, 2, dim]
        query_embed = query_embed.reshape(N, -1, self.args.embed_dim)
        _, query_cls = self.transformer.forward(query_embed)
        return self.mapper.forward(query_cls)

    def obtain_weights(self, query_meta: ConnectionMetaV2):
        query_embed = self.neighbor_encoder(query_meta.connections)
        cls, global_weights, local_weights = self.transformer.obtain_weights(query_embed,
                                                                             aggr_heads=self.aggr,
                                                                             unify_cls=self.args.unify_cls)
        return self.mapper.forward(cls), global_weights, local_weights

    def pooling(self, x: Tensor):
        if self.args.aggr_heads == 'keep':
            return x
        if x.dim() == 3:
            B, T, _ = x.shape
            x_ = x.view(B, T, self.args.num_heads, -1)
        elif x.dim() == 2:
            B = x.shape[0]
            x_ = x.view(B, self.args.num_heads, -1)
        else:
            return x
        if self.args.aggr_heads == 'sum':
            return x_.sum(dim=-2)
        elif self.args.aggr_heads == 'mean':
            return x_.mean(dim=-2)
        else:
            return x

    def forward(self, query_meta: ConnectionMetaV2):
        if self.args.encoder == 'bare':
            return self.bare_forward(query_meta)
        else:
            query_embed = self.neighbor_encoder(query_meta.connections)
            last_states, query_cls = self.transformer.forward(query_embed,
                                                              aggr_heads=self.aggr,
                                                              unify_cls=self.args.unify_cls)

            return self.mapper.forward(query_cls)
