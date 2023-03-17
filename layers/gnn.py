from typing import Optional
import torch
from torch import nn, Tensor
import numpy as np


class LinearFusionLayer(nn.Module):
    def __init__(self, dim, dropout: float = 0.5):
        super(LinearFusionLayer, self).__init__()
        self.left_linear = nn.Linear(dim, dim, False)
        self.right_linear = nn.Linear(dim, dim, False)
        nn.init.xavier_normal_(self.left_linear.weight)
        nn.init.xavier_normal_(self.right_linear.weight)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, left, right):
        x1 = torch.relu(self.left_linear(left) + self.right_linear(right))
        x1 = self.dropout(x1)
        out = self.layer_norm(x1 + right)
        return out


class AvgNeighborEncoder(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.3):
        super(AvgNeighborEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.gcn_b = nn.Parameter(torch.randn(self.embed_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.gcn_w.weight)
        nn.init.constant_(self.gcn_b, 0)

    def forward(self, rel_embeds, ent_embeds, num_neighbors):
        num_neighbors = num_neighbors.unsqueeze(1)
        concat_embeds = self.dropout(torch.cat((rel_embeds, ent_embeds), dim=-1))

        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()


class HetroGNN(nn.Module):
    def __init__(self, model_dim: int,
                 dropout: float = 0.1, max_neighbors: int = 30):
        super(HetroGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.neigh_att_W = nn.Linear(2 * model_dim, model_dim)
        self.neigh_att_u = nn.Linear(model_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.max_neighbors = max_neighbors
        self.embed_dim = model_dim
        nn.init.xavier_normal_(self.neigh_att_W.weight)
        nn.init.xavier_normal_(self.neigh_att_u.weight)

    def forward(self, rel_embeds: torch.Tensor, ent_embeds: torch.Tensor,
                mask: Optional[Tensor] = None, return_weights: bool = False,
                preserve_rel: bool = False):

        assert rel_embeds.is_same_size(ent_embeds)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)  # (batch, 200, 2*embed_dim)
        out = self.neigh_att_W(concat_embeds).tanh()
        att_w = self.neigh_att_u(out)  # [N, Max, 1]

        if mask is not None:
            mask_ = mask.unsqueeze(dim=-1)
            att_w = torch.masked_fill(att_w, mask_, value=-np.Inf)
        att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, self.max_neighbors)
        if preserve_rel:
            er_embeds = ent_embeds + rel_embeds
            out = torch.bmm(att_w, er_embeds).view(concat_embeds.size()[0], self.embed_dim)
        else:
            out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim)
        if return_weights:
            return out.tanh(), att_w
        return out.tanh()


class HetroGNNWithFusion(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.1, max_neighbors: int = 30):
        super(HetroGNNWithFusion, self).__init__()
        self.neigh_att_W = nn.Linear(2 * model_dim, model_dim)
        self.neigh_att_u = nn.Linear(model_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.max_neighbors = max_neighbors
        self.embed_dim = model_dim
        self.fusion = LinearFusionLayer(model_dim, dropout)
        nn.init.xavier_normal_(self.neigh_att_W.weight)
        nn.init.xavier_normal_(self.neigh_att_u.weight)

    def forward(self, rel_embeds: torch.Tensor, ent_embeds: torch.Tensor,
                origin_embeds: torch.Tensor, mask: torch.Tensor):
        assert rel_embeds.is_same_size(ent_embeds)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.neigh_att_W(concat_embeds).tanh()
        att_w = self.neigh_att_u(out)
        if mask is not None:
            mask_ = mask.unsqueeze(dim=-1)
            att_w = torch.masked_fill(att_w, mask_, value=-np.Inf)
        att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, self.max_neighbors)
        out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim).tanh()
        fusion_ = self.fusion.forward(out, origin_embeds)
        return fusion_


class HetroGateGNN(nn.Module):
    def __init__(self, model_dim: int,
                 dropout: float = 0.1, max_neighbors: int = 30):
        super(HetroGateGNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.neigh_att_W = nn.Linear(2 * model_dim, model_dim)
        self.neigh_att_u = nn.Linear(model_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.max_neighbors = max_neighbors
        self.embed_dim = model_dim
        self.gate_mapper = nn.Linear(in_features=model_dim, out_features=1)
        self.fusion_W = nn.Linear(in_features=model_dim, out_features=model_dim, bias=False)
        self.fusion_b = nn.Parameter(torch.randn(model_dim), requires_grad=True)
        self.leakly_relu_layer = nn.LeakyReLU()
        nn.init.zeros_(self.fusion_b)
        nn.init.xavier_normal_(self.neigh_att_W.weight)
        nn.init.xavier_normal_(self.neigh_att_u.weight)

    def forward(self, rel_embeds: torch.Tensor, ent_embeds: torch.Tensor,
                origin_embeds: torch.Tensor):
        dim = rel_embeds.size()[-1]
        N = rel_embeds.size()[0]
        assert rel_embeds.is_same_size(ent_embeds)
        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.neigh_att_W(concat_embeds)
        att_w = self.neigh_att_u(out)
        att_w = self.leakly_relu_layer(att_w)
        att_w = self.softmax(att_w).view(concat_embeds.size()[0], 1, self.max_neighbors)
        out = torch.bmm(att_w, ent_embeds).view(concat_embeds.size()[0], self.embed_dim)
        g_ = self.gate_mapper.forward(out).sigmoid().repeat(1, dim)  # [N, 1]
        gate_out = g_ * out  # [N, dim]
        b_ = self.fusion_b.repeat(N, 1)
        out = gate_out + (1.0 - g_) * self.fusion_W.forward(origin_embeds) + b_
        return out.tanh()
