from torch import nn
import torch
import numpy as np
from typing import Optional, Literal
from torch.nn import functional as F
import layers.utils as utils
import warnings
from torch import Tensor
from torch.nn import Parameter


def layer_norm(normalized_shape, eps=1e-5, elementwise_affine=True):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 scale_factor,  # type: float
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights=True,  # type: bool
                                 attn_mask: Optional[Tensor] = None,  # type
                                 attn_bias=None,  # type: Optional[Tensor]
                                 sinusoidal_pos: Optional[Tensor] = None,
                                 rotatry_values: bool = False,
                                 local_attn_mask: Optional[Tensor] = None,
                                 split_rotray_qk: bool = False,
                                 local_only: bool = False,
                                 add_local_role: bool = False
                                 ):
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim * scale_factor) ** -0.5
    if split_rotray_qk:
        q, k, v, rql, rkl = F.linear(query, in_proj_weight, in_proj_bias).chunk(5, dim=-1)
        q = q * scaling
        rql = rql * scaling
    else:
        q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
        q = q * scaling
        rql = None
        rkl = None

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
               attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(
                    attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if rql is not None:
        rql = rql.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if rkl is not None:
        rkl = rkl.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
                [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)],
                dim=1)
        v = torch.cat(
                [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)],
                dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    local_attn_out = None
    # MARK: rotat here; Local stage role attention
    if rotatry_values and sinusoidal_pos is not None:
        #     sinusoidal_pos shape [N, L, DIM-per-Head]
        sinusoidal_pos_ = sinusoidal_pos.repeat(num_heads, 1, 1)  # [NH, L, DIM]
        if rql is not None:
            rq, rk = apply_rotary_role_embeddings_v2(sinusoidal_pos_, rql, rkl)
        elif add_local_role:
            rq = q + sinusoidal_pos_
            rk = k + sinusoidal_pos_
        else:
            rq, rk = apply_rotary_role_embeddings_v2(sinusoidal_pos_, q, k)
        local_attn_weights = torch.bmm(rq, rk.transpose(1, 2))
        # MARK: qk -> masked
        if local_attn_mask is not None:
            local_attn_weights.masked_fill_(local_attn_mask, float('-inf'))
            local_attn_weights = F.softmax(local_attn_weights, dim=-1)
            local_attn_weights = F.dropout(local_attn_weights,
                                           p=dropout_p, training=training)
            local_attn_out = torch.bmm(local_attn_weights, v)
            assert list(local_attn_out.size()) == [bsz * num_heads, tgt_len, head_dim]
        if local_only:
            local_attn_out = local_attn_out.transpose(0,
                                                      1).contiguous().view(tgt_len, bsz, embed_dim)
            attn_output = F.linear(local_attn_out, out_proj_weight, out_proj_bias)
            attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
            return attn_output, None
    else:
        local_attn_out = None

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            # attn_mask is all zero
            attn_output_weights += attn_mask

    if attn_bias is not None:
        # MARK: inject seperate position encoding here
        # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights += attn_bias
        # attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    if local_attn_out is not None:
        assert attn_output.is_same_size(local_attn_out), \
            f"Shape ERROR! {attn_output.size()=} but {local_attn_out.size()=}"
        attn_output = attn_output + local_attn_out

    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        local_attn_weights = local_attn_weights.view(bsz, num_heads, tgt_len, src_len)
        if rotatry_values and sinusoidal_pos is not None:
            return attn_output, attn_output_weights.sum(
                    dim=1) / num_heads, local_attn_weights.sum(dim=1) / num_heads
        else:
            return attn_output, attn_output_weights.sum(
                    dim=1) / num_heads
    else:
        return attn_output, None


class MultiheadAttention(nn.Module):
    """MultiHeadAttention
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 scale_factor=1.0, add_bias_kv=False, add_zero_attn=False,
                 split_rotray_qk: bool = False,
                 local_only: bool = False,
                 balanced: float = 0.5,
                 add_local_role: bool = False,
                 rotatry_values: bool = False
                 ):
        super().__init__()
        self.rotatry_values: bool = rotatry_values
        if balanced < 0:
            self.alpha = None
        else:
            self.alpha = balanced
        self.embed_dim = embed_dim
        self.local_only = local_only
        self.add_local_role = add_local_role
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            "embed_dim must be divisible by num_heads"
        if split_rotray_qk:
            WEIGHT_SHAPE = 5
        else:
            WEIGHT_SHAPE = 3
        self.in_proj_weight = Parameter(torch.randn(WEIGHT_SHAPE * embed_dim, embed_dim))
        self.split_rotray_qk = split_rotray_qk

        if bias:
            self.in_proj_bias = Parameter(torch.randn(WEIGHT_SHAPE * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.randn(1, 1, embed_dim))
            self.bias_v = Parameter(torch.randn(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.scale_factor = scale_factor
        self.add_zero_attn = add_zero_attn
        self.dropout_p = dropout

        self.reset_parameters()

    def quick_multi_head_attention_forward(self,
                                           query,  # type: Tensor
                                           key,  # type: Tensor
                                           value,  # type: Tensor
                                           need_weights=True,  # type: bool
                                           attn_mask: Optional[Tensor] = None,  # type
                                           key_padding_mask=None,
                                           attn_bias=None,  # type: Optional[Tensor]
                                           sinusoidal_pos: Optional[Tensor] = None,
                                           local_attn_mask: Optional[Tensor] = None,
                                           ):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert key.size() == value.size()

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        scaling = float(head_dim * self.scale_factor) ** -0.5
        if self.split_rotray_qk:
            q, k, v, rql, rkl = F.linear(query, self.in_proj_weight,
                                         self.in_proj_bias).chunk(5, dim=-1)
            q = q * scaling
            rql = rql * scaling
        else:
            q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
            q = q * scaling
            rql = None
            rkl = None

        if attn_mask is not None:
            assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
                   attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
                'Only float, byte, and bool types are supported for attn_mask, not {}'.format(
                        attn_mask.dtype)
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                        "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError(
                        "attn_mask's dimension {} is not supported".format(attn_mask.dim()))
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                    "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            key_padding_mask = key_padding_mask.to(torch.bool)

        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert self.bias_k is None
            assert self.bias_v is None

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if rql is not None:
            rql = rql.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if rkl is not None:
            rkl = rkl.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                    [k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)],
                    dim=1)
            v = torch.cat(
                    [v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)],
                    dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        local_attn_out = None
        # MARK: rotat here; Local stage role attention
        if self.rotatry_values and sinusoidal_pos is not None:
            #     sinusoidal_pos shape [N, L, DIM-per-Head]
            sinusoidal_pos_ = sinusoidal_pos.repeat(self.num_heads, 1, 1)  # [NH, L, DIM]
            if rql is not None:
                rq, rk = apply_rotary_role_embeddings_v2(sinusoidal_pos_, rql, rkl)
            elif self.add_local_role:
                rq = q + sinusoidal_pos_
                rk = k + sinusoidal_pos_
            else:
                rq, rk = apply_rotary_role_embeddings_v2(sinusoidal_pos_, q, k)
            local_attn_weights = torch.bmm(rq, rk.transpose(1, 2))
            # MARK: qk -> masked
            if local_attn_mask is not None:
                local_attn_weights.masked_fill_(local_attn_mask, float('-inf'))
                local_attn_weights = F.softmax(local_attn_weights, dim=-1)
                local_attn_weights = F.dropout(local_attn_weights,
                                               p=self.dropout_p,
                                               training=self.training)
                local_attn_out = torch.bmm(local_attn_weights, v)
                assert list(local_attn_out.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        else:
            local_attn_out = None

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                # attn_mask is all zero
                attn_output_weights += attn_mask

        if attn_bias is not None:
            # MARK: inject seperate position encoding here
            # attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
            attn_output_weights += attn_bias
            # attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout_p,
                                        training=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        if local_attn_out is not None:
            assert attn_output.is_same_size(local_attn_out), \
                f"Shape ERROR! {attn_output.size()=} but {local_attn_out.size()=}"
            if self.alpha:
                attn_output = self.alpha * attn_output + (1 - self.alpha) * local_attn_out
            else:
                attn_output = attn_output + local_attn_out

        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            local_attn_weights = local_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.rotatry_values and sinusoidal_pos is not None:
                return attn_output, attn_output_weights.sum(
                        dim=1) / self.num_heads, local_attn_weights.sum(dim=1) / self.num_heads
            else:
                return attn_output, attn_output_weights.sum(
                        dim=1) / self.num_heads
        else:
            return attn_output, None

    def reset_parameters(self):
        # Note: these initilaztion will be overrided in `init_bert_params`, if using BERT
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def obtain_weights(self,
                       query, key, value,
                       key_padding_mask=None,
                       attn_mask=None,
                       attn_bias=None,
                       sinusoidal_pos: Optional[Tensor] = None,
                       rotatry_values: bool = False,
                       local_attn_mask: Optional[Tensor] = None,
                       local_only=False
                       ):
        if sinusoidal_pos is not None and rotatry_values:
            _, global_weights, local_weights = multi_head_attention_forward(query, key, value,
                                                                            scale_factor=self.scale_factor,
                                                                            embed_dim_to_check=self.embed_dim,
                                                                            num_heads=self.num_heads,
                                                                            in_proj_weight=self.in_proj_weight,
                                                                            in_proj_bias=self.in_proj_bias,
                                                                            bias_k=self.bias_k,
                                                                            bias_v=self.bias_v,
                                                                            add_zero_attn=self.add_zero_attn,
                                                                            dropout_p=self.dropout,
                                                                            out_proj_weight=self.out_proj.weight,
                                                                            out_proj_bias=self.out_proj.bias,
                                                                            training=self.training,
                                                                            key_padding_mask=key_padding_mask,
                                                                            need_weights=True,
                                                                            attn_mask=attn_mask,
                                                                            attn_bias=attn_bias,
                                                                            sinusoidal_pos=sinusoidal_pos,
                                                                            rotatry_values=rotatry_values,
                                                                            local_attn_mask=local_attn_mask,
                                                                            split_rotray_qk=self.split_rotray_qk,
                                                                            local_only=local_only)
        else:
            _, global_weights = multi_head_attention_forward(query, key, value,
                                                             scale_factor=self.scale_factor,
                                                             embed_dim_to_check=self.embed_dim,
                                                             num_heads=self.num_heads,
                                                             in_proj_weight=self.in_proj_weight,
                                                             in_proj_bias=self.in_proj_bias,
                                                             bias_k=self.bias_k,
                                                             bias_v=self.bias_v,
                                                             add_zero_attn=self.add_zero_attn,
                                                             dropout_p=self.dropout,
                                                             out_proj_weight=self.out_proj.weight,
                                                             out_proj_bias=self.out_proj.bias,
                                                             training=self.training,
                                                             key_padding_mask=key_padding_mask,
                                                             need_weights=True,
                                                             attn_mask=attn_mask,
                                                             attn_bias=attn_bias,
                                                             sinusoidal_pos=sinusoidal_pos,
                                                             rotatry_values=rotatry_values,
                                                             local_attn_mask=local_attn_mask,
                                                             split_rotray_qk=self.split_rotray_qk)

    def forward(
            self,
            query, key, value,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            attn_bias=None,
            sinusoidal_pos: Optional[Tensor] = None,
            rotatry_values: bool = False,
            local_attn_mask: Optional[Tensor] = None,
            local_only=False,
            add_local_role: bool = False
    ):
        return multi_head_attention_forward(query, key, value,
                                            scale_factor=self.scale_factor,
                                            embed_dim_to_check=self.embed_dim,
                                            num_heads=self.num_heads,
                                            in_proj_weight=self.in_proj_weight,
                                            in_proj_bias=self.in_proj_bias,
                                            bias_k=self.bias_k,
                                            bias_v=self.bias_v,
                                            add_zero_attn=self.add_zero_attn,
                                            dropout_p=self.dropout,
                                            out_proj_weight=self.out_proj.weight,
                                            out_proj_bias=self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            need_weights=need_weights,
                                            attn_mask=attn_mask,
                                            attn_bias=attn_bias,
                                            sinusoidal_pos=sinusoidal_pos,
                                            rotatry_values=rotatry_values,
                                            local_attn_mask=local_attn_mask,
                                            split_rotray_qk=self.split_rotray_qk,
                                            local_only=local_only,
                                            add_local_role=add_local_role)


def apply_rotary_role_embeddings_v2(
        sinusoidal_pos, query_layer, key_layer, value_layer=None
):
    # MARK: KEY PART
    # https://kexue.fm/archives/8265
    # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
    ).reshape_as(query_layer)
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
    # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
    rotate_half_key_layer = torch.stack(
            [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
    ).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
    if value_layer is not None:
        # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
        rotate_half_value_layer = torch.stack(
                [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
        ).reshape_as(value_layer)
        value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
        return query_layer, key_layer, value_layer
    return query_layer, key_layer


def apply_rotary_role_embeddings(
        sinusoidal_pos, ent_repr
):
    # MARK: KEY PART
    # https://kexue.fm/archives/8265
    # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_repr = torch.stack(
            [-ent_repr[..., 1::2], ent_repr[..., ::2]], dim=-1
    ).reshape_as(ent_repr)
    query_layer = ent_repr * cos_pos + rotate_half_repr * sin_pos

    return query_layer


class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, dim = out.shape
        position_enc = np.array(
                [
                    [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                    for pos in [0, -1, 1]
                ]
        )  # 0 for CLS, -1 for head and 1 for tail
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,
        )
        return super().forward(positions)


class TransformerSeqEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            attn_scale_factor: int = 1,
            encoder_normalize_before: bool = False,
            split_rotray_qk: bool = False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                bias=True,
                scale_factor=attn_scale_factor,
                split_rotray_qk=split_rotray_qk
        )

        # new added
        self.normalize_before = encoder_normalize_before

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = layer_norm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = layer_norm(self.embedding_dim)

    def obtain_weights(self,
                       x: torch.Tensor,
                       self_attn_mask: torch.Tensor = None,
                       self_attn_padding_mask: torch.Tensor = None,
                       self_attn_bias: torch.Tensor = None,
                       sinusoidal_pos: Optional[torch.Tensor] = None,
                       rotatry_values: bool = False,
                       local_attn_mask: Optional[torch.Tensor] = None):
        residual = x
        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, global_weight, local_weight = self.self_attn.forward(
                x,
                x,
                x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=True,
                attn_mask=self_attn_mask,
                attn_bias=self_attn_bias,
                sinusoidal_pos=sinusoidal_pos,
                rotatry_values=rotatry_values,
                local_attn_mask=local_attn_mask
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x, global_weight, local_weight

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            self_attn_bias: torch.Tensor = None,
            sinusoidal_pos: Optional[torch.Tensor] = None,
            rotatry_values: bool = False,
            local_attn_mask: Optional[torch.Tensor] = None,
            local_only=False,
            add_local_role: bool = False
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, before=True)
        x, _ = self.self_attn.forward(
                x,
                x,
                x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                attn_bias=self_attn_bias,
                sinusoidal_pos=sinusoidal_pos,
                rotatry_values=rotatry_values,
                local_attn_mask=local_attn_mask,
                local_only=local_only,
                add_local_role=add_local_role
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.maybe_layer_norm(self.self_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        return x

    def maybe_layer_norm(self, layer_norm_, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm_(x)
        else:
            return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            num_encoder_layers: int = 3,
            embedding_dim: int = 100,
            role_dim: int = 32,
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
            device=torch.device('cpu')) -> None:

        super().__init__()
        self.dropout = dropout
        d_model = embedding_dim + role_dim
        self.embedding_dim = embedding_dim
        self.model_input_dim = d_model
        self.transformer_dim = num_attention_heads * d_model
        self.role_embeddings = nn.Embedding(3, embedding_dim=role_dim)

        role_indices = [0]
        for _ in range(shot):
            role_indices.append(1)
            role_indices.append(2)
        self.role_indices = torch.tensor(role_indices, dtype=torch.long,
                                         device=device)
        self.embed_scale = embed_scale
        self.cls_repr = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                     requires_grad=True)
        self.attn_scale_factor = 2
        self.num_attention_heads = num_attention_heads
        self.pos = nn.Embedding(shot + 1, d_model)
        self.pair_pos = nn.Embedding(shot + 2, self.model_input_dim)
        self.pos_q_linear = nn.Linear(self.model_input_dim, self.model_input_dim)
        self.pos_k_linear = nn.Linear(self.model_input_dim, self.model_input_dim)
        self.pos_scaling = float(
                self.model_input_dim * self.attn_scale_factor) ** -0.5
        self.pos_ln = layer_norm(self.model_input_dim)
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
                    )
                    for _ in range(num_encoder_layers)
                ]
        )

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

    # @pysnooper.snoop(custom_repr=(torch.Tensor, debug_tensor))
    def forward(
            self,
            entity_seq: torch.Tensor,
            last_state_only: bool = False,
            aggr_heads: Literal['keep', 'mean', 'sum', 'max'] = 'keep',
            unify_cls: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        :param entity_seq: shape [N, 2*K, dim]
        :param last_state_only:
        :param aggr_heads: 'keep' for cat all heads, 'mean' for mean heads, 'sum' for sum heads
        :param unify_cls: True for replace cls with unify value
        :return:
        """
        N, T, DIM = entity_seq.shape

        cls_repr_ = self.cls_repr.repeat(N, 1, 1)  # [N, 1, DIM]
        x_ = torch.cat([cls_repr_, entity_seq], dim=1)  # [N, 2*K+1, DIM]
        seq_len = x_.size(1)
        role_embeds = self.role_embeddings.forward(
                self.role_indices).unsqueeze(dim=0)  # [1, 2K+1, rDIM]
        role_embeds = role_embeds.repeat(N, 1, 1)  # [N, 2*K+1, DIM]

        # cat x and role
        x_ = torch.cat([x_, role_embeds], dim=-1)

        # B x T x C -> T x B x C
        x = x_.transpose(0, 1)

        # 0 is for other-to-cls 1 is for cls-to-other
        # Assume the input is ordered.
        # If your input token is permuted, you may need to update this accordingly
        weight = self.pos_ln(self.pair_pos.weight)
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

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1),
                                                        -1, -1, -1).reshape(-1, seq_len, seq_len)

        x = self.repeat_dim(x)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(x, self_attn_bias=abs_pos_bias)
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

        return inner_states, cls_repr


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
            rotary_value: bool = False,
            device=torch.device('cpu')) -> None:

        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.model_input_dim = embedding_dim
        self.transformer_dim = num_attention_heads * embedding_dim
        _role_embd_model = RoFormerSinusoidalPositionalEmbedding(3, self.embedding_dim)
        # MARK: 2 means head and tail, shape [2, DIM]
        self.role_vector = _role_embd_model.forward(torch.empty(1, 2).shape,
                                                    past_key_values_length=1).repeat(shot, 1)
        self.role_vector = self.role_vector.to(device)
        self.role_parameters = nn.Parameter(torch.randn(2, embedding_dim), requires_grad=True)

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
        self.pair_pos = nn.Embedding(shot + 2, self.model_input_dim)
        self.pos_q_linear = nn.Linear(self.model_input_dim, self.model_input_dim)
        self.pos_k_linear = nn.Linear(self.model_input_dim, self.model_input_dim)
        self.pos_scaling = float(
                self.model_input_dim * self.attn_scale_factor) ** -0.5
        self.pos_ln = layer_norm(self.model_input_dim)
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

    # @pysnooper.snoop(custom_repr=(torch.Tensor, debug_tensor))
    def forward(
            self,
            entity_seq: torch.Tensor,
            last_state_only: bool = False,
            aggr_heads: Literal['keep', 'mean', 'sum', 'max'] = 'keep',
            unify_cls: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, T, DIM = entity_seq.shape

        cls_repr_ = self.cls_repr.repeat(N, 1, 1)  # [N, 1, DIM]
        role_embeds = self.role_parameters.repeat(N, self.shot, 1)
        entity_seq_ = entity_seq * role_embeds
        x_ = torch.cat([cls_repr_, entity_seq_], dim=1)  # [N, 2*K+1, DIM]
        seq_len = x_.size(1)

        x = x_.transpose(0, 1)

        weight = self.pos_ln(self.pair_pos.weight)
        pos_len = weight.shape[0]
        pos_q = self.pos_q_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1) * self.pos_scaling
        pos_k = self.pos_k_linear(weight).view(pos_len, self.num_attention_heads, -1).transpose(
                0, 1)
        abs_pos_bias = torch.bmm(pos_q, pos_k.transpose(1, 2))

        cls_2_other = abs_pos_bias[:, 0, 0]
        other_2_cls = abs_pos_bias[:, 1, 1]
        abs_pos_bias: torch.Tensor = abs_pos_bias[:, 1:, 1:]
        pos_len_ = abs_pos_bias.shape[-1]
        abs_pos_bias = abs_pos_bias.unsqueeze(
                dim=-1).unsqueeze(dim=2).repeat(1, 1, 2, 1, 2).reshape(-1, 2 * pos_len_,
                                                                       2 * pos_len_)
        abs_pos_bias = abs_pos_bias[:, 1:, 1:]
        if unify_cls:
            abs_pos_bias[:, :, 0] = other_2_cls.view(-1, 1)
            abs_pos_bias[:, 0, :] = cls_2_other.view(-1, 1)

        abs_pos_bias = abs_pos_bias.unsqueeze(0).expand(x.size(1),
                                                        -1, -1, -1).reshape(-1, seq_len, seq_len)

        x = self.repeat_dim(x)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)
        for layer in self.layers:
            x = layer(x, self_attn_bias=abs_pos_bias)
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

        return inner_states, cls_repr
