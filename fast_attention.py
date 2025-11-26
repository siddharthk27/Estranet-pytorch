import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_projection_matrix(m, d, seed=0):
    """Generate orthogonal projection matrix for random features."""
    n_block = m // d
    block_list = []

    rng = torch.Generator().manual_seed(seed)
    cur_seed = seed

    for _ in range(n_block):
        block = torch.randn(d, d, generator=rng)
        q, _ = torch.linalg.qr(block)
        q = q.T
        block_list.append(q)
        cur_seed += 1
        rng.manual_seed(cur_seed)

    rem_rows = m - n_block * d
    if rem_rows > 0:
        block = torch.randn(d, d, generator=rng)
        q, _ = torch.linalg.qr(block)
        q = q.T
        block_list.append(q[:rem_rows])
        cur_seed += 1
        rng.manual_seed(cur_seed)

    proj_matrix = torch.vstack(block_list)

    multiplier = torch.norm(torch.randn(m, d, generator=rng), dim=1)

    return torch.mm(torch.diag(multiplier), proj_matrix)


def positive_kernel_transformation(data, is_query, projection_matrix=None,
                                   numerical_stabilizer=0.000001):
    """Apply positive random feature transformation."""
    data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    data = data_normalizer * data
    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)

    # [B, L, H, D] x [M, D]^T -> [B, L, H, M]
    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=-1, keepdim=True)

    if is_query:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(
                data_dash, dim=-1, keepdim=True)[0]) + numerical_stabilizer)
    else:
        # Reduce max over both dimension -3 and -1
        max_vals = torch.amax(data_dash, dim=(-3, -1), keepdim=True)
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - max_vals) + numerical_stabilizer)

    return data_dash


def fourier_kernel_transformation(data, projection_matrix):
    """Apply Fourier random feature transformation."""
    data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
    data = data_normalizer * data
    ratio = 1.0 / (projection_matrix.shape[0] ** 0.5)

    data_dash = torch.einsum("blhd,md->blhm", data, projection_matrix)
    data_sin = ratio * torch.sin(data_dash)
    data_cos = ratio * torch.cos(data_dash)

    return torch.cat([data_sin, data_cos], dim=-1)


def attention_numerator(qs, ks, vs):
    """Compute attention numerator using linear attention."""
    # qs, ks, vs: [L, B, H, M/D]
    kvs = torch.einsum("lbhm,lbhd->bhmd", ks, vs)
    return torch.einsum("lbhm,bhmd->lbhd", qs, kvs)


def attention_denominator(qs, ks):
    """Compute attention denominator for normalization."""
    # qs, ks: [L, B, H, M]
    all_ones = torch.ones(ks.shape[0], device=ks.device)
    ks_sum = torch.einsum("lbhm,l->bhm", ks, all_ones)
    return torch.einsum("lbhm,bhm->lbh", qs, ks_sum)


def linear_attention(value, query_pos_ft, key_pos_ft, projection_matrix=None,
                    feature_map_type='fourier', normalize_attn=False):
    """Compute linear attention using random features."""
    if feature_map_type == 'fourier':
        query_prime = fourier_kernel_transformation(query_pos_ft, projection_matrix)
        key_prime = fourier_kernel_transformation(key_pos_ft, projection_matrix)
    elif feature_map_type == 'positive':
        query_prime = positive_kernel_transformation(query_pos_ft, True, projection_matrix)
        key_prime = positive_kernel_transformation(key_pos_ft, False, projection_matrix)
    else:
        raise ValueError("feature_type must be in ['fourier', 'positive']")

    # Transpose to [L, B, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3)
    key_prime = key_prime.permute(1, 0, 2, 3)
    value = value.permute(1, 0, 2, 3)

    av_attention = attention_numerator(query_prime, key_prime, value)
    av_attention = av_attention.permute(1, 0, 2, 3)

    if normalize_attn:
        attention_normalizer = attention_denominator(query_prime, key_prime)
        attention_normalizer = attention_normalizer.permute(1, 0, 2)
        attention_normalizer = attention_normalizer.unsqueeze(-1)
        av_attention = av_attention / attention_normalizer

    return [av_attention, query_prime, key_prime]


class SelfAttention(nn.Module):
    """GaussiP attention layer for shift-invariant attention."""

    def __init__(self, d_model, d_head, n_head, attention_dropout,
                 feature_map_type='fourier', normalize_attn=False,
                 d_kernel_map=128, head_init_range=(0, 1)):
        super(SelfAttention, self).__init__()

        self.d_model = d_model
        self.size_per_head = d_head
        self.n_head = n_head
        self.attention_dropout = attention_dropout
        self.d_kernel_map = d_kernel_map
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.head_init_range = head_init_range

        # Xavier/Glorot initialization
        def glorot_initializer(fan_in, fan_out):
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            return limit

        limit = glorot_initializer(self.d_model, self.n_head * self.size_per_head)

        self.value_weight = nn.Parameter(
            torch.empty(self.d_model, self.n_head, self.size_per_head).uniform_(-limit, limit)
        )
        self.pos_ft_weight = nn.Parameter(
            torch.empty(self.d_model, self.n_head, self.size_per_head).uniform_(-limit, limit),
            requires_grad=False
        )
        self.pos_ft_scale = nn.Parameter(torch.ones(1, 1, self.n_head, 1))

        # Initialize head positions
        head_left = self.head_init_range[0]
        head_right = self.head_init_range[1]
        head_range = head_right - head_left
        head_pos = torch.arange(
            head_left + head_range / (2.0 * self.n_head),
            head_right,
            head_range / self.n_head
        )
        self.pos_ft_offsets = nn.Parameter(head_pos.view(1, 1, self.n_head, 1))

        output_limit = glorot_initializer(self.n_head * self.size_per_head, self.d_model)
        self.output_weight = nn.Parameter(
            torch.empty(self.n_head * self.size_per_head, self.d_model).uniform_(-output_limit, output_limit)
        )

        self.output_dropout = nn.Dropout(self.attention_dropout)

        # Generate projection matrix
        seed = np.random.randint(1e8)
        projection_matrix = gen_projection_matrix(self.d_kernel_map, self.size_per_head, seed=seed)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, source_input, pos_ft, pos_ft_slopes, training=True):
        """
        Args:
            source_input: [B, L, D]
            pos_ft: [B, L, D]
            pos_ft_slopes: [B, L, D]
        """
        value = torch.einsum("bnm,mhd->bnhd", source_input, self.value_weight)
        pos_ft_projected = torch.einsum("bnm,mhd->bnhd", pos_ft, self.pos_ft_weight)
        pos_ft_slopes_projected = torch.einsum("bnm,mhd->bnhd", pos_ft_slopes, self.pos_ft_weight)

        query_pos_ft = self.pos_ft_scale * pos_ft_projected
        slope_pos = self.pos_ft_scale * pos_ft_slopes_projected
        key_pos_ft = query_pos_ft + self.pos_ft_offsets * slope_pos

        attention_outputs = linear_attention(
            value, query_pos_ft, key_pos_ft,
            self.projection_matrix,
            self.feature_map_type,
            self.normalize_attn
        )

        bsz, slen = attention_outputs[0].shape[:2]

        # Scale by normalized slope norms
        norms = torch.norm(pos_ft_slopes_projected, dim=-1, keepdim=True) / float(slen)
        attention_outputs[0] = norms * attention_outputs[0]

        attention_outputs[0] = attention_outputs[0].reshape(bsz, slen, -1)
        attention_outputs[0] = torch.einsum("bnm,md->bnd", attention_outputs[0], self.output_weight)

        if training:
            attention_outputs[0] = self.output_dropout(attention_outputs[0])

        return attention_outputs
