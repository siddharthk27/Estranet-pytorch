import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_attention import SelfAttention
from normalization import LayerCentering


class PositionalFeature(nn.Module):
    """Generate positional features for shift-invariant attention."""

    def __init__(self, d_feature, beta_hat_2):
        super(PositionalFeature, self).__init__()

        slopes = torch.arange(d_feature, 0, -4.0) / d_feature
        slopes = slopes * beta_hat_2
        self.register_buffer('slopes', slopes)

    def forward(self, slen, bsz=None):
        """
        Args:
            slen: sequence length (int or Tensor)
            bsz: batch size (optional)

        Returns:
            pos_feature: [B, L, 4*D] or [1, L, 4*D]
            pos_feature_slopes: [B, L, 4*D] or [1, L, 4*D]
        """
        device = self.slopes.device

        # Handle both int and Tensor slen
        if isinstance(slen, int):
            slen_val = slen
        else:
            slen_val = slen.item() if slen.numel() == 1 else int(slen)

        pos_seq = torch.arange(0, slen_val, 1.0, device=device)

        denom = max(float(slen_val - 1), 1.0)
        normalized_slopes = (1.0 / denom) * self.slopes

        # Create positional encodings
        forward = torch.outer(pos_seq, normalized_slopes)
        backward = torch.flip(forward, dims=[0])
        neg_forward = -forward
        neg_backward = -backward

        pos_feature = torch.cat([forward, backward, neg_forward, neg_backward], dim=-1)

        pos_feature_slopes = torch.cat([
            normalized_slopes,
            -normalized_slopes,
            -normalized_slopes,
            normalized_slopes
        ], dim=0)

        pos_feature_slopes = float(slen_val - 1) * pos_feature_slopes.unsqueeze(0)

        # Add batch dimension
        pos_feature = pos_feature.unsqueeze(0)

        if bsz is not None:
            pos_feature = pos_feature.repeat(bsz, 1, 1)
            pos_feature_slopes = pos_feature_slopes.repeat(bsz, 1, 1)

        return pos_feature, pos_feature_slopes


class PositionwiseFF(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.layer_1 = nn.Linear(d_model, d_inner)
        self.drop_1 = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(d_inner, d_model)
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, inp, training=True):
        core_out = F.relu(self.layer_1(inp))
        if training:
            core_out = self.drop_1(core_out)

        core_out = self.layer_2(core_out)
        if training:
            core_out = self.drop_2(core_out)

        return [core_out]


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward."""

    def __init__(self, n_head, d_head, d_model, d_inner, dropout,
                 feature_map_type, normalize_attn, d_kernel_map,
                 model_normalization, head_init_range):
        super(TransformerLayer, self).__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.feature_map_type = feature_map_type
        self.normalize_attn = normalize_attn
        self.d_kernel_map = d_kernel_map
        self.model_normalization = model_normalization
        self.head_init_range = head_init_range

        self.self_attn = SelfAttention(
            d_model=self.d_model,
            d_head=self.d_head,
            n_head=self.n_head,
            attention_dropout=self.dropout,
            feature_map_type=self.feature_map_type,
            normalize_attn=self.normalize_attn,
            d_kernel_map=self.d_kernel_map,
            head_init_range=self.head_init_range
        )

        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout
        )

        assert self.model_normalization in ['preLC', 'postLC', 'none'], \
            "model_normalization must be one of 'preLC', 'postLC' or 'none'"

        if self.model_normalization in ['preLC', 'postLC']:
            self.lc1 = LayerCentering(self.d_model)
            self.lc2 = LayerCentering(self.d_model)

    def forward(self, inputs, training=True):
        inp, pos_ft, pos_ft_slopes = inputs

        # Self-attention with optional pre-normalization
        if self.model_normalization == 'preLC':
            attn_in = self.lc1(inp)
        else:
            attn_in = inp

        attn_outputs = self.self_attn(attn_in, pos_ft, pos_ft_slopes, training=training)
        attn_outputs[0] = attn_outputs[0] + inp

        if self.model_normalization == 'postLC':
            attn_outputs[0] = self.lc1(attn_outputs[0])

        # Feed-forward with optional pre-normalization
        if self.model_normalization == 'preLC':
            ff_in = self.lc2(attn_outputs[0])
        else:
            ff_in = attn_outputs[0]

        ff_output = self.pos_ff(ff_in, training=training)
        ff_output[0] = ff_output[0] + attn_outputs[0]

        if self.model_normalization == 'postLC':
            ff_output[0] = self.lc2(ff_output[0])

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs


class SoftmaxAttention(nn.Module):
    """Softmax attention for final pooling layer."""

    def __init__(self, d_model, n_head, d_head):
        super(SoftmaxAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.q_heads = nn.Parameter(torch.randn(d_head, n_head))
        self.k_net = nn.Linear(d_model, d_head * n_head)
        self.v_net = nn.Linear(d_model, d_head * n_head)

        self.scale = 1.0 / (self.d_head ** 0.5)
        self.register_buffer('softmax_attn_smoothing', torch.tensor(0.0))

    def forward(self, inp, softmax_attn_smoothing=1.0, training=True):
        bsz, slen = inp.shape[:2]

        if training:
            self.softmax_attn_smoothing.fill_(softmax_attn_smoothing)

        k_head = self.k_net(inp)
        v_head = self.v_net(inp)

        k_head = k_head.reshape(bsz, slen, self.d_head, self.n_head)
        v_head = v_head.reshape(bsz, slen, self.d_head, self.n_head)

        attn_score = torch.einsum("bndh,dh->bnh", k_head, self.q_heads)
        attn_score = attn_score * self.scale * self.softmax_attn_smoothing

        attn_prob = F.softmax(attn_score, dim=1)

        attn_out = torch.einsum("bndh,bnh->bnhd", v_head, attn_prob)
        attn_out = attn_out.reshape(bsz, slen, -1)

        return attn_out, attn_score


class Transformer(nn.Module):
    """EstraNet: Efficient Shift-Invariant Transformer Network."""

    def __init__(self, n_layer, d_model, d_head, n_head, d_inner,
                 d_head_softmax, n_head_softmax, dropout, n_classes,
                 conv_kernel_size, n_conv_layer, pool_size, d_kernel_map,
                 beta_hat_2, model_normalization, head_initialization='forward',
                 softmax_attn=True, output_attn=False):
        super(Transformer, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_inner = d_inner
        self.d_head_softmax = d_head_softmax
        self.n_head_softmax = n_head_softmax
        self.feature_map_type = 'fourier'
        self.normalize_attn = False
        self.d_kernel_map = d_kernel_map
        self.beta_hat_2 = beta_hat_2
        self.model_normalization = model_normalization
        self.head_initialization = head_initialization
        self.softmax_attn = softmax_attn
        self.dropout = dropout
        self.n_classes = n_classes
        self.conv_kernel_size = conv_kernel_size
        self.n_conv_layer = n_conv_layer
        self.pool_size = pool_size
        self.output_attn = output_attn

        # Convolutional layers
        conv_filters = [min(8 * 2**i, self.d_model) for i in range(self.n_conv_layer - 1)] + [self.d_model]

        self.conv_layers = nn.ModuleList()
        self.relu_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        in_channels = 1
        for l in range(self.n_conv_layer):
            ks = 11 if l == 0 else self.conv_kernel_size
            self.conv_layers.append(nn.Conv1d(in_channels, conv_filters[l], ks, padding=ks//2))
            self.relu_layers.append(nn.ReLU())
            self.pool_layers.append(nn.AvgPool1d(self.pool_size, self.pool_size))
            in_channels = conv_filters[l]

        # Positional features
        self.pos_feature = PositionalFeature(self.d_model, self.beta_hat_2)

        # Initialize head ranges based on initialization type
        head_init_ranges = []
        if self.head_initialization == 'forward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((0.0, 0.5))
                else:
                    head_init_ranges.append((0.0, 1.0))
        elif self.head_initialization == 'backward':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.0))
                else:
                    head_init_ranges.append((-1.0, 0.0))
        elif self.head_initialization == 'symmetric':
            for i in range(self.n_layer):
                if i == 0:
                    head_init_ranges.append((-0.5, 0.5))
                else:
                    head_init_ranges.append((-1.0, 1.0))
        else:
            raise ValueError("head_initialization must be one of ['forward', 'backward', 'symmetric']")

        # Transformer layers
        self.tran_layers = nn.ModuleList()
        for i in range(self.n_layer):
            self.tran_layers.append(
                TransformerLayer(
                    n_head=self.n_head,
                    d_head=self.d_head,
                    d_model=self.d_model,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    feature_map_type=self.feature_map_type,
                    normalize_attn=self.normalize_attn,
                    d_kernel_map=self.d_kernel_map,
                    model_normalization=self.model_normalization,
                    head_init_range=head_init_ranges[i]
                )
            )

        self.out_dropout = nn.Dropout(dropout)

        # Output layers
        if self.softmax_attn:
            self.out_attn = SoftmaxAttention(
                d_model=self.d_model,
                n_head=self.n_head_softmax,
                d_head=self.d_head_softmax
            )

        self.fc_output = nn.Linear(self.d_model, self.n_classes)

    def forward(self, inp, softmax_attn_smoothing=1.0, training=True):
        """
        Args:
            inp: [B, L] input traces
            softmax_attn_smoothing: smoothing parameter for softmax attention
            training: whether in training mode

        Returns:
            List containing scores and optionally attention outputs
        """
        # Convert [B, L] to [B, 1, L] for Conv1d
        inp = inp.unsqueeze(1)

        # Apply convolutional blocks
        for l in range(self.n_conv_layer):
            inp = self.conv_layers[l](inp)
            inp = self.relu_layers[l](inp)
            inp = self.pool_layers[l](inp)

        # Transpose to [B, L, C]
        inp = inp.permute(0, 2, 1)

        bsz, slen = inp.shape[:2]

        # Generate positional features
        pos_ft, pos_ft_slopes = self.pos_feature(slen, bsz)
        pos_ft = pos_ft.to(inp.device)
        pos_ft_slopes = pos_ft_slopes.to(inp.device)

        # Apply transformer layers
        core_out = inp
        out_list = []
        for i, layer in enumerate(self.tran_layers):
            all_out = layer([core_out, pos_ft, pos_ft_slopes], training=training)
            core_out = all_out[0]
            out_list.append(all_out[1:])

        if training:
            core_out = self.out_dropout(core_out)

        # Apply softmax attention or mean pooling
        if self.softmax_attn:
            core_out, softmax_attn_score = self.out_attn(
                core_out, softmax_attn_smoothing, training=training
            )
        else:
            softmax_attn_score = None

        output = torch.mean(core_out, dim=1)

        # Get final class scores
        scores = self.fc_output(output)

        # Transpose attention outputs
        for i in range(len(out_list)):
            for j in range(len(out_list[i])):
                out_list[i][j] = out_list[i][j].permute(1, 0, 2, 3)

        if self.output_attn:
            return [scores, out_list, softmax_attn_score]
        else:
            return [scores]
