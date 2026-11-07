"""Transformer and DNA building blocks of ASAHI.

Tensors are ``[B, L, S, H]``: batch, CpGs (the sliding-window axis), libraries in the
slice (the full-attention axis), hidden size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNAEncoder(nn.Module):
    """Per-CpG DNA encoder: one ``RF``-bp window of base codes -> a 128-d vector.

    ``RF`` sets the width of the final ``Linear``, so it is fixed for a checkpoint.
    """

    def __init__(self, dropout=0, RF=1001):
        super().__init__()
        self.RF = RF
        self.hlen = (((RF - 10) // 4) - 2) // 2
        self.embed = nn.Embedding(16, 4)
        self.CNN = nn.Sequential(nn.Conv1d(4, 128, 11), nn.ReLU(inplace=True), nn.MaxPool1d(4),
                                 nn.Conv1d(128, 256, 3), nn.ReLU(inplace=True), nn.MaxPool1d(2))
        self.lin = nn.Sequential(nn.Linear(256 * self.hlen, 128), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout))

    def forward(self, x):
        """``[N, RF]`` integer base codes -> ``[N, 128]``."""
        x = self.embed(x).permute(0, 2, 1)
        x = self.CNN(x).reshape(-1, 256 * self.hlen)
        return self.lin(x)


class RelPositionalWindowEmbedding(nn.Module):
    """Transformer-XL relative positions for sliding windows, from real base-pair distances.

    ``pos [B, L]`` -> ``[B, W, L, H]``: one sinusoidal encoding per CpG and window offset.
    """

    def __init__(self, embedding_size, window=21):
        super().__init__()
        self.embedding_size = embedding_size
        self.inv_freq = 1 / (10000 ** (torch.arange(0., embedding_size, 2) / embedding_size))
        self.w = int((window - 1) / 2)

    def forward(self, pos, x):
        seqlen = pos.shape[-1]
        relpos = F.pad(pos, (self.w,) * 2).unfold(-1, seqlen, 1) - pos.unsqueeze(-2)
        sizes = relpos.size()
        sinusoid_inp = self.inv_freq.type_as(x) * relpos.unsqueeze(-1)
        return torch.stack((sinusoid_inp.sin(), sinusoid_inp.cos()),
                           dim=-1).view(*sizes, self.embedding_size)


class SlidingWindowWithinSampleAttention(nn.Module):
    """Each library attends along its own CpGs, within a window of ``window`` CpGs."""

    def __init__(self, in_features, head_dim, n_head, out_features, window=21, dropout=0.20):
        super().__init__()
        self.qkv_lin = nn.Linear(in_features, head_dim * n_head * 3)
        self.embed_lin = nn.Linear(in_features, head_dim * n_head)
        self.out_lin = nn.Linear(head_dim * n_head, out_features)
        self.bias_r_w = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        self.bias_r_r = self._init_bias(nn.Parameter(torch.Tensor(n_head, head_dim)))
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.w = int((window - 1) / 2)
        self.h = head_dim
        self.nh = n_head

    def forward(self, x, r):
        shp = x.shape
        ndim = len(shp)
        bsz, seqlen = shp[0], shp[1]
        n_reps = shp[2:-1].numel()
        windows = 2 * self.w + 1

        q, k, v = torch.split(self.qkv_lin(x), self.h * self.nh, dim=-1)
        q = q.transpose(1, 2).reshape(-1, seqlen, self.nh, self.h) * (self.h ** -0.5)
        k = k.transpose(1, 2).reshape(-1, seqlen, self.nh, self.h)
        v = v.transpose(1, 2).reshape(-1, seqlen, self.nh, self.h)

        # All rows of a slice share one genomic window, so ``r`` has no sample axis
        # and is broadcast over it in the BD term below.
        r = self.embed_lin(r).view(bsz, windows, seqlen, self.nh, self.h)

        k = F.pad(k, (0,) * (ndim * 2 - 4) + (self.w,) * 2).unfold(1, seqlen, 1)
        v = F.pad(v, (0,) * (ndim * 2 - 4) + (self.w,) * 2).unfold(1, seqlen, 1)

        q_k = q + self.bias_r_w
        q_r = q + self.bias_r_r

        AC = torch.einsum('bs...nh,bwnhs->bsn...w', q_k, k)
        BD = torch.einsum('bcsnh,bwsnh->bcsnw',
                          q_r.reshape(bsz, n_reps, seqlen, self.nh, self.h), r)
        BD = BD.reshape(-1, seqlen, self.nh, windows)
        A = AC + BD

        # Offsets that fall off either end of the patch.
        mask = torch.zeros(shp[1:-2], device=k.device).bool()
        mask = F.pad(mask, (0,) * (ndim * 2 - 8) + (self.w,) * 2, value=True).unfold(0, seqlen, 1).T
        for _ in range(ndim - 3):
            mask = mask.unsqueeze(1)
        A.masked_fill_(mask, -torch.finfo(A.dtype).max)

        A = self.dropout(self.softmax(A))
        z = torch.einsum('bsn...w,bwnhs->bs...nh', A, v)
        z = z.view(bsz, n_reps, seqlen, -1).transpose(1, 2)
        return self.out_lin(z)

    def _init_bias(self, bias):
        bound = 1 / bias.size(1) ** 0.5
        return nn.init.uniform_(bias, -bound, bound)


class BetweenSampleAttention(nn.Module):
    """At each CpG, the libraries of a slice attend to each other."""

    def __init__(self, in_features, head_dim, n_head, out_features, dropout=0.20):
        super().__init__()
        self.qkv_lin = nn.Linear(in_features, head_dim * n_head * 3)
        self.out_lin = nn.Linear(head_dim * n_head, out_features)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-2)
        self.h = head_dim
        self.nh = n_head

    def forward(self, x, r):
        """``r`` is unused: attention here is at one position, so no distance applies."""
        shp = x.shape
        bsz, seqlen = shp[0], shp[1]
        n_reps = shp[2:-1].numel()
        q, k, v = torch.split(self.qkv_lin(x), self.h * self.nh, dim=-1)
        q = q.view(-1, n_reps, self.nh, self.h) * (self.h ** -0.5)
        k = k.view(-1, n_reps, self.nh, self.h)
        v = v.view(-1, n_reps, self.nh, self.h)
        A = self.dropout(self.softmax(torch.einsum('b q n h, b k n h -> b q k n', q, k)))
        z = torch.einsum('b q k n, b k n h -> b q n h', A, v)
        return self.out_lin(z.reshape(bsz, seqlen, n_reps, -1))


class AxialTransformerLayer(nn.Module):
    """Sliding-window attention, between-library attention, feed-forward; each residual + LayerNorm."""

    def __init__(self, hidden_dim, head_dim, n_head, ff_dim, window=21, dropout=0.20, activation='relu'):
        super().__init__()
        act = nn.ReLU() if activation.lower() == 'relu' else nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn_list = nn.ModuleList([
            SlidingWindowWithinSampleAttention(hidden_dim, head_dim, n_head, hidden_dim,
                                               window=window, dropout=dropout),
            BetweenSampleAttention(hidden_dim, head_dim, n_head, hidden_dim, dropout=dropout)])
        self.norm_attn_list = nn.ModuleList([nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)])
        self.pos_emb = RelPositionalWindowEmbedding(hidden_dim, window=window)
        self.ff = nn.Sequential(nn.Linear(hidden_dim, ff_dim), act, nn.Linear(ff_dim, hidden_dim))

    def forward(self, x_pos):
        x, pos = x_pos
        r = self.pos_emb(pos, x)
        for attn, norm in zip(self.attn_list, self.norm_attn_list):
            x = norm(attn(x, r) + x)
        return self.norm2(self.ff(x) + x), pos
