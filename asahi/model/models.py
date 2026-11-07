"""The ASAHI model."""

import types

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint

from asahi.model.blocks import AxialTransformerLayer, DNAEncoder


class CpGEmbedder(nn.Module):
    """Linear embedding of betas; unknown cells get a single learned mask vector."""

    def __init__(self, hidden_size):
        super().__init__()
        self.embed_linear = nn.Linear(1, hidden_size)
        self.mask_embed = nn.Parameter(torch.Tensor(1, hidden_size))
        bound = 1 / hidden_size ** 0.5
        nn.init.uniform_(self.mask_embed, -bound, bound)

    def forward(self, betas, input_mask):
        """``[B, L, S]`` -> ``[B, L, S, H]``."""
        z = self.embed_linear(betas.unsqueeze(-1).to(self.embed_linear.weight.dtype))
        if input_mask.any():
            z = z.clone()
            z[input_mask] = self.mask_embed.to(z.dtype)
        return z


class ASAHI(nn.Module):
    """Denoise and impute low-depth methylomes from slices.

    Input is a slice of betas ``(B, S, L)``: the same ``L`` CpGs in ``S`` libraries.
    Each cell's input embedding concatenates

    - its beta, or the learned mask vector if unknown;
    - the DNA around the CpG: a 1001-bp window through a CNN, once per CpG and shared
      by every row;
    - the depth behind that beta: a learned token per integer depth up to ``depth_cap``,
      plus a scaled ``log1p`` that keeps the ordering and the tail above the cap.

    A stack of axial transformer layers then mixes along CpGs and across libraries.
    The main head predicts each cell's full-depth beta. Two auxiliary heads, used only
    in training, predict the local mean and the training-cohort mean at each cell.

    Parameters follow the paper's values by default.
    """

    def __init__(self, cpg_embed_size=32, transf_hsz=64, n_transformers=4, n_heads=8,
                 head_dim=8, window=21, transf_do=0.20, act='relu',
                 dna_embed_size=32, dna_window=1001, dna_conv_do=0.2, dna_chunks=16,
                 depth_embed_size=8, depth_cap=32,
                 aux_local_bp=500, aux_local_weight=0.25, aux_cross_weight=0.25,
                 aux_warmup_epochs=10, lr=5e-4, lr_decay_factor=0.998719, warmup_steps=1000):
        super().__init__()
        if dna_window % 2 == 0:
            raise ValueError("dna_window must be odd: the CpG sits at its centre")
        self.hparams = types.SimpleNamespace(
            cpg_embed_size=cpg_embed_size, transf_hsz=transf_hsz, n_transformers=n_transformers,
            n_heads=n_heads, head_dim=head_dim, window=window, transf_do=transf_do, act=act,
            dna_embed_size=dna_embed_size, dna_window=dna_window, dna_conv_do=dna_conv_do,
            dna_chunks=dna_chunks, depth_embed_size=depth_embed_size, depth_cap=depth_cap,
            aux_local_bp=aux_local_bp, aux_local_weight=aux_local_weight,
            aux_cross_weight=aux_cross_weight, aux_warmup_epochs=aux_warmup_epochs,
            lr=lr, lr_decay_factor=lr_decay_factor, warmup_steps=warmup_steps)
        self.dna_window = dna_window
        self.dna_chunks = dna_chunks
        self._depth_log_scale = float(np.log1p(depth_cap + 1))

        # Creation order fixes the initial weights for a given seed.
        self.CpG_embed = CpGEmbedder(cpg_embed_size)
        self.combine_embeds = nn.Sequential(nn.Linear(cpg_embed_size, transf_hsz), nn.ReLU())
        self.transformer = nn.Sequential(*[
            AxialTransformerLayer(transf_hsz, head_dim, n_heads, transf_hsz * 4, dropout=transf_do,
                                  window=window, activation=act)
            for _ in range(n_transformers)])
        self.output_head = nn.Linear(transf_hsz, 1)
        self.dna_cnn = DNAEncoder(dropout=dna_conv_do, RF=dna_window)
        self.dna_proj = nn.Linear(128, dna_embed_size)
        self.combine_embeds = nn.Sequential(
            nn.Linear(cpg_embed_size + dna_embed_size, transf_hsz), nn.ReLU())
        self.aux_local_head = nn.Linear(transf_hsz, 1)
        self.aux_cross_head = nn.Linear(transf_hsz, 1)
        self.depth_embed = nn.Embedding(depth_cap + 1, depth_embed_size)
        self.combine_embeds = nn.Sequential(
            nn.Linear(cpg_embed_size + dna_embed_size + depth_embed_size + 1, transf_hsz), nn.ReLU())

        # Reference data, attached after construction and kept out of checkpoints.
        self.register_buffer("dna_seq", torch.zeros(0, dtype=torch.uint8), persistent=False)
        self.register_buffer("cohort_mean", None, persistent=False)

    def set_genome(self, seq):
        """Attach the chromosome's encoded sequence. Needed for training and inference."""
        self.dna_seq = torch.as_tensor(seq, dtype=torch.uint8).to(self.dna_seq.device)
        return self

    def set_cohort_mean(self, vec):
        """Attach the per-CpG training-cohort mean. Needed for training only."""
        self.cohort_mean = torch.as_tensor(vec, dtype=torch.float32)
        return self

    def _dna_embed(self, pos):
        """``pos (B, L)`` -> ``(B, L, dna_embed_size)``.

        Windows are gathered on the device from the attached sequence; bases outside
        the chromosome read as N. The CNN is chunked and gradient-checkpointed because
        it sees ``B * L`` windows per step.
        """
        if self.dna_seq.numel() == 0:
            raise RuntimeError("no genome attached; call model.set_genome(seq) first")
        B, L = pos.shape
        w = (self.dna_window - 1) // 2
        idx = pos.unsqueeze(-1) + torch.arange(-w, w + 1, device=pos.device, dtype=pos.dtype)
        inside = (idx >= 0) & (idx < self.dna_seq.shape[0])
        win = self.dna_seq[idx.clamp(0, self.dna_seq.shape[0] - 1)]
        flat = torch.where(inside, win, win.new_full((), 4)).reshape(-1, self.dna_window).int()

        def run(chunk):
            return self.dna_proj(self.dna_cnn(chunk))

        parts = []
        for c in torch.chunk(flat, max(1, min(self.dna_chunks, flat.shape[0])), dim=0):
            if self.training and torch.is_grad_enabled():
                parts.append(torch.utils.checkpoint.checkpoint(run, c, use_reentrant=False))
            else:
                parts.append(run(c))
        return torch.cat(parts, 0).view(B, L, -1)

    def _depth_feats(self, depth, dtype):
        """``(B, S, L)`` read depth -> ``(B, L, S, depth_embed_size + 1)``. 0 = missing or hidden."""
        d = depth.transpose(1, 2)
        cap = self.hparams.depth_cap
        d = torch.nan_to_num(d, nan=0.0, posinf=float(cap), neginf=0.0)
        tok = self.depth_embed(d.clamp(0, cap).round().long())
        sc = torch.log1p(d.to(dtype).clamp_min(0.0)) / self._depth_log_scale
        return torch.cat((tok, sc.unsqueeze(-1)), dim=-1)

    def forward(self, sparse, input_mask, pos, depth, return_aux=False):
        """Predict every cell.

        Parameters
        ----------
        sparse : (B, S, L) float    input betas, NaN-free
        input_mask : (B, S, L) bool True where the beta is unknown to the model
        pos : (B, L) int64          genomic coordinate of each CpG
        depth : (B, S, L) float     read depth of the input betas; 0 where unknown

        Returns ``(B, S, L)`` predictions; with ``return_aux``, also the auxiliary heads.
        """
        betas = sparse.transpose(1, 2)
        mask = input_mask.transpose(1, 2)
        cpg = self.CpG_embed(betas, mask)
        dna = self._dna_embed(pos).unsqueeze(-2).expand(-1, -1, betas.shape[-1], -1)
        dep = self._depth_feats(depth, cpg.dtype)
        x = self.combine_embeds(torch.cat((cpg, dna, dep), dim=-1))
        x, _ = self.transformer((x, pos))
        out = self.output_head(x).squeeze(-1).transpose(1, 2)
        if not return_aux:
            return out
        return out, {"local": self.aux_local_head(x).squeeze(-1).transpose(1, 2),
                     "cross": self.aux_cross_head(x).squeeze(-1).transpose(1, 2)}

    def save_checkpoint(self, path):
        """Save ``{hparams, state_dict}``."""
        torch.save({"hparams": vars(self.hparams), "state_dict": self.state_dict()}, path)


def load_checkpoint(path, map_location="cpu"):
    """Rebuild a model from a checkpoint. Attach a genome before calling it."""
    ckpt = torch.load(path, map_location=map_location)
    model = ASAHI(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
