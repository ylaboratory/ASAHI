"""Objective and training loop: masked + read-thinned MSE, auxiliary heads, early stopping."""

import os
import sys
import time

import torch
import torch.nn.functional as F


def _masked_mse(pred, true, m):
    """``(loss, n)`` over the selected cells; ``(None, 0)`` if none."""
    n = int(m.sum().item())
    if n == 0:
        return None, 0
    return F.mse_loss(pred[m], true[m]), n


def local_mean_target(true, cpg_mask, pos, bp):
    """Mean full-depth beta of the same library's other CpGs within ``bp`` base pairs.

    Leave-one-out, so a cell is never part of its own target. NaN where no neighbour
    was observed. Sums are in float64 because they are differences of large cumulative sums.
    """
    obs = (~cpg_mask).to(true.dtype)
    vals = true * obs
    B, S, L = true.shape
    lo = torch.searchsorted(pos, pos - bp).unsqueeze(1).expand(B, S, L)
    hi = torch.searchsorted(pos, pos + bp, right=True).unsqueeze(1).expand(B, S, L)
    z = torch.zeros(B, S, 1, dtype=torch.float64, device=true.device)
    csum = torch.cat([z, vals.to(torch.float64).cumsum(dim=2)], dim=2)
    ccnt = torch.cat([z, obs.to(torch.float64).cumsum(dim=2)], dim=2)
    ssum = csum.gather(2, hi) - csum.gather(2, lo) - vals.to(torch.float64)
    scnt = ccnt.gather(2, hi) - ccnt.gather(2, lo) - obs.to(torch.float64)
    loc = torch.where(scnt > 0, ssum / scnt.clamp_min(1.0), torch.full_like(ssum, float("nan")))
    return loc.to(true.dtype)


def cohort_mean_target(model, patch_id, true):
    """The training-cohort per-CpG mean, for the cells of each slice. NaN where undefined."""
    if model.cohort_mean is None:
        raise RuntimeError("no cohort mean attached; call model.set_cohort_mean(vec) first")
    B, S, L = true.shape
    pid = patch_id.to(true.device).long()
    idx = (pid[:, None] * L + torch.arange(L, device=true.device)[None, :]).clamp_(
        max=model.cohort_mean.numel() - 1)
    return model.cohort_mean.to(true.device)[idx][:, None, :].expand(B, S, L).contiguous()


def compute_loss(pred, true, eval_mask, thin_mask, aux):
    """Pooled MSE over hidden and thinned cells, plus the weighted auxiliary MSEs.

    ``aux`` is ``{name: (prediction, target, weight)}``, scored on the same cells as the
    main loss wherever the target is defined.

    Returns ``(total, parts)``; ``total`` is ``None`` when nothing is supervised. ``parts``
    has ``main``, ``impute`` and ``upscale``, each ``(loss, n)``: ``impute`` is the hidden
    cells and ``upscale`` the read-thinned ones.
    """
    loss_i, n_i = _masked_mse(pred, true, eval_mask)
    loss_t, n_t = _masked_mse(pred, true, thin_mask)
    parts = {"impute": (loss_i, n_i), "upscale": (loss_t, n_t)}
    if n_i == 0 and n_t == 0:
        return None, parts
    if n_t == 0:
        main = loss_i
    elif n_i == 0:
        main = loss_t
    else:
        main = (n_i * loss_i + n_t * loss_t) / (n_i + n_t)
    parts["main"] = (main, n_i + n_t)

    total = main
    sup = eval_mask | thin_mask
    for name, (apred, atgt, weight) in aux.items():
        l_a, n_a = _masked_mse(apred, atgt, sup & torch.isfinite(atgt))
        parts[f"aux_{name}"] = (l_a, n_a)
        if n_a and float(weight):
            total = total + float(weight) * l_a
    return total, parts


def _forward_batch(model, batch, device, aux_scale):
    true = batch["true"].to(device, non_blocking=True)
    sparse = batch["sparse"].to(device, non_blocking=True)
    cpg_mask = batch["cpg_mask"].to(device, non_blocking=True)
    eval_mask = batch["eval_mask"].to(device, non_blocking=True)
    pos = batch["pos"].to(device, non_blocking=True)
    thin_mask = batch["thin_mask"].to(device, non_blocking=True)
    thin_drop = batch["thin_drop_mask"].to(device, non_blocking=True)
    depth = batch["cov_sparse"].to(device, non_blocking=True)

    # Thinned cells with reads left are deliberately not flagged: at inference a real
    # low-depth library carries no such flag.
    input_mask = cpg_mask | eval_mask | thin_drop
    pred, aux_pred = model(sparse, input_mask, pos, depth, return_aux=True)

    hp = model.hparams
    aux = {"local": (aux_pred["local"], local_mean_target(true, cpg_mask, pos, hp.aux_local_bp),
                     hp.aux_local_weight * aux_scale),
           "cross": (aux_pred["cross"], cohort_mean_target(model, batch["patch_id"], true),
                     hp.aux_cross_weight * aux_scale)}
    return compute_loss(pred, true, eval_mask, thin_mask, aux)


class _ArmStats:
    """Count-weighted running means of each loss term."""

    def __init__(self):
        self.sum, self.n = {}, {}

    def add(self, parts):
        for key, (loss, n) in parts.items():
            if n:
                self.sum[key] = self.sum.get(key, 0.0) + loss.item() * n
                self.n[key] = self.n.get(key, 0) + n

    def mean(self, key):
        return self.sum[key] / self.n[key] if self.n.get(key) else float("nan")


def _report(desc, i, total, start, stats):
    el = time.time() - start
    print(f"  {desc}: {i}/{total} {i / el:.2f} it/s elapsed {el:.0f}s "
          f"loss={stats.mean('main'):.5f}", file=sys.stderr, flush=True)


def train_step(model, loader, optimizer, device, *, global_step, warmup_steps, base_lr,
               aux_scale, desc="train"):
    """One training epoch. Returns ``(stats, global_step)``."""
    model.train()
    stats = _ArmStats()
    every = max(1, len(loader) // 4)
    start = time.time()
    for i, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        if warmup_steps and global_step < warmup_steps:
            for pg in optimizer.param_groups:
                pg['lr'] = min(1.0, float(global_step + 1) / warmup_steps) * base_lr
        loss, parts = _forward_batch(model, batch, device, aux_scale)
        if loss is None:
            continue
        loss.backward()
        optimizer.step()
        global_step += 1
        stats.add(parts)
        if i % every == 0:
            _report(desc, i, len(loader), start, stats)
    return stats, global_step


@torch.no_grad()
def valid_step(model, loader, device, aux_scale):
    """One validation epoch."""
    model.eval()
    stats = _ArmStats()
    for batch in loader:
        loss, parts = _forward_batch(model, batch, device, aux_scale)
        if loss is not None:
            stats.add(parts)
    return stats


class EarlyStopper:
    """Stop when validation loss has not improved for ``patience`` epochs."""

    def __init__(self, patience=5):
        self.patience = patience
        self.best = float('inf')
        self.counter = 0

    def step(self, value):
        if value < self.best:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def fit(model, train_loader, val_loader, *, device, ckpt_path, max_epochs=100, patience=5):
    """Train, keeping the checkpoint with the lowest validation denoising loss.

    Early stopping and checkpoint selection both use the read-thinned arm of the
    validation loss. Returns the checkpoint path.
    """
    device = torch.device(device)
    model = model.to(device)
    hp = model.hparams
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer, lr_lambda=lambda epoch: hp.lr_decay_factor)
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)

    stopper = EarlyStopper(patience=patience)
    best_val = float('inf')
    global_step = 0
    for epoch in range(max_epochs):
        # The auxiliary weights ramp from 0 to full over the first aux_warmup_epochs.
        aux_scale = min(1.0, epoch / float(hp.aux_warmup_epochs)) if hp.aux_warmup_epochs else 1.0
        tag = f"Epoch {epoch + 1}/{max_epochs}"
        tr, global_step = train_step(model, train_loader, optimizer, device,
                                     global_step=global_step, warmup_steps=hp.warmup_steps,
                                     base_lr=hp.lr, aux_scale=aux_scale, desc=tag)
        scheduler.step()
        va = valid_step(model, val_loader, device, aux_scale)
        sel = va.mean("upscale")
        print(f"epoch {epoch + 1}/{max_epochs}  train_loss {tr.mean('main'):.5f}  "
              f"val_loss {va.mean('main'):.5f}  | val_impute {va.mean('impute'):.5f}  "
              f"val_upscale {sel:.5f}  lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if sel != sel:
            raise RuntimeError("no read-thinned validation cells; check data.thin_min_cov")
        if sel < best_val:
            best_val = sel
            model.save_checkpoint(ckpt_path)
        if stopper.step(sel):
            print(f"Early stopping: val_upscale has not improved for {patience} epochs.", flush=True)
            break
    print(f"Best val_upscale {best_val:.5f}, saved to {ckpt_path}", flush=True)
    return ckpt_path
