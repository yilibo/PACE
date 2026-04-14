#!/usr/bin/env python3
"""Trajectory relation encoding and sparse write-back modules."""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pos_enc(x: torch.Tensor, out_dim: int) -> torch.Tensor:
    """
    Encode 2D coordinates with sinusoidal features.
    Args:
        x: [..., 2] normalized coordinates in [-1, 1].
        out_dim: output dimension.
    """
    if out_dim <= 0:
        return x.new_zeros(*x.shape[:-1], 0)
    base = max(1, out_dim // 4)
    freq = torch.arange(base, device=x.device, dtype=x.dtype)
    freq = torch.exp(-math.log(10000.0) * freq / max(base - 1, 1))
    px = x[..., 0:1] * freq
    py = x[..., 1:2] * freq
    enc = torch.cat([torch.sin(px), torch.cos(px), torch.sin(py), torch.cos(py)], dim=-1)
    if enc.shape[-1] < out_dim:
        pad = out_dim - enc.shape[-1]
        enc = torch.cat([enc, enc.new_zeros(*enc.shape[:-1], pad)], dim=-1)
    return enc[..., :out_dim]


def _fill_missing_points(points: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """
    Fill missing trajectory points with nearest valid coordinates.
    Args:
        points: [B, T, M, 2]
        vis: [B, T, M] boolean
    """
    filled = points.clone()
    bsz, tdim, mdim, _ = points.shape
    for b in range(bsz):
        for m in range(mdim):
            valid_idx = torch.where(vis[b, :, m])[0]
            if valid_idx.numel() == 0:
                filled[b, :, m] = 0.0
                continue
            first_i = int(valid_idx[0].item())
            last_i = int(valid_idx[-1].item())
            filled[b, :first_i, m] = points[b, first_i, m]
            filled[b, last_i + 1 :, m] = points[b, last_i, m]
            for i in range(valid_idx.numel() - 1):
                l = int(valid_idx[i].item())
                r = int(valid_idx[i + 1].item())
                if r - l <= 1:
                    continue
                p0 = points[b, l, m]
                p1 = points[b, r, m]
                for t in range(l + 1, r):
                    alpha = float(t - l) / float(r - l)
                    filled[b, t, m] = p0 * (1.0 - alpha) + p1 * alpha
    return filled


class MaskedTrajectoryRelationEncoder(nn.Module):
    """Temporal + neighborhood relation encoder with trajectory masking."""

    def __init__(self, d_model: int, num_heads: int, k_nn: int):
        super().__init__()
        self.d_model = d_model
        self.k_nn = k_nn
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.rel_q = nn.Linear(d_model, d_model)
        self.rel_k = nn.Linear(d_model, d_model)
        self.rel_v = nn.Linear(d_model, d_model)
        self.rel_norm = nn.LayerNorm(d_model)
        self.rel_out = nn.Linear(d_model, d_model)
        self.seg_mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.traj_mask_token = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        nn.init.normal_(self.seg_mask_token, std=0.02)
        nn.init.normal_(self.traj_mask_token, std=0.02)

    def _apply_mask(
        self,
        h: torch.Tensor,
        vis: torch.Tensor,
        seg_mask_prob: float,
        traj_mask_prob: float,
        seg_min_ratio: float,
        seg_max_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            valid = vis.bool()
            return h, valid
        bsz, tdim, mdim, _ = h.shape
        out = h
        valid = vis.bool().clone()
        if traj_mask_prob > 0:
            traj_mask = (torch.rand(bsz, mdim, device=h.device) < traj_mask_prob)
            traj_mask = traj_mask.unsqueeze(1).expand(-1, tdim, -1)
            out = torch.where(traj_mask.unsqueeze(-1), self.traj_mask_token.expand_as(out), out)
            valid = valid & (~traj_mask)
        if seg_mask_prob > 0:
            for b in range(bsz):
                for m in range(mdim):
                    if torch.rand(1, device=h.device).item() >= seg_mask_prob:
                        continue
                    ratio = torch.empty(1, device=h.device).uniform_(seg_min_ratio, seg_max_ratio).item()
                    seg_len = max(1, int(round(tdim * ratio)))
                    start = int(torch.randint(0, max(1, tdim - seg_len + 1), (1,), device=h.device).item())
                    end = min(tdim, start + seg_len)
                    out[b, start:end, m] = self.seg_mask_token
                    valid[b, start:end, m] = False
        return out, valid

    def forward(
        self,
        h: torch.Tensor,
        points: torch.Tensor,
        vis: torch.Tensor,
        seg_mask_prob: float,
        traj_mask_prob: float,
        seg_min_ratio: float,
        seg_max_ratio: float,
    ) -> torch.Tensor:
        """
        Args:
            h: [B, T, M, Dh]
            points: [B, T, M, 2]
            vis: [B, T, M]
        """
        x, valid = self._apply_mask(h, vis, seg_mask_prob, traj_mask_prob, seg_min_ratio, seg_max_ratio)
        x = torch.nan_to_num(x)
        bsz, tdim, mdim, ddim = x.shape

        # Temporal propagation per trajectory.
        x_tm = x.permute(0, 2, 1, 3).reshape(bsz * mdim, tdim, ddim)
        key_padding = ~valid.permute(0, 2, 1).reshape(bsz * mdim, tdim)
        x_tm_norm = self.temporal_norm(x_tm)
        tm_out, _ = self.temporal_attn(x_tm_norm, x_tm_norm, x_tm_norm, key_padding_mask=key_padding)
        x_tm = x_tm + tm_out
        x_tm = x_tm + self.temporal_ffn(self.temporal_norm(x_tm))
        x = x_tm.reshape(bsz, mdim, tdim, ddim).permute(0, 2, 1, 3).contiguous()

        # Neighborhood interaction at each frame.
        points_filled = _fill_missing_points(points, vis.bool())
        out_frames = []
        for t in range(tdim):
            pt_t = points_filled[:, t]  # [B, M, 2]
            x_t = x[:, t]               # [B, M, D]
            v_t = valid[:, t]           # [B, M]
            dist = torch.cdist(pt_t, pt_t, p=2)  # [B, M, M]
            k = min(self.k_nn, mdim)
            nn_idx = dist.topk(k=k, dim=-1, largest=False).indices  # [B, M, K]

            q = self.rel_q(x_t)
            k_t = self.rel_k(x_t)
            v_t_proj = self.rel_v(x_t)
            knn_k = torch.gather(k_t.unsqueeze(1).expand(-1, mdim, -1, -1), 2, nn_idx.unsqueeze(-1).expand(-1, -1, -1, ddim))
            knn_v = torch.gather(v_t_proj.unsqueeze(1).expand(-1, mdim, -1, -1), 2, nn_idx.unsqueeze(-1).expand(-1, -1, -1, ddim))
            logits = (q.unsqueeze(2) * knn_k).sum(-1) / math.sqrt(ddim)
            nbr_valid = torch.gather(v_t, 1, nn_idx.reshape(bsz, -1)).reshape(bsz, mdim, k)
            logits = logits.masked_fill(~nbr_valid, -1e4)
            attn = logits.softmax(dim=-1)
            rel = (attn.unsqueeze(-1) * knn_v).sum(dim=2)
            rel = self.rel_out(rel)
            out_frames.append(self.rel_norm(x_t + rel))
        return torch.nan_to_num(torch.stack(out_frames, dim=1))


class TrajectoryRelationFusion(nn.Module):
    """Build trajectory tokens, encode relations, and write back to patch tokens."""

    def __init__(self, d_x: int, d_h: int, num_heads: int, max_traj: int, max_time: int, k_nn: int):
        super().__init__()
        self.d_h = d_h
        self.pos_dim = max(16, d_h // 4)
        self.mot_dim = max(16, d_h // 4)
        in_dim = d_x + self.pos_dim + self.mot_dim
        self.proj_in = nn.Linear(in_dim, d_h)
        self.time_embed = nn.Embedding(max_time, d_h)
        self.traj_embed = nn.Embedding(max_traj, d_h)
        self.missing_motion = nn.Parameter(torch.zeros(1, 1, 1, self.mot_dim))
        self.unk_token = nn.Parameter(torch.zeros(1, 1, 1, d_h))
        self.global_motion_gamma = nn.Parameter(torch.tensor(0.5))
        self.rel_encoder = MaskedTrajectoryRelationEncoder(d_h, num_heads, k_nn)
        self.proj_out = nn.Linear(d_h, d_x)
        self.writeback_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.missing_motion, std=0.02)
        nn.init.normal_(self.unk_token, std=0.02)

    def _build_tokens(self, traj_feat: torch.Tensor, points: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        bsz, tdim, mdim, _ = traj_feat.shape
        pos_enc = _sinusoidal_pos_enc(points, self.pos_dim)
        disp = points[:, 1:] - points[:, :-1]
        valid_pair = vis[:, 1:] & vis[:, :-1]
        if valid_pair.any():
            g = torch.zeros_like(disp)
            denom = valid_pair.sum(dim=2, keepdim=True).clamp(min=1).float()
            g_val = (disp * valid_pair.unsqueeze(-1).float()).sum(dim=2, keepdim=True) / denom.unsqueeze(-1)
            g = g_val.expand_as(disp)
        else:
            g = torch.zeros_like(disp)
        disp_comp = disp - torch.clamp(self.global_motion_gamma, 0.0, 1.0) * g
        mot_enc = _sinusoidal_pos_enc(disp_comp, self.mot_dim)
        mot_full = torch.cat([self.missing_motion.expand(bsz, 1, mdim, -1), mot_enc], dim=1)
        valid_disp = torch.cat(
            [torch.ones(bsz, 1, mdim, device=vis.device, dtype=torch.bool), valid_pair],
            dim=1,
        )
        mot_full = torch.where(
            valid_disp.unsqueeze(-1),
            mot_full,
            self.missing_motion.expand_as(mot_full),
        )
        raw = torch.cat([traj_feat, pos_enc, mot_full], dim=-1)
        h = self.proj_in(raw)

        t_idx = torch.arange(tdim, device=h.device).view(1, tdim, 1)
        m_idx = torch.arange(mdim, device=h.device).view(1, 1, mdim)
        h = h + self.time_embed(t_idx).expand(bsz, -1, mdim, -1)
        h = h + self.traj_embed(m_idx).expand(bsz, tdim, -1, -1)
        h = torch.where(vis.unsqueeze(-1), h, self.unk_token.expand_as(h))
        return h

    def _sparse_writeback(self, patch_feat: torch.Tensor, traj_upd: torch.Tensor, points: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        bsz, tdim, npatch, d_x = patch_feat.shape
        g = int(round(math.sqrt(npatch)))
        if g * g != npatch:
            return patch_feat
        ys = torch.linspace(-1.0, 1.0, steps=g, device=patch_feat.device, dtype=patch_feat.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=g, device=patch_feat.device, dtype=patch_feat.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [Np, 2]
        sigma = 2.0 / float(max(1, g))
        updated_frames = []
        for t in range(tdim):
            base_patch = patch_feat[:, t]
            pt_t = points[:, t]      # [B, M, 2]
            vis_t = vis[:, t]        # [B, M]
            traj_t = traj_upd[:, t]  # [B, M, Dx]
            diff = pt_t.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)  # [B, M, Np, 2]
            dist2 = (diff ** 2).sum(-1)
            w = torch.exp(-dist2 / (2.0 * sigma * sigma)) * vis_t.unsqueeze(-1).float()  # [B, M, Np]
            delta = torch.einsum("bmn,bmd->bnd", w, traj_t)
            norm = w.sum(dim=1, keepdim=False).unsqueeze(-1).clamp(min=1e-6)
            delta = delta / norm
            updated_frames.append(base_patch + self.writeback_scale * delta)
        return torch.stack(updated_frames, dim=1)

    def forward(
        self,
        patch_feat: torch.Tensor,
        traj_feat: torch.Tensor,
        points: torch.Tensor,
        vis: torch.Tensor,
        seg_mask_prob: float,
        traj_mask_prob: float,
        seg_min_ratio: float,
        seg_max_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._build_tokens(traj_feat, points, vis)
        h = torch.nan_to_num(h)
        h_rel = self.rel_encoder(
            h, points, vis,
            seg_mask_prob=seg_mask_prob,
            traj_mask_prob=traj_mask_prob,
            seg_min_ratio=seg_min_ratio,
            seg_max_ratio=seg_max_ratio,
        )
        traj_upd = torch.nan_to_num(self.proj_out(h_rel))
        patch_upd = torch.nan_to_num(self._sparse_writeback(patch_feat, traj_upd, points, vis.bool()))
        return patch_upd, torch.nan_to_num(h_rel)
