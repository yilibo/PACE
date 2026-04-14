
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _pos_enc_2d(points: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        return points.new_zeros(*points.shape[:-1], 0)
    base = max(1, dim // 4)
    freq = torch.arange(base, device=points.device, dtype=points.dtype)
    freq = torch.exp(-math.log(10000.0) * freq / max(base - 1, 1))
    x = points[..., 0:1] * freq
    y = points[..., 1:2] * freq
    enc = torch.cat([torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)], dim=-1)
    if enc.shape[-1] < dim:
        enc = torch.cat([enc, enc.new_zeros(*enc.shape[:-1], dim - enc.shape[-1])], dim=-1)
    return enc[..., :dim]


def _fill_missing(points: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    filled = points.clone()
    bsz, tdim, mdim, _ = points.shape
    for b in range(bsz):
        for m in range(mdim):
            valid = torch.where(visibility[b, :, m])[0]
            if valid.numel() == 0:
                filled[b, :, m] = 0.0
                continue
            first_i = int(valid[0].item())
            last_i = int(valid[-1].item())
            filled[b, :first_i, m] = points[b, first_i, m]
            filled[b, last_i + 1 :, m] = points[b, last_i, m]
            for i in range(valid.numel() - 1):
                l = int(valid[i].item())
                r = int(valid[i + 1].item())
                if r - l <= 1:
                    continue
                p0 = points[b, l, m]
                p1 = points[b, r, m]
                for t in range(l + 1, r):
                    alpha = float(t - l) / float(r - l)
                    filled[b, t, m] = p0 * (1.0 - alpha) + p1 * alpha
    return filled


class TrajectoryMethodAdapter(nn.Module):
    """
    Implements:
    1) trajectory token construction from patch tokens + trajectory points
    2) masked trajectory relation encoding
    3) sparse trajectory-to-patch write-back
    4) context token summarization
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_time: int,
        max_traj: int,
        context_k: int,
        k_nn: int = 8,
        seg_mask_prob: float = 0.3,
        traj_mask_prob: float = 0.1,
        seg_min_ratio: float = 0.2,
        seg_max_ratio: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_k = context_k
        self.k_nn = k_nn
        self.seg_mask_prob = seg_mask_prob
        self.traj_mask_prob = traj_mask_prob
        self.seg_min_ratio = seg_min_ratio
        self.seg_max_ratio = seg_max_ratio

        self.pos_dim = max(16, d_model // 4)
        self.mot_dim = max(16, d_model // 4)
        self.proj_in = nn.Linear(d_model + self.pos_dim + self.mot_dim, d_model)
        self.temb = nn.Embedding(max_time, d_model)
        self.memb = nn.Embedding(max_traj, d_model)
        self.missing_motion = nn.Parameter(torch.zeros(1, 1, 1, self.mot_dim))
        self.unk = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.seg_mask = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.traj_mask = nn.Parameter(torch.zeros(1, 1, 1, d_model))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Parameter(torch.tensor(1.0))

        self.tmp_norm = nn.LayerNorm(d_model)
        self.tmp_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.tmp_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.rel_norm = nn.LayerNorm(d_model)
        self.rel_q = nn.Linear(d_model, d_model)
        self.rel_k = nn.Linear(d_model, d_model)
        self.rel_v = nn.Linear(d_model, d_model)
        self.rel_out = nn.Linear(d_model, d_model)
        self.proj_out = nn.Linear(d_model, d_model)

        nn.init.normal_(self.missing_motion, std=0.02)
        nn.init.normal_(self.unk, std=0.02)
        nn.init.normal_(self.seg_mask, std=0.02)
        nn.init.normal_(self.traj_mask, std=0.02)

    def _sample_patch_feat(self, patch_tokens: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # patch_tokens [B, T, Np, D], points [B, T, M, 2] in [-1,1]
        bsz, tdim, npatch, ddim = patch_tokens.shape
        g = int(round(math.sqrt(npatch)))
        if g * g != npatch:
            # fallback: nearest index mapping
            px = ((points[..., 0] + 1.0) * 0.5 * (g - 1)).round().long().clamp(0, g - 1)
            py = ((points[..., 1] + 1.0) * 0.5 * (g - 1)).round().long().clamp(0, g - 1)
            idx = py * g + px
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, -1, ddim)
            return torch.gather(patch_tokens, 2, gather_idx)
        feat_map = patch_tokens.reshape(bsz * tdim, g, g, ddim).permute(0, 3, 1, 2)
        grid = points.reshape(bsz * tdim, -1, 1, 2)
        sampled = torch.nn.functional.grid_sample(feat_map, grid, align_corners=True, mode="bilinear")
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # [BT, M, D]
        return sampled.reshape(bsz, tdim, -1, ddim)

    def _make_tokens(
        self,
        patch_tokens: torch.Tensor,
        points: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        bsz, tdim, mdim, _ = points.shape
        traj_feat = self._sample_patch_feat(patch_tokens, points)
        pos = _pos_enc_2d(points, self.pos_dim)
        disp = points[:, 1:] - points[:, :-1]
        valid_pair = visibility[:, 1:] & visibility[:, :-1]
        denom = valid_pair.sum(dim=2, keepdim=True).clamp(min=1).float().unsqueeze(-1)
        global_disp = (disp * valid_pair.unsqueeze(-1).float()).sum(dim=2, keepdim=True) / denom
        global_disp = global_disp.expand_as(disp)
        motion = disp - torch.clamp(self.gamma, 0.0, 1.0) * global_disp
        mot = _pos_enc_2d(motion, self.mot_dim)
        mot = torch.cat([self.missing_motion.expand(bsz, 1, mdim, -1), mot], dim=1)
        valid_motion = torch.cat(
            [torch.ones(bsz, 1, mdim, device=visibility.device, dtype=torch.bool), valid_pair],
            dim=1,
        )
        mot = torch.where(valid_motion.unsqueeze(-1), mot, self.missing_motion.expand_as(mot))
        h = self.proj_in(torch.cat([traj_feat, pos, mot], dim=-1))
        t_idx = torch.arange(tdim, device=h.device).view(1, tdim, 1)
        m_idx = torch.arange(mdim, device=h.device).view(1, 1, mdim)
        h = h + self.temb(t_idx).expand(bsz, -1, mdim, -1)
        h = h + self.memb(m_idx).expand(bsz, tdim, -1, -1)
        h = torch.where(visibility.unsqueeze(-1), h, self.unk.expand_as(h))
        return h

    def _apply_masks(self, h: torch.Tensor, visibility: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = visibility.bool().clone()
        out = h
        if not self.training:
            return out, valid
        bsz, tdim, mdim, _ = h.shape
        if self.traj_mask_prob > 0:
            traj_m = (torch.rand(bsz, mdim, device=h.device) < self.traj_mask_prob)
            traj_m = traj_m.unsqueeze(1).expand(-1, tdim, -1)
            out = torch.where(traj_m.unsqueeze(-1), self.traj_mask.expand_as(out), out)
            valid = valid & (~traj_m)
        if self.seg_mask_prob > 0:
            for b in range(bsz):
                for m in range(mdim):
                    if torch.rand(1, device=h.device).item() >= self.seg_mask_prob:
                        continue
                    ratio = torch.empty(1, device=h.device).uniform_(self.seg_min_ratio, self.seg_max_ratio).item()
                    seg_len = max(1, int(round(tdim * ratio)))
                    start = int(torch.randint(0, max(1, tdim - seg_len + 1), (1,), device=h.device).item())
                    end = min(tdim, start + seg_len)
                    out[b, start:end, m] = self.seg_mask
                    valid[b, start:end, m] = False
        return out, valid

    def _encode_relations(self, h: torch.Tensor, points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        bsz, tdim, mdim, ddim = h.shape
        x = h.permute(0, 2, 1, 3).reshape(bsz * mdim, tdim, ddim)
        pad = ~valid.permute(0, 2, 1).reshape(bsz * mdim, tdim)
        x_norm = self.tmp_norm(x)
        x_attn, _ = self.tmp_attn(x_norm, x_norm, x_norm, key_padding_mask=pad)
        x = x + x_attn
        x = x + self.tmp_ffn(self.tmp_norm(x))
        x = x.reshape(bsz, mdim, tdim, ddim).permute(0, 2, 1, 3).contiguous()

        out = x
        pfill = _fill_missing(points, valid)
        for t in range(tdim):
            pt = pfill[:, t]      # [B, M, 2]
            xt = out[:, t]        # [B, M, D]
            vt = valid[:, t]      # [B, M]
            dist = torch.cdist(pt, pt)
            k = min(self.k_nn, mdim)
            nn_idx = dist.topk(k=k, largest=False, dim=-1).indices  # [B, M, K]
            q = self.rel_q(xt)
            kk = self.rel_k(xt)
            vv = self.rel_v(xt)
            knn_k = torch.gather(kk.unsqueeze(1).expand(-1, mdim, -1, -1), 2,
                                 nn_idx.unsqueeze(-1).expand(-1, -1, -1, ddim))
            knn_v = torch.gather(vv.unsqueeze(1).expand(-1, mdim, -1, -1), 2,
                                 nn_idx.unsqueeze(-1).expand(-1, -1, -1, ddim))
            logits = (q.unsqueeze(2) * knn_k).sum(-1) / math.sqrt(ddim)
            nbr_valid = torch.gather(vt, 1, nn_idx.reshape(bsz, -1)).reshape(bsz, mdim, k)
            logits = logits.masked_fill(~nbr_valid, -1e4)
            attn = logits.softmax(dim=-1)
            rel = (attn.unsqueeze(-1) * knn_v).sum(dim=2)
            out[:, t] = self.rel_norm(out[:, t] + self.rel_out(rel))
        return out

    def _writeback(self, patch_tokens: torch.Tensor, traj_tokens: torch.Tensor, points: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        # patch_tokens [B,T,Np,D], traj_tokens [B,T,M,D]
        bsz, tdim, npatch, _ = patch_tokens.shape
        g = int(round(math.sqrt(npatch)))
        if g * g != npatch:
            return patch_tokens
        ys = torch.linspace(-1.0, 1.0, steps=g, device=patch_tokens.device, dtype=patch_tokens.dtype)
        xs = torch.linspace(-1.0, 1.0, steps=g, device=patch_tokens.device, dtype=patch_tokens.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [Np,2]
        sigma = 2.0 / max(1, g)
        upd = patch_tokens
        up_traj = self.proj_out(traj_tokens)
        for t in range(tdim):
            pt = points[:, t]              # [B,M,2]
            vt = visibility[:, t].float()  # [B,M]
            tt = up_traj[:, t]             # [B,M,D]
            diff = pt.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1)
            w = torch.exp(-dist2 / (2 * sigma * sigma)) * vt.unsqueeze(-1)  # [B,M,Np]
            delta = torch.einsum("bmn,bmd->bnd", w, tt)
            norm = w.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
            upd[:, t] = upd[:, t] + self.scale * (delta / norm)
        return upd

    def _context_from_traj(
        self,
        traj_tokens: torch.Tensor,
        visibility: torch.Tensor,
        cls_scores: Optional[torch.Tensor],
        patch_tokens: torch.Tensor,
        points: torch.Tensor,
        trace_source: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # traj_tokens [B,T,M,D], choose top-K trajectories and pool over time
        bsz, tdim, mdim, ddim = traj_tokens.shape
        vis_count = visibility.float().sum(dim=1)  # [B,M]
        if cls_scores is not None:
            # sample cls score at trajectory points from nearest patch
            npatch = patch_tokens.shape[2]
            g = int(round(math.sqrt(npatch)))
            px = ((points[..., 0] + 1.0) * 0.5 * (g - 1)).round().long().clamp(0, g - 1)
            py = ((points[..., 1] + 1.0) * 0.5 * (g - 1)).round().long().clamp(0, g - 1)
            idx = py * g + px
            score = torch.gather(cls_scores, 2, idx).mean(dim=1)
            rank = score + 0.1 * vis_count
        else:
            rank = vis_count
        k = min(self.context_k, mdim)
        top_idx = torch.topk(rank, k=k, dim=-1).indices  # [B,K]
        gather_idx = top_idx.unsqueeze(1).unsqueeze(-1).expand(-1, tdim, -1, ddim)
        top_traj = torch.gather(traj_tokens, 2, gather_idx)  # [B,T,K,D]
        top_vis = torch.gather(visibility, 2, top_idx.unsqueeze(1).expand(-1, tdim, -1)).float()  # [B,T,K]
        denom = top_vis.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
        context = (top_traj * top_vis.unsqueeze(-1)).sum(dim=1) / denom  # [B,K,D]

        source = None
        if trace_source:
            # nearest selected trajectory assignment for each patch token
            npatch = patch_tokens.shape[2]
            source = torch.zeros(bsz, k, tdim, npatch, device=patch_tokens.device, dtype=patch_tokens.dtype)
            top_idx_t = top_idx.unsqueeze(1).expand(-1, tdim, -1)  # [B,T,K]
            top_points = torch.gather(points, 2, top_idx_t.unsqueeze(-1).expand(-1, -1, -1, 2))  # [B,T,K,2]
            g = int(round(math.sqrt(npatch)))
            ys = torch.linspace(-1.0, 1.0, steps=g, device=patch_tokens.device, dtype=patch_tokens.dtype)
            xs = torch.linspace(-1.0, 1.0, steps=g, device=patch_tokens.device, dtype=patch_tokens.dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')
            centers = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [Np,2]
            for t in range(tdim):
                d = torch.cdist(top_points[:, t], centers.unsqueeze(0).expand(bsz, -1, -1))  # [B,K,Np]
                assign = d.argmin(dim=1)  # [B,Np]
                onehot = torch.nn.functional.one_hot(assign, num_classes=k).permute(0, 2, 1).to(source.dtype)
                source[:, :, t, :] = onehot
            source = source.reshape(bsz, k, tdim * npatch)
        return context, source

    def forward(
        self,
        patch_tokens: torch.Tensor,
        points: torch.Tensor,
        visibility: torch.Tensor,
        cls_scores: Optional[torch.Tensor] = None,
        trace_source: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            patch_tokens: [B, T, Np, D]
            points: [B, T, M, 2] in [-1, 1]
            visibility: [B, T, M] bool
            cls_scores: [B, T, Np] optional
        Returns:
            patch_updated: [B, T, Np, D]
            context_tokens: [B, K, D]
            source_map: [B, K, T*Np] optional
        """
        h = self._make_tokens(patch_tokens, points, visibility.bool())
        h, valid = self._apply_masks(h, visibility.bool())
        h = self._encode_relations(h, points, valid)
        patch_updated = self._writeback(patch_tokens, h, points, visibility.bool())
        context, source = self._context_from_traj(
            h, visibility.bool(), cls_scores, patch_updated, points, trace_source
        )
        return patch_updated, context, source
