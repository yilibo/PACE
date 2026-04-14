#!/usr/bin/env python3
"""PACE model: Process-Aligned Concept Encoder for video action recognition."""

import json
import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from clip import clip as clip_lib
from .build import MODEL_REGISTRY
from .trajectory_relation import TrajectoryRelationFusion


def _to_video_tensor(x):
    if isinstance(x, (list, tuple)):
        return x[0]
    return x


def _build_default_class_names(num_classes: int) -> List[str]:
    return [f"class_{i}" for i in range(num_classes)]


def _load_prompt_spec(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, str):
                out[str(k)] = [v]
            elif isinstance(v, list):
                out[str(k)] = [str(x) for x in v]
            else:
                raise ValueError(f"Unsupported prompt type for key {k}: {type(v)}")
        return out
    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict) and "class_name" in item and "prompts" in item:
                prompts = item["prompts"]
                if isinstance(prompts, str):
                    prompts = [prompts]
                out[str(item["class_name"])] = [str(x) for x in prompts]
            else:
                raise ValueError("Prompt list items must be dicts with class_name and prompts")
        return out
    raise ValueError(f"Unsupported prompt file format: {type(data)}")


@MODEL_REGISTRY.register()
class PACE(nn.Module):
    """Main method model: CLIP patch features + trajectory relation + text alignment."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = int(cfg.MODEL.NUM_CLASSES)
        self.num_frames = int(cfg.DATA.NUM_FRAMES)
        self.input_size = int(cfg.DATA.TRAIN_CROP_SIZE)
        self.temperature = float(cfg.MODEL.METHOD.ALIGN_TEMPERATURE)
        self.text_prompt_num = int(cfg.MODEL.METHOD.TEXT_PROMPT_NUM)
        self.text_template = str(cfg.MODEL.METHOD.TEXT_TEMPLATE)
        self.freeze_clip = bool(cfg.MODEL.METHOD.FREEZE_CLIP)
        self.insert_layers = int(cfg.MODEL.METHOD.TRAJ_INSERT_LAYERS)
        self.share_traj_module = bool(cfg.MODEL.METHOD.TRAJ_SHARE_MODULE)

        clip_name = str(cfg.MODEL.METHOD.CLIP_MODEL_NAME)
        clip_ckpt = str(cfg.MODEL.METHOD.CLIP_PRETRAIN_PATH)
        clip_source = clip_ckpt if clip_ckpt else clip_name
        self.clip_model, _ = clip_lib.load(clip_source, device="cpu", jit=False)
        self.visual = self.clip_model.visual
        self.text_encoder = self.clip_model

        if not hasattr(self.visual, "conv1"):
            raise ValueError("Current CLIP visual backbone does not expose conv1 patch embedding.")

        self.patch_size = int(self.visual.conv1.kernel_size[0])
        self.patch_grid = self.input_size // self.patch_size
        self.num_patches = self.patch_grid * self.patch_grid
        self.patch_dim = int(self.visual.conv1.out_channels)
        self.clip_embed_dim = int(self.clip_model.text_projection.shape[1])
        if not hasattr(self.visual, "transformer"):
            raise ValueError("Current CLIP visual backbone does not expose transformer blocks.")
        self.num_visual_layers = int(len(self.visual.transformer.resblocks))
        self.insert_layers = max(0, min(self.insert_layers, self.num_visual_layers))
        self.inject_layer_idx = list(range(self.num_visual_layers - self.insert_layers, self.num_visual_layers))

        self.traj_relation = TrajectoryRelationFusion(
            d_x=self.patch_dim,
            d_h=int(cfg.MODEL.METHOD.TRAJ_HIDDEN_DIM),
            num_heads=int(cfg.MODEL.METHOD.TRAJ_NUM_HEADS),
            max_traj=max(512, int(cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)),
            max_time=max(16, self.num_frames),
            k_nn=int(cfg.MODEL.METHOD.TRAJ_KNN),
        )
        if self.share_traj_module:
            self.traj_relations = nn.ModuleList([self.traj_relation])
        else:
            self.traj_relations = nn.ModuleList(
                [
                    TrajectoryRelationFusion(
                        d_x=self.patch_dim,
                        d_h=int(cfg.MODEL.METHOD.TRAJ_HIDDEN_DIM),
                        num_heads=int(cfg.MODEL.METHOD.TRAJ_NUM_HEADS),
                        max_traj=max(512, int(cfg.POINT_INFO.NUM_POINTS_TO_SAMPLE)),
                        max_time=max(16, self.num_frames),
                        k_nn=int(cfg.MODEL.METHOD.TRAJ_KNN),
                    )
                    for _ in self.inject_layer_idx
                ]
            )

        self.class_names = self._load_class_names()
        self.register_buffer(
            "text_prototypes",
            self._build_text_prototypes(),
            persistent=True,
        )

        if self.freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

    def _load_class_names(self) -> List[str]:
        class_path = str(self.cfg.MODEL.METHOD.CLASS_NAME_PATH)
        if class_path and os.path.exists(class_path):
            with open(class_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                names = [str(x) for x in data]
            elif isinstance(data, dict):
                # allow {"0":"jumping", ...} or {"class_names":[...]}
                if "class_names" in data:
                    names = [str(x) for x in data["class_names"]]
                else:
                    names = [str(v) for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))]
            else:
                names = _build_default_class_names(self.num_classes)
        else:
            names = _build_default_class_names(self.num_classes)
        if len(names) < self.num_classes:
            names += _build_default_class_names(self.num_classes - len(names))
        return names[: self.num_classes]

    def _build_prompts(self) -> List[List[str]]:
        prompt_path = str(self.cfg.MODEL.METHOD.TEXT_PROMPT_PATH)
        if prompt_path and os.path.exists(prompt_path):
            spec = _load_prompt_spec(prompt_path)
            prompts = []
            for cname in self.class_names:
                class_prompts = spec.get(cname, [])
                if len(class_prompts) == 0:
                    class_prompts = [self.text_template.format(action=cname)]
                prompts.append(class_prompts[: self.text_prompt_num])
            return prompts

        prompts = []
        for cname in self.class_names:
            cls_prompts = []
            for _ in range(self.text_prompt_num):
                cls_prompts.append(self.text_template.format(action=cname))
            prompts.append(cls_prompts)
        return prompts

    def _build_text_prototypes(self) -> torch.Tensor:
        prompts = self._build_prompts()
        flat_prompts = [p for cls_ps in prompts for p in cls_ps]
        with torch.no_grad():
            tok = clip_lib.tokenize(flat_prompts)
            txt = self.text_encoder.encode_text(tok)
            txt = F.normalize(txt.float(), dim=-1)
        txt = txt.view(self.num_classes, self.text_prompt_num, -1)
        return txt

    def _traj_module_by_idx(self, layer_i: int):
        if self.share_traj_module:
            return self.traj_relations[0]
        rel_i = self.inject_layer_idx.index(layer_i)
        return self.traj_relations[rel_i]

    def _encode_video_with_traj(self, video: torch.Tensor, points: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
        # video: [B,T,C,H,W], points: [B,T,M,2] in [-1,1], vis: [B,T,M]
        bsz, tdim, cdim, height, width = video.shape
        x = rearrange(video, "b t c h w -> (b t) c h w")
        x = self.visual.conv1(x)  # [BT,D,gh,gw]
        gh, gw = x.shape[-2:]
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [BT,Np,D]

        cls = self.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)  # [BT,1+Np,D]
        if hasattr(self.visual, "positional_embedding"):
            pe = self.visual.positional_embedding
            if pe.ndim == 2 and pe.shape[0] >= (1 + gh * gw):
                x = x + pe[: 1 + gh * gw].to(x.dtype)

        if getattr(self.visual, "VPT_shallow", False):
            visual_ctx = self.visual.VPT.expand(x.shape[0], -1, -1).to(x.dtype)
            x = torch.cat([x, visual_ctx], dim=1)

        x = self.visual.ln_pre(x)                 # [BT,L,D]
        x = x.permute(1, 0, 2)                    # [L,BT,D]

        for i, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if i not in self.inject_layer_idx:
                continue
            x_nld = x.permute(1, 0, 2).contiguous()  # [BT,L,D]
            cls_tokens = x_nld[:, :1, :]
            patch_tokens = x_nld[:, 1 : 1 + (gh * gw), :]
            extra_tokens = x_nld[:, 1 + (gh * gw) :, :]
            patch_tokens = rearrange(patch_tokens, "(b t) n d -> b t n d", b=bsz, t=tdim)
            traj_feat = self._sample_traj_feat(patch_tokens, points)
            traj_module = self._traj_module_by_idx(i)
            patch_upd, _ = traj_module(
                patch_feat=patch_tokens,
                traj_feat=traj_feat,
                points=points,
                vis=vis,
                seg_mask_prob=float(self.cfg.MODEL.METHOD.SEG_MASK_PROB),
                traj_mask_prob=float(self.cfg.MODEL.METHOD.TRAJ_MASK_PROB),
                seg_min_ratio=float(self.cfg.MODEL.METHOD.SEG_MASK_MIN_RATIO),
                seg_max_ratio=float(self.cfg.MODEL.METHOD.SEG_MASK_MAX_RATIO),
            )
            patch_upd = rearrange(torch.nan_to_num(patch_upd), "b t n d -> (b t) n d")
            x_nld = torch.cat([cls_tokens, patch_upd, extra_tokens], dim=1)
            x = x_nld.permute(1, 0, 2).contiguous()

        x = x.permute(1, 0, 2)   # [BT,L,D]
        x = self.visual.ln_post(x[:, 0, :])  # [BT,D]
        if self.visual.proj is not None:
            x = x @ self.visual.proj
        x = torch.nan_to_num(x)
        return rearrange(x, "(b t) d -> b t d", b=bsz, t=tdim)

    def _sample_traj_feat(self, patch_tokens: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # patch_tokens: [B,T,Np,D], points: [B,T,M,2] in [-1,1]
        bsz, tdim, npatch, ddim = patch_tokens.shape
        g = int(round(math.sqrt(npatch)))
        fmap = rearrange(patch_tokens, "b t (gh gw) d -> (b t) d gh gw", gh=g, gw=g)
        grid = rearrange(points, "b t m d -> (b t) m 1 d")
        sampled = F.grid_sample(fmap, grid, align_corners=True, mode="bilinear")
        sampled = rearrange(sampled, "(b t) d m 1 -> b t m d", b=bsz, t=tdim)
        return sampled

    def _frame_text_scores(self, z: torch.Tensor, text_proto: torch.Tensor) -> torch.Tensor:
        # z: [B,T,D], text_proto: [C,Ns,D]
        sim = torch.einsum("btd,cnd->btcn", z, text_proto)
        s_v2t = sim.max(dim=3).values.mean(dim=1)   # [B,C]
        s_t2v = sim.max(dim=1).values.mean(dim=2)   # [B,C]
        return 0.5 * (s_v2t + s_t2v)

    def forward(self, input_to_use):
        video = _to_video_tensor(input_to_use["video"])
        metadata = input_to_use["metadata"]
        pred_tracks = metadata["pred_tracks"].float()
        pred_visibility = metadata["pred_visibility"].bool()

        if pred_tracks.abs().max().item() > 1.5:
            # convert pixel coords to [-1,1]
            h, w = video.shape[-2:]
            div = pred_tracks.new_tensor([max(w - 1, 1), max(h - 1, 1)]).view(1, 1, 1, 2)
            pred_tracks = pred_tracks / div
            pred_tracks = pred_tracks * 2.0 - 1.0

        z = F.normalize(self._encode_video_with_traj(video, pred_tracks, pred_visibility), dim=-1)  # [B,T,D]

        text_proto = self.text_prototypes.to(z.device)
        text_proto = F.normalize(text_proto, dim=-1)
        logits = self._frame_text_scores(z, text_proto) / max(self.temperature, 1e-6)
        video_embed = F.normalize(z.mean(dim=1), dim=-1)
        if self.cfg.TASK == "few_shot":
            return logits, video_embed
        return logits


@MODEL_REGISTRY.register()
class TrajClipMethod(PACE):
    """Backward-compatible alias for previous model naming."""
