# -*- coding: utf-8 -*-
"""
ODAM-Net public framework module.

This file keeps only the model architecture and core auxiliary losses used by
ODAM-Net. It intentionally excludes dataset construction, file paths, cache
management, training loops, evaluation scripts, visualization utilities, and
experiment-specific hyperparameter schedules.

Expected model inputs:
    x_rsi4:        Tensor[B, 4, H, W]   RGB + parcel mask / spatial mask
    x_aux:         Tensor[B, C_aux, H, W]
    text_features: Tensor[B, 768]       pre-computed text embedding, e.g. BERT pooler output

Core components included:
    - Focal-LDAM loss
    - Orthogonal disentanglement loss
    - Cross-modal supervised contrastive loss
    - RSI and auxiliary feature extractors
    - Modality-aware asymmetric cross-attention
    - Sparse Mixture-of-Experts feed-forward block
    - Masked multimodal reconstruction branch
    - ODAM-Net / MultimodalNet
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Missing dependency: timm. Install it with `pip install timm`.") from exc


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------
class FocalLDAMLoss(nn.Module):
    """
    Focal-LDAM loss for long-tailed classification.

    LDAM enlarges margins for minority classes according to class frequency,
    while the focal term emphasizes difficult samples.
    """

    def __init__(
        self,
        cls_num_list: List[int],
        max_m: float = 0.5,
        weight: torch.Tensor | None = None,
        s: float = 30.0,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        cls_num_arr = np.array(cls_num_list, dtype=np.float32)
        cls_num_arr = np.maximum(cls_num_arr, 1.0)

        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_arr))
        m_list = m_list * (max_m / np.max(m_list))

        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)

        batch_m = self.m_list[target]
        logits_m = logits - batch_m.view(-1, 1)
        adjusted_logits = torch.where(index, logits_m, logits)

        log_p = F.log_softmax(self.s * adjusted_logits, dim=1)
        ce_loss = F.nll_loss(log_p, target, weight=self.weight, reduction="none")
        p = torch.exp(-ce_loss)
        focal_loss = ((1.0 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()


class OrthogonalLoss(nn.Module):
    """Penalizes correlation between shared and modality-specific representations."""

    def forward(self, shared: torch.Tensor, specific: torch.Tensor) -> torch.Tensor:
        shared = F.normalize(shared, dim=-1)
        specific = F.normalize(specific, dim=-1)
        corr = (shared * specific).sum(dim=-1)
        return corr.pow(2).mean()


class CrossModalSupConLoss(nn.Module):
    """
    Cross-modal supervised contrastive loss.

    The input is a list of modality embeddings. Embeddings with the same class
    label are treated as positives across modalities.
    """

    def __init__(self, temperature: float = 0.07, ignore_index: int = -100) -> None:
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        features_list: List[torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = features_list[0].device
        valid_mask = labels != self.ignore_index

        if not valid_mask.any():
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"align_corr": zero}

        features_list = [features[valid_mask] for features in features_list]
        labels = labels[valid_mask]

        features = torch.cat(features_list, dim=0)
        features = F.normalize(features, p=2, dim=1)
        labels = labels.repeat(len(features_list))

        logits = torch.matmul(features, features.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0], device=device).view(-1, 1),
            0.0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        pos_count = torch.clamp(mask.sum(1), min=1e-5)
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_count
        loss = -mean_log_prob_pos.mean()

        return loss, {"align_corr": torch.tensor(0.5, device=device)}


# -----------------------------------------------------------------------------
# Feature extractors
# -----------------------------------------------------------------------------
class RSIFeatureExtractor(nn.Module):
    """Remote-sensing image branch based on ConvNeXt V2."""

    def __init__(
        self,
        model_name: str = "convnextv2_large.fcmae_ft_in22k_in1k",
        in_chans: int = 4,
        pretrained: bool = True,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
            drop_path_rate=drop_path_rate,
        )
        self.out_channels = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class AuxFeatureExtractor(nn.Module):
    """Auxiliary geographic feature branch based on ResNet-18."""

    def __init__(self, in_chans: int, pretrained: bool = False) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=in_chans,
        )
        self.out_channels = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class GateNet(nn.Module):
    """Small reliability gate for each modality."""

    def __init__(self, in_dim: int = 512, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Asymmetric modality-aware attention and sparse MoE
# -----------------------------------------------------------------------------
class AsymmetricModalityAwareAttention(nn.Module):
    """
    Modality-aware asymmetric cross-attention.

    Query tokens are restricted to the fusion token and RSI tokens, while keys
    and values include all modality tokens. A learnable modality bias encodes
    token-level cross-modal relations.

    Query sequence:
        [fusion, rsi_shared, rsi_specific]
    Key/value sequence:
        [fusion, rsi_shared, rsi_specific, aux_shared, aux_specific, txt_shared, txt_specific]
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_len: int = 3,
        kv_len: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.modality_bias = nn.Parameter(torch.zeros(1, num_heads, q_len, kv_len))
        nn.init.normal_(self.modality_bias, std=0.02)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + self.modality_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.out_proj(out)


class Expert(nn.Module):
    """Feed-forward expert used inside SparseMoE."""

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SparseMoE(nn.Module):
    """Top-k sparse mixture-of-experts feed-forward module."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k must be less than or equal to num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, ffn_dim, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        router_logits = self.router(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        importance = routing_probs.sum(0)
        expert_ids = torch.arange(self.num_experts, device=x.device)
        load = (topk_indices.unsqueeze(-1) == expert_ids).float().sum(dim=(0, 1))
        balance_loss = self.num_experts * (importance * load).sum() / (
            x_flat.size(0) ** 2 * self.top_k
        )

        out_flat = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            expert_prob = topk_probs[:, i].unsqueeze(-1)
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    out_flat[mask] += expert_prob[mask] * self.experts[expert_id](x_flat[mask])

        return out_flat.view(batch_size, seq_len, dim), balance_loss


class MoETransformerEncoderLayer(nn.Module):
    """Transformer-style layer with asymmetric attention and SparseMoE."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = AsymmetricModalityAwareAttention(
            d_model=d_model,
            num_heads=nhead,
            q_len=3,
            kv_len=7,
            dropout=dropout,
        )
        self.moe = SparseMoE(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.norm1_q = nn.LayerNorm(d_model)
        self.norm1_kv = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_seq: torch.Tensor,
        kv_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_norm = self.norm1_q(q_seq)
        kv_norm = self.norm1_kv(kv_seq)

        attn_out = self.attn(q_norm, kv_norm, kv_norm)
        q_seq = q_seq + self.dropout(attn_out)

        moe_in = self.norm2(q_seq)
        moe_out, balance_loss = self.moe(moe_in)
        q_seq = q_seq + self.dropout(moe_out)

        return q_seq, balance_loss


# -----------------------------------------------------------------------------
# ODAM-Net main model
# -----------------------------------------------------------------------------
class MultimodalNet(nn.Module):
    """
    ODAM-Net multimodal framework.

    The model decouples each modality into shared and specific representations,
    aligns shared semantics through auxiliary losses, performs asymmetric
    modality-aware fusion, and uses sparse MoE routing for final decision-making.
    """

    def __init__(
        self,
        num_classes: int,
        aux_in_channels: int,
        rsi_in_channels: int = 4,
        text_feat_dim: int = 768,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 1024,
        num_experts: int = 4,
        top_k: int = 2,
        dropout: float = 0.1,
        gate_floor: float = 0.02,
        cca_dim: int = 128,
        rsi_model_name: str = "convnextv2_large.fcmae_ft_in22k_in1k",
        rsi_pretrained: bool = True,
        aux_pretrained: bool = False,
        mask_prob: float = 0.3,
    ) -> None:
        super().__init__()

        self.rsi_extractor = RSIFeatureExtractor(
            model_name=rsi_model_name,
            in_chans=rsi_in_channels,
            pretrained=rsi_pretrained,
        )
        self.aux_extractor = AuxFeatureExtractor(
            in_chans=aux_in_channels,
            pretrained=aux_pretrained,
        )

        self.d_model = d_model
        self.gate_floor = gate_floor
        self.mask_prob = mask_prob

        self.proj_sh_rsi = nn.Sequential(
            nn.Linear(self.rsi_extractor.out_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_sp_rsi = nn.Sequential(
            nn.Linear(self.rsi_extractor.out_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_sh_aux = nn.Sequential(
            nn.Linear(self.aux_extractor.out_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_sp_aux = nn.Sequential(
            nn.Linear(self.aux_extractor.out_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_sh_txt = nn.Sequential(
            nn.Linear(text_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.proj_sp_txt = nn.Sequential(
            nn.Linear(text_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.cca_head_rsi = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, cca_dim),
        )
        self.cca_head_aux = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, cca_dim),
        )
        self.cca_head_txt = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, cca_dim),
        )

        self.gate_rsi = GateNet(d_model * 2, 128)
        self.gate_aux = GateNet(d_model * 2, 128)
        self.gate_txt = GateNet(d_model * 2, 128)

        self.modality_embed = nn.Parameter(torch.randn(1, 7, d_model) * 0.02)
        self.query_bias = nn.Parameter(torch.zeros(1, 1, d_model))

        self.moe_layers = nn.ModuleList(
            [
                MoETransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    ffn_dim=ffn_dim,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.mask_token_aux_sh = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.mask_token_aux_sp = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.mask_token_txt_sh = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.mask_token_txt_sp = nn.Parameter(torch.randn(1, d_model) * 0.02)

        self.mmm_decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 4),
        )

    def _gate_scale(self, gate_value: torch.Tensor) -> torch.Tensor:
        return self.gate_floor + (1.0 - self.gate_floor) * gate_value

    def forward(
        self,
        x_rsi4: torch.Tensor,
        x_aux: torch.Tensor,
        text_features: torch.Tensor,
        return_align: bool = False,
    ):
        batch_size = x_rsi4.size(0)

        f_rsi = self.rsi_extractor(x_rsi4)
        f_aux = self.aux_extractor(x_aux)
        text_features = F.normalize(text_features, dim=-1)

        sh_rsi = self.proj_sh_rsi(f_rsi)
        sp_rsi = self.proj_sp_rsi(f_rsi)
        sh_aux = self.proj_sh_aux(f_aux)
        sp_aux = self.proj_sp_aux(f_aux)
        sh_txt = self.proj_sh_txt(text_features)
        sp_txt = self.proj_sp_txt(text_features)

        z_rsi = self.cca_head_rsi(sh_rsi)
        z_aux = self.cca_head_aux(sh_aux)
        z_txt = self.cca_head_txt(sh_txt)

        g_rsi = self._gate_scale(self.gate_rsi(torch.cat([sh_rsi, sp_rsi], dim=-1)))
        g_aux = self._gate_scale(self.gate_aux(torch.cat([sh_aux, sp_aux], dim=-1)))
        g_txt = self._gate_scale(self.gate_txt(torch.cat([sh_txt, sp_txt], dim=-1)))

        sh_rsi, sp_rsi = sh_rsi * g_rsi, sp_rsi * g_rsi
        sh_aux, sp_aux = sh_aux * g_aux, sp_aux * g_aux
        sh_txt, sp_txt = sh_txt * g_txt, sp_txt * g_txt

        recon_loss = torch.zeros((), device=x_rsi4.device)
        do_mmm = self.training and random.random() < self.mask_prob

        if do_mmm:
            target_aux = torch.cat([sh_aux, sp_aux], dim=-1).detach()
            target_txt = torch.cat([sh_txt, sp_txt], dim=-1).detach()

            sh_aux = self.mask_token_aux_sh.expand(batch_size, -1)
            sp_aux = self.mask_token_aux_sp.expand(batch_size, -1)
            sh_txt = self.mask_token_txt_sh.expand(batch_size, -1)
            sp_txt = self.mask_token_txt_sp.expand(batch_size, -1)

        fusion_token = sh_txt.unsqueeze(1) + self.query_bias
        modality_tokens = torch.stack(
            [sh_rsi, sp_rsi, sh_aux, sp_aux, sh_txt, sp_txt],
            dim=1,
        )
        kv_seq = torch.cat([fusion_token, modality_tokens], dim=1) + self.modality_embed
        q_seq = kv_seq[:, 0:3, :]

        total_moe_loss = torch.zeros((), device=x_rsi4.device)
        for layer in self.moe_layers:
            q_seq, balance_loss = layer(q_seq, kv_seq)
            total_moe_loss = total_moe_loss + balance_loss
            kv_seq = torch.cat([q_seq, kv_seq[:, 3:, :]], dim=1)

        feat = self.post_norm(q_seq[:, 0, :])
        logits = self.classifier(feat)

        if do_mmm:
            rsi_out = torch.cat([q_seq[:, 1, :], q_seq[:, 2, :]], dim=-1)
            recon_preds = self.mmm_decoder(rsi_out)
            pred_aux, pred_txt = recon_preds.chunk(2, dim=-1)
            recon_loss = F.mse_loss(pred_aux, target_aux) + F.mse_loss(pred_txt, target_txt)

        gate_info = {
            "g_rsi": g_rsi.squeeze(1),
            "g_aux": g_aux.squeeze(1),
            "g_text": g_txt.squeeze(1),
        }

        if return_align:
            align_info = {
                "z_rsi": z_rsi,
                "z_aux": z_aux,
                "z_txt": z_txt,
            }
            ortho_info = {
                "sh_rsi": sh_rsi,
                "sp_rsi": sp_rsi,
                "sh_aux": sh_aux,
                "sp_aux": sp_aux,
                "sh_txt": sh_txt,
                "sp_txt": sp_txt,
            }
            return logits, gate_info, align_info, ortho_info, total_moe_loss, recon_loss

        return logits, gate_info


ODAMNet = MultimodalNet


def build_odamnet(
    num_classes: int,
    aux_in_channels: int,
    **kwargs,
) -> MultimodalNet:
    """Convenience factory for ODAM-Net."""
    return MultimodalNet(
        num_classes=num_classes,
        aux_in_channels=aux_in_channels,
        **kwargs,
    )


__all__ = [
    "FocalLDAMLoss",
    "OrthogonalLoss",
    "CrossModalSupConLoss",
    "RSIFeatureExtractor",
    "AuxFeatureExtractor",
    "GateNet",
    "AsymmetricModalityAwareAttention",
    "Expert",
    "SparseMoE",
    "MoETransformerEncoderLayer",
    "MultimodalNet",
    "ODAMNet",
    "build_odamnet",
]
