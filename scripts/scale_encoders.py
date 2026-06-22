"""Image-blind RF->metric-scale predictors for the encoder comparison (Exp B).

Every encoder consumes the SAME inputs (rf_paths [B,S,K,F], mask [B,S,K], global [B,S,G])
and predicts one log metric-scale per window. They differ only in how the per-path set is
pooled: DeepSets (mean), PointNet (max), Set Transformer (self-attn + learned-query pool).
This isolates the path-encoder architecture as the variable, mapping onto the literature
baselines (Deep Sets NeurIPS'17 / PointNet CVPR'17 / Set Transformer ICML'19).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x, m, dim):
    w = m.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(dim) / w.sum(dim).clamp_min(1.0)


def _masked_max(x, m, dim):
    neg = torch.finfo(x.dtype).min
    xm = x.masked_fill(~m.unsqueeze(-1).bool(), neg)
    out = xm.max(dim)[0]
    # frames with no valid path -> 0
    any_valid = m.bool().any(dim)
    return out.masked_fill(~any_valid.unsqueeze(-1), 0.0)


class ScalePredictor(nn.Module):
    def __init__(self, path_dim=17, global_dim=7, hidden=256, pool="mean",
                 st_layers=2, st_heads=8):
        super().__init__()
        self.pool = pool
        self.path_mlp = nn.Sequential(nn.Linear(path_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.global_mlp = nn.Sequential(nn.Linear(global_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        if pool == "settransformer":
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=st_heads,
                                               dim_feedforward=hidden * 4, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.set_enc = nn.TransformerEncoder(layer, num_layers=st_layers)
            self.query = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
            self.pma = nn.MultiheadAttention(hidden, st_heads, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, 1))
        self.log_scale_bias = nn.Parameter(torch.zeros(1))

    def _pool_paths(self, pf, m):
        # pf [B,S,K,H], m [B,S,K] -> [B,S,H]
        B, S, K, H = pf.shape
        if self.pool == "mean":
            return _masked_mean(pf, m, dim=2)
        if self.pool == "max":
            return _masked_max(pf, m, dim=2)
        if self.pool == "settransformer":
            x = pf.reshape(B * S, K, H)
            mm = m.reshape(B * S, K).bool()
            safe = mm.clone(); safe[~mm.any(1), 0] = True
            x = self.set_enc(x, src_key_padding_mask=~safe)
            q = self.query.expand(B * S, -1, -1)
            pooled, _ = self.pma(q, x, x, key_padding_mask=~safe, need_weights=False)
            return pooled.reshape(B, S, H)
        raise ValueError(self.pool)

    def forward(self, rf_paths, rf_path_mask, rf_global):
        pf = self.path_mlp(rf_paths.float())              # [B,S,K,H]
        pf = self._pool_paths(pf, rf_path_mask)           # [B,S,H]
        gf = self.global_mlp(rf_global.float())           # [B,S,H]
        feat = torch.cat([pf, gf], dim=-1)
        log_s = self.head(feat).squeeze(-1)               # [B,S]
        return log_s.mean(dim=1) + self.log_scale_bias    # [B]
