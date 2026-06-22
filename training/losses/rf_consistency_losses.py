import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _zero_from_predictions(predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    for container in (predictions, batch):
        for value in container.values():
            if torch.is_tensor(value) and value.is_floating_point():
                return value.sum() * 0.0
    return torch.tensor(0.0)


def _normalize_map_per_frame(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_v = x.amin(dim=(-2, -1), keepdim=True)
    max_v = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_v) / (max_v - min_v).clamp_min(eps)


def _world_to_camera(points: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    if extrinsics.shape[-2:] == (4, 4):
        extrinsics = extrinsics[..., :3, :4]
    R = extrinsics[..., :3]
    t = extrinsics[..., 3]
    return torch.einsum("bsij,bshwj->bshwi", R, points) + t[:, :, None, None, :]


def _backproject_depth(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    if depth.shape[-1] == 1:
        depth = depth[..., 0]
    B, S, H, W = depth.shape
    device = depth.device
    dtype = depth.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    fx = intrinsics[..., 0, 0].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    z = depth
    x = (xx.view(1, 1, H, W) - cx[:, :, None, None]) / fx[:, :, None, None] * z
    y = (yy.view(1, 1, H, W) - cy[:, :, None, None]) / fy[:, :, None, None] * z
    return torch.stack([x, y, z], dim=-1)


def get_pred_points_for_rf_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    pose_source: str = "gt_first",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    points = None
    points_are_world = False
    if "world_points" in predictions:
        points = predictions["world_points"]
        points_are_world = True
    elif "point_map" in predictions:
        points = predictions["point_map"]
    elif "pred_points" in predictions:
        points = predictions["pred_points"]
    elif "depth" in predictions and "intrinsics" in batch and batch["intrinsics"] is not None:
        points = _backproject_depth(predictions["depth"], batch["intrinsics"])
    else:
        return None, None, "missing_pred_points_for_rf"

    points = torch.nan_to_num(points.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if points.shape[-1] != 3:
        return None, None, "invalid_pred_points_shape"

    mask = torch.isfinite(points).all(dim=-1) & (torch.linalg.norm(points, dim=-1) > 1e-6)
    if "point_masks" in batch and batch["point_masks"] is not None and batch["point_masks"].shape == mask.shape:
        mask = mask & batch["point_masks"].to(device=mask.device, dtype=torch.bool)

    if "rf_T_camera" in batch and batch["rf_T_camera"] is not None:
        # Optional calibration path. The common case is RF aligned to camera frame.
        T = batch["rf_T_camera"].to(device=points.device, dtype=points.dtype)
        if T.shape[-2:] == (4, 4):
            homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = torch.einsum("bsij,bshwj->bshwi", T, homo)[..., :3]
    elif "camera_T_rf" in batch and batch["camera_T_rf"] is not None:
        T = torch.linalg.inv(batch["camera_T_rf"].to(device=points.device, dtype=points.dtype))
        homo = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        points = torch.einsum("bsij,bshwj->bshwi", T, homo)[..., :3]
    elif points_are_world and pose_source == "gt_first" and "extrinsics" in batch and batch["extrinsics"] is not None:
        points = _world_to_camera(points, batch["extrinsics"].to(device=points.device, dtype=points.dtype))

    return points, mask, "ok"


def _select_valid_points(points: torch.Tensor, mask: torch.Tensor, max_points: int) -> torch.Tensor:
    valid_points = points[mask]
    if valid_points.numel() == 0:
        return valid_points.reshape(0, 3)
    if valid_points.shape[0] > max_points:
        idx = torch.linspace(0, valid_points.shape[0] - 1, max_points, device=valid_points.device).long()
        valid_points = valid_points[idx]
    return valid_points


def project_points_to_rf_angular(points: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = points.unbind(dim=-1)
    radius = torch.sqrt((points * points).sum(dim=-1).clamp_min(eps))
    azimuth = torch.atan2(x, z)
    elevation = torch.asin((y / radius).clamp(-1.0 + eps, 1.0 - eps))
    return azimuth, elevation, radius


def soft_splat_angular_map(
    points: torch.Tensor,
    mask: torch.Tensor,
    angular_grid: Tuple[int, int] = (90, 360),
    max_points_per_view: int = 4096,
) -> torch.Tensor:
    B, S, H_img, W_img, _ = points.shape
    H_ang, W_ang = angular_grid
    out = points.new_zeros(B * S, H_ang * W_ang)
    flat_points = points.reshape(B * S, H_img * W_img, 3)
    flat_mask = mask.reshape(B * S, H_img * W_img)

    for frame_idx in range(B * S):
        pts = _select_valid_points(flat_points[frame_idx], flat_mask[frame_idx], max_points_per_view)
        if pts.numel() == 0:
            continue
        az, el, radius = project_points_to_rf_angular(pts)
        col = (az + math.pi) / (2.0 * math.pi) * W_ang
        row = (el + math.pi / 2.0) / math.pi * (H_ang - 1)
        row = row.clamp(0.0, float(H_ang - 1) - 1e-4)
        col_floor = torch.floor(col)
        row_floor = torch.floor(row)
        c0 = col_floor.long().remainder(W_ang)
        c1 = (c0 + 1).remainder(W_ang)
        r0 = row_floor.long().clamp(0, H_ang - 1)
        r1 = (r0 + 1).clamp(0, H_ang - 1)
        dc = col - col_floor
        dr = row - row_floor
        base_weight = (1.0 / radius.clamp_min(1e-3)).clamp(max=10.0)
        weights = (
            ((1.0 - dr) * (1.0 - dc), r0, c0),
            ((1.0 - dr) * dc, r0, c1),
            (dr * (1.0 - dc), r1, c0),
            (dr * dc, r1, c1),
        )
        for w, rr, cc in weights:
            out[frame_idx].scatter_add_(0, rr * W_ang + cc, w * base_weight)

    out = out.view(B, S, 1, H_ang, W_ang)
    out = out / out.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return out


def build_rf_angular_target(rf: torch.Tensor, angular_grid: Tuple[int, int] = (90, 360)) -> torch.Tensor:
    dense_energy = torch.linalg.norm(rf[:, :, 0:3].float(), dim=2)
    sparse_energy = torch.linalg.norm(rf[:, :, 3:6].float(), dim=2)
    mask_map = rf[:, :, 6].float().clamp(0.0, 1.0)
    count_map = rf[:, :, 7].float()
    dense_energy = _normalize_map_per_frame(dense_energy)
    sparse_energy = _normalize_map_per_frame(sparse_energy)
    count_map = _normalize_map_per_frame(count_map)
    target = (0.35 * mask_map + 0.25 * count_map + 0.25 * dense_energy + 0.15 * sparse_energy).clamp(0.0, 1.0)
    if target.shape[-2:] != angular_grid:
        target = F.interpolate(target.reshape(-1, 1, *target.shape[-2:]), size=angular_grid, mode="bilinear", align_corners=False)
        target = target.reshape(rf.shape[0], rf.shape[1], *angular_grid)
    return target.unsqueeze(2)


def _soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.flatten(1)
    target_f = target.flatten(1)
    inter = (pred_f * target_f).sum(dim=1)
    denom = pred_f.sum(dim=1) + target_f.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def _pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_f = pred.flatten(1)
    target_f = target.flatten(1)
    pred_f = pred_f - pred_f.mean(dim=1, keepdim=True)
    target_f = target_f - target_f.mean(dim=1, keepdim=True)
    corr = (pred_f * target_f).sum(dim=1) / (
        torch.sqrt((pred_f * pred_f).sum(dim=1).clamp_min(eps))
        * torch.sqrt((target_f * target_f).sum(dim=1).clamp_min(eps))
    )
    return 1.0 - corr.mean()


class RFAngularConsistencyLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.4,
        dice_weight: float = 0.3,
        corr_weight: float = 0.3,
        max_points_per_view: int = 4096,
        angular_grid=(90, 360),
        pose_source: str = "gt_first",
        frame_mode: str = "camera",
        **kwargs,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.corr_weight = float(corr_weight)
        self.max_points_per_view = int(max_points_per_view)
        self.angular_grid = (int(angular_grid[0]), int(angular_grid[1]))
        self.pose_source = pose_source
        self.frame_mode = frame_mode

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zero = _zero_from_predictions(predictions, batch)
        if "rf" not in batch or batch["rf"] is None:
            return {"loss_rf_angular": zero, "rf_angular_missing_rf": zero + 1.0}
        points, mask, status = get_pred_points_for_rf_loss(predictions, batch, pose_source=self.pose_source)
        if points is None or mask is None:
            return {"loss_rf_angular": zero, status: zero + 1.0}

        pred_ang = soft_splat_angular_map(points, mask, self.angular_grid, self.max_points_per_view)
        target_ang = build_rf_angular_target(batch["rf"].to(device=pred_ang.device, dtype=pred_ang.dtype), self.angular_grid)
        bce = F.binary_cross_entropy(pred_ang.clamp(1e-5, 1.0 - 1e-5), target_ang)
        dice = _soft_dice_loss(pred_ang, target_ang)
        corr = _pearson_corr_loss(pred_ang, target_ang)
        loss = self.bce_weight * bce + self.dice_weight * dice + self.corr_weight * corr
        return {
            "loss_rf_angular": loss,
            "rf_angular_bce": bce.detach(),
            "rf_angular_dice": dice.detach(),
            "rf_angular_corr": corr.detach(),
            "rf_angular_pred_mean": pred_ang.mean().detach(),
            "rf_angular_target_mean": target_ang.mean().detach(),
        }


def build_range_histogram(
    values: torch.Tensor,
    weights: torch.Tensor,
    range_min: float = 0.0,
    range_max: float = 20.0,
    range_bins: int = 64,
    hist_sigma: float = 0.15,
) -> torch.Tensor:
    centers = torch.linspace(range_min, range_max, range_bins, device=values.device, dtype=values.dtype)
    sigma = max(float(hist_sigma), 1e-4)
    kernel = torch.exp(-0.5 * ((values.unsqueeze(-1) - centers) / sigma) ** 2)
    hist = (kernel * weights.unsqueeze(-1)).sum(dim=-2)
    return hist / hist.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class RFPathConsistencyLoss(nn.Module):
    def __init__(
        self,
        path_delay_index: int = 0,
        path_amplitude_index: Optional[int] = None,
        delay_to_meter: float = 1.0,
        range_min: float = 0.0,
        range_max: float = 50.0,
        range_bins: int = 64,
        hist_sigma: float = 0.5,
        max_points_per_view: int = 4096,
        pose_source: str = "gt_first",
        range_to_depth_scale: float = 0.37,
        **kwargs,
    ) -> None:
        super().__init__()
        self.path_delay_index = int(path_delay_index)
        self.path_amplitude_index = None if path_amplitude_index is None else int(path_amplitude_index)
        self.delay_to_meter = float(delay_to_meter)
        self.range_min = float(range_min)
        self.range_max = float(range_max)
        self.range_bins = int(range_bins)
        self.hist_sigma = float(hist_sigma)
        self.max_points_per_view = int(max_points_per_view)
        self.pose_source = pose_source
        # RF multipath ranges are bistatic path lengths (~2.7x one-way camera depth);
        # this calibration maps RF range -> comparable camera-depth range so the histograms align.
        self.range_to_depth_scale = float(range_to_depth_scale)

    def _pred_hist(self, points: torch.Tensor, mask: torch.Tensor, frame_scale: torch.Tensor) -> torch.Tensor:
        B, S, H, W, _ = points.shape
        flat_points = points.reshape(B * S, H * W, 3)
        flat_mask = mask.reshape(B * S, H * W)
        flat_scale = frame_scale.reshape(B * S)
        hists = []
        for frame_idx in range(B * S):
            pts = _select_valid_points(flat_points[frame_idx], flat_mask[frame_idx], self.max_points_per_view)
            if pts.numel() == 0:
                hists.append(points.new_zeros(self.range_bins))
                continue
            # Predicted point ranges in METERS = unit-scale range * predicted metric scale.
            ranges = torch.linalg.norm(pts, dim=-1) * flat_scale[frame_idx]
            weights = torch.ones_like(ranges)
            hists.append(
                build_range_histogram(
                    ranges[None],
                    weights[None],
                    self.range_min,
                    self.range_max,
                    self.range_bins,
                    self.hist_sigma,
                )[0]
            )
        return torch.stack(hists, dim=0).reshape(B, S, self.range_bins)

    def _target_hist_from_ranges(self, range_m: torch.Tensor, rf_path_mask: torch.Tensor) -> torch.Tensor:
        # range_m [B,S,K] are metric bistatic path lengths; calibrate to comparable camera depth.
        ranges = range_m.float().clamp_min(0.0) * self.range_to_depth_scale
        weight = rf_path_mask.to(device=range_m.device, dtype=ranges.dtype)
        B, S, K = ranges.shape
        return build_range_histogram(
            ranges.reshape(B * S, K),
            weight.reshape(B * S, K),
            self.range_min,
            self.range_max,
            self.range_bins,
            self.hist_sigma,
        ).reshape(B, S, self.range_bins)

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        zero = _zero_from_predictions(predictions, batch)
        if "rf_path_range_m" not in batch or batch["rf_path_range_m"] is None \
                or "rf_path_mask" not in batch or batch["rf_path_mask"] is None:
            return {"loss_rf_path": zero}
        points, mask, status = get_pred_points_for_rf_loss(predictions, batch, pose_source=self.pose_source)
        if points is None or mask is None:
            return {"loss_rf_path": zero, status: zero + 1.0}

        B, S = points.shape[:2]
        # Per-frame predicted metric scale: use RF-predicted log_scale if available, else align
        # the predicted range median to the (calibrated) RF range median so the loss is scale-consistent.
        if "rf_log_scale" in predictions and predictions["rf_log_scale"] is not None:
            frame_scale = torch.exp(predictions["rf_log_scale"].float().detach()).reshape(B, 1).expand(B, S)
        else:
            frame_scale = points.new_ones(B, S)

        pred_hist = self._pred_hist(points, mask, frame_scale)
        target_hist = self._target_hist_from_ranges(
            batch["rf_path_range_m"].to(device=points.device, dtype=points.dtype),
            batch["rf_path_mask"].to(device=points.device),
        )
        l1 = F.smooth_l1_loss(pred_hist, target_hist)
        eps = 1e-6
        p = pred_hist.clamp_min(eps)
        q = target_hist.clamp_min(eps)
        m = 0.5 * (p + q)
        js = 0.5 * (p * (p.log() - m.log())).sum(dim=-1).mean() + 0.5 * (q * (q.log() - m.log())).sum(dim=-1).mean()
        emd = torch.abs(torch.cumsum(pred_hist, dim=-1) - torch.cumsum(target_hist, dim=-1)).mean()
        loss = 0.5 * l1 + 0.3 * js + 0.2 * emd
        return {
            "loss_rf_path": loss,
            "rf_path_l1": l1.detach(),
            "rf_path_js": js.detach(),
            "rf_path_emd": emd.detach(),
            "rf_path_pred_mean": pred_hist.mean().detach(),
            "rf_path_target_mean": target_hist.mean().detach(),
        }
