import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import extri_intri_to_pose_encoding


def parse_args():
    parser = argparse.ArgumentParser(description="Quick RF encoder ablation on a single scene with camera loss.")
    parser.add_argument("--scene-root", type=str, required=True, help="Path to scene folder, e.g. dataset/AI53_001")
    parser.add_argument(
        "--rf-feature-root",
        type=str,
        default=None,
        help="Root folder containing RF angular features, e.g. dataset/rf_feature_gaussian",
    )
    parser.add_argument("--rf-feature-key", type=str, default="angular_image")
    parser.add_argument(
        "--methods",
        type=str,
        default="rgb_only,rf_cnn,rf_shallow_vit,rf_hybrid_vit",
        help="Comma-separated methods to compare",
    )
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=252)
    parser.add_argument("--image-width", type=int, default=336)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to load with strict=False")
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_frame_ids(num_total: int, num_frames: int) -> np.ndarray:
    if num_frames > num_total:
        raise ValueError(f"Requested {num_frames} frames, but scene only has {num_total}")
    return np.linspace(0, num_total - 1, num_frames, dtype=int)


def load_and_resize_image(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((target_w, target_h), Image.BILINEAR)
    image = np.asarray(image, dtype=np.float32) / 255.0
    return image


def resize_intrinsics(K: np.ndarray, orig_h: int, orig_w: int, target_h: int, target_w: int) -> np.ndarray:
    K = K.copy().astype(np.float32)
    scale_x = target_w / float(orig_w)
    scale_y = target_h / float(orig_h)
    K[0, 0] *= scale_x
    K[0, 2] *= scale_x
    K[1, 1] *= scale_y
    K[1, 2] *= scale_y
    return K


def load_rf_angular_image(rf_feature_root: Path, seq_name: str, frame_id: int, rf_feature_key: str) -> np.ndarray:
    rf_path = rf_feature_root / seq_name / "npz" / f"{frame_id:06d}.npz"
    if not rf_path.is_file():
        raise FileNotFoundError(f"RF feature file not found: {rf_path}")
    with np.load(rf_path) as data:
        if rf_feature_key not in data:
            raise KeyError(f"RF feature key '{rf_feature_key}' not found in {rf_path}")
        return data[rf_feature_key].astype(np.float32)


def build_scene_batch(
    scene_root: Path,
    rf_feature_root: Optional[Path],
    rf_feature_key: str,
    frame_ids: np.ndarray,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    cameras = np.load(scene_root / "cameras.npz")
    extrinsics = cameras["extrinsics"][frame_ids, :3, :4].astype(np.float32)
    intrinsics = cameras["intrinsics"][frame_ids].astype(np.float32)
    image_relpaths = cameras["images"][frame_ids]
    orig_h, orig_w = int(cameras["image_size"][0]), int(cameras["image_size"][1])

    images = []
    intrinsics_resized = []
    rf_frames = []
    seq_name = scene_root.name

    for frame_id, image_relpath, K in zip(frame_ids, image_relpaths, intrinsics):
        image = load_and_resize_image(scene_root / image_relpath, target_h, target_w)
        images.append(image.transpose(2, 0, 1))
        intrinsics_resized.append(resize_intrinsics(K, orig_h, orig_w, target_h, target_w))

        if rf_feature_root is not None:
            rf_frames.append(load_rf_angular_image(rf_feature_root, seq_name, int(frame_id), rf_feature_key))

    images = torch.from_numpy(np.stack(images)).unsqueeze(0).to(device)
    extrinsics = torch.from_numpy(extrinsics).unsqueeze(0).to(device)
    intrinsics = torch.from_numpy(np.stack(intrinsics_resized)).unsqueeze(0).to(device)
    point_masks = torch.ones((1, len(frame_ids), target_h, target_w), dtype=torch.bool, device=device)

    batch = {
        "images": images,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "point_masks": point_masks,
    }
    if rf_frames:
        rf = torch.from_numpy(np.stack(rf_frames)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        batch["rf"] = rf

    return batch


def build_method_model(method: str, target_h: int, target_w: int, checkpoint: Optional[str], device: torch.device) -> VGGT:
    aggregator_kwargs = {
        "depth": 4,
        "num_heads": 6,
        "patch_embed": "conv",
    }

    enable_rf = method != "rgb_only"
    rf_encoder_type = "cnn"
    if method == "rf_shallow_vit":
        rf_encoder_type = "shallow_vit"
    elif method == "rf_hybrid_vit":
        rf_encoder_type = "hybrid_vit"
    elif method == "rf_cnn":
        rf_encoder_type = "cnn"
    elif method != "rgb_only":
        raise ValueError(f"Unknown method: {method}")

    model = VGGT(
        img_size=(target_h, target_w),
        patch_size=14,
        embed_dim=384,
        aggregator_kwargs=aggregator_kwargs,
        enable_camera=True,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_rf=enable_rf,
        rf_encoder_type=rf_encoder_type,
        rf_in_chans=3,
        rf_encoder_hidden_dim=192,
        rf_latent_grid=(2, 4),
        rf_img_size=(90, 360),
        rf_fusion_num_heads=6,
    ).to(device)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        state = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[{method}] loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)}")

    return model


def compute_camera_loss_local(predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    pred_pose_encodings = predictions["pose_enc_list"]
    gt_extrinsics = batch["extrinsics"]
    gt_intrinsics = batch["intrinsics"]
    point_masks = batch["point_masks"]
    image_hw = batch["images"].shape[-2:]

    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics,
        gt_intrinsics,
        image_hw,
        pose_encoding_type="absT_quaR_FoV",
    )

    valid_batch_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    n_stages = len(pred_pose_encodings)
    total_loss_T = total_loss_R = total_loss_FL = 0.0

    for stage_idx, pred_pose_stage in enumerate(pred_pose_encodings):
        stage_weight = 0.6 ** (n_stages - stage_idx - 1)
        if valid_batch_mask.sum() == 0:
            loss_T_stage = (pred_pose_stage * 0).mean()
            loss_R_stage = (pred_pose_stage * 0).mean()
            loss_FL_stage = (pred_pose_stage * 0).mean()
        else:
            pred_valid = pred_pose_stage[valid_batch_mask]
            gt_valid = gt_pose_encoding[valid_batch_mask]
            loss_T_stage = (pred_valid[..., :3] - gt_valid[..., :3]).abs().clamp(max=100).mean()
            loss_R_stage = (pred_valid[..., 3:7] - gt_valid[..., 3:7]).abs().mean()
            loss_FL_stage = (pred_valid[..., 7:] - gt_valid[..., 7:]).abs().mean()
        total_loss_T = total_loss_T + stage_weight * loss_T_stage
        total_loss_R = total_loss_R + stage_weight * loss_R_stage
        total_loss_FL = total_loss_FL + stage_weight * loss_FL_stage

    total_loss_T = total_loss_T / n_stages
    total_loss_R = total_loss_R / n_stages
    total_loss_FL = total_loss_FL / n_stages
    total_camera_loss = total_loss_T + total_loss_R + 0.5 * total_loss_FL

    return {
        "loss_camera": total_camera_loss,
        "loss_T": total_loss_T,
        "loss_R": total_loss_R,
        "loss_FL": total_loss_FL,
    }


def run_method(
    method: str,
    batch: Dict[str, torch.Tensor],
    args,
) -> Dict:
    device = torch.device(args.device)
    set_seed(args.seed)
    model = build_method_model(method, args.image_height, args.image_width, args.checkpoint, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    batch_rf = batch.get("rf")
    if method == "rgb_only":
        batch_rf = None

    losses: List[float] = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images=batch["images"], rf=batch_rf)
        loss_dict = compute_camera_loss_local(predictions, batch)
        loss = loss_dict["loss_camera"]
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        print(f"[{method}] step={step:03d} loss_camera={loss_value:.6f}")

    return {
        "method": method,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "loss_curve": losses,
    }


def main():
    args = parse_args()
    if args.image_height % 14 != 0 or args.image_width % 14 != 0:
        raise ValueError("image-height and image-width must both be divisible by patch_size=14")

    scene_root = Path(args.scene_root)
    rf_feature_root = Path(args.rf_feature_root) if args.rf_feature_root is not None else None
    device = torch.device(args.device)

    cameras = np.load(scene_root / "cameras.npz")
    frame_ids = select_frame_ids(len(cameras["images"]), args.num_frames)
    batch = build_scene_batch(
        scene_root=scene_root,
        rf_feature_root=rf_feature_root,
        rf_feature_key=args.rf_feature_key,
        frame_ids=frame_ids,
        target_h=args.image_height,
        target_w=args.image_width,
        device=device,
    )

    results = []
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    for method in methods:
        if method != "rgb_only" and "rf" not in batch:
            raise ValueError(f"Method {method} requires RF features, but no rf-feature-root was provided")
        results.append(run_method(method, batch, args))

    summary = {
        "scene_root": str(scene_root),
        "rf_feature_root": str(rf_feature_root) if rf_feature_root is not None else None,
        "frame_ids": frame_ids.tolist(),
        "num_frames": args.num_frames,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "steps": args.steps,
        "methods": results,
    }

    print("\nSummary")
    for item in results:
        print(
            f"{item['method']}: initial={item['initial_loss']:.6f} "
            f"final={item['final_loss']:.6f} best={item['best_loss']:.6f}"
        )

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
