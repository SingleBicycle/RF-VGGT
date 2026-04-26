import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "vggt"))
sys.path.insert(0, str(PROJECT_ROOT / "vggt" / "training"))

from data.composed_dataset import _convert_optional_sequence_to_tensor
from data.datasets.rf_scene import RFSceneDataset
from data.rf_utils import RAW_RF_GLOBAL_FEATURE_DIM, RAW_RF_PATH_FEATURE_DIM, pack_angular_rf_npz, pack_raw_rf_npz
from loss import compute_camera_loss
from vggt.models.vggt import VGGT


def common_conf(**overrides):
    conf = SimpleNamespace(
        img_size=28,
        patch_size=14,
        rescale=True,
        rescale_aug=False,
        landscape_check=False,
        training=False,
        inside_random=False,
        allow_duplicate_img=True,
        load_depth=False,
        rf_feature_key="angular_image",
        rf_pack_mode="dense_sparse_mask",
        use_raw_rf_paths=False,
        raw_rf_top_k=4,
        raw_rf_sort_by="pdp_power",
        augs=SimpleNamespace(scales=None),
    )
    for key, value in overrides.items():
        setattr(conf, key, value)
    return conf


def write_angular_npz(path: Path, value: float = 1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    dense = np.full((90, 360, 3), value, dtype=np.float32)
    sparse = np.full((90, 360, 3), value * 0.5, dtype=np.float32)
    mask = np.ones((90, 360), dtype=np.float32)
    count = np.ones((90, 360), dtype=np.float32) * 2
    np.savez(
        path,
        angular_image=dense,
        sparse_angular_image=sparse,
        mask_map=mask,
        count_map=count,
    )


def write_raw_npz(path: Path, n: int = 3):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cir_coefficients=(np.arange(n, dtype=np.float32) + 1j * np.arange(n, dtype=np.float32)).astype(np.complex64),
        cir_delays=np.linspace(1e-9, 3e-9, n, dtype=np.float32),
        path_loss_db=np.array(-80.0, dtype=np.float32),
        total_path_gain=np.array(-60.0, dtype=np.float32),
        per_path_gain_db=np.linspace(-90.0, -30.0, n, dtype=np.float32),
        pdp_power=np.linspace(0.1, 0.9, n, dtype=np.float32),
        pdp_delay_s=np.linspace(1e-9, 3e-9, n, dtype=np.float32),
        aoa=np.zeros((n, 2), dtype=np.float32),
        aod=np.ones((n, 2), dtype=np.float32) * 0.5,
        num_paths=np.array(n, dtype=np.int64),
        tx_position=np.zeros(3, dtype=np.float64),
        rx_position=np.zeros(3, dtype=np.float64),
        frequency_hz=np.array(28e9, dtype=np.float64),
        max_depth=np.array(5, dtype=np.int64),
        samples_per_src=np.array(1000, dtype=np.int64),
        min_paths_required=np.array(0, dtype=np.int64),
        retry_count=np.array(1, dtype=np.int64),
    )


def write_empty_raw_npz(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cir_coefficients=np.asarray([], dtype=np.complex64),
        cir_delays=np.asarray([], dtype=np.float32),
        path_loss_db=np.array(-80.0, dtype=np.float32),
        total_path_gain=np.array(-60.0, dtype=np.float32),
        per_path_gain_db=np.asarray([], dtype=np.float32),
        pdp_power=np.asarray([], dtype=np.float32),
        pdp_delay_s=np.asarray([], dtype=np.float32),
        aoa=np.zeros((0, 2), dtype=np.float32),
        aod=np.zeros((0, 2), dtype=np.float32),
        num_paths=np.array(0, dtype=np.int64),
        tx_position=np.zeros(3, dtype=np.float64),
        rx_position=np.zeros(3, dtype=np.float64),
        frequency_hz=np.array(28e9, dtype=np.float64),
        max_depth=np.array(5, dtype=np.int64),
        samples_per_src=np.array(1000, dtype=np.int64),
        min_paths_required=np.array(0, dtype=np.int64),
        retry_count=np.array(1, dtype=np.int64),
    )


def make_scene(tmp_path: Path, num_images: int = 3) -> Path:
    scene = tmp_path / "scene"
    (scene / "images").mkdir(parents=True)
    image_paths = []
    for idx in range(num_images):
        rel = f"images/{idx:06d}.png"
        Image.new("RGB", (42, 28), color=(idx * 20, 10, 30)).save(scene / rel)
        image_paths.append(rel)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], num_images, axis=0)
    for idx in range(num_images):
        extrinsics[idx, 0, 3] = idx * 0.1
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], num_images, axis=0)
    intrinsics[:, 0, 0] = 21.0
    intrinsics[:, 1, 1] = 14.0
    intrinsics[:, 0, 2] = 20.5
    intrinsics[:, 1, 2] = 13.5
    np.savez(
        scene / "cameras.npz",
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        images=np.asarray(image_paths),
        image_size=np.asarray([42, 28], dtype=np.int64),
    )
    return scene


def test_pack_angular_rf_npz(tmp_path):
    path = tmp_path / "000001.npz"
    write_angular_npz(path)
    with np.load(path) as data:
        dense = data["angular_image"].copy()
        sparse = data["sparse_angular_image"].copy()
        mask = data["mask_map"].copy()
        count = data["count_map"].copy()
    dense[0, 0, 0] = np.nan
    sparse[0, 0, 1] = np.inf
    np.savez(path, angular_image=dense, sparse_angular_image=sparse, mask_map=mask, count_map=count)

    angular = pack_angular_rf_npz(path, pack_mode="angular_image")
    packed = pack_angular_rf_npz(path, pack_mode="dense_sparse_mask")
    assert angular.shape == (90, 360, 3)
    assert packed.shape == (90, 360, 8)
    assert np.isfinite(angular).all()
    assert np.isfinite(packed).all()


def test_pack_raw_rf_npz(tmp_path):
    path = tmp_path / "000001.npz"
    write_raw_npz(path, n=3)
    features, mask, global_features = pack_raw_rf_npz(path, top_k=4)
    assert features.shape == (4, RAW_RF_PATH_FEATURE_DIM)
    assert mask.shape == (4,)
    assert global_features.shape == (RAW_RF_GLOBAL_FEATURE_DIM,)
    assert mask.tolist() == [True, True, True, False]
    assert np.isfinite(features).all()

    empty = tmp_path / "000002.npz"
    write_empty_raw_npz(empty)
    features, mask, global_features = pack_raw_rf_npz(empty, top_k=4)
    assert features.shape == (4, RAW_RF_PATH_FEATURE_DIM)
    assert not mask.any()
    assert np.isfinite(global_features).all()


def test_composed_dataset_rf_conversion():
    rf = np.zeros((2, 90, 360, 8), dtype=np.float32)
    tensor = _convert_optional_sequence_to_tensor(rf)
    assert tensor.shape == (2, 8, 90, 360)


def test_rf_scene_dataset_valid_ids(tmp_path):
    scene = make_scene(tmp_path, num_images=3)
    write_angular_npz(scene / "rf_angular_images_gaussian/npz/000001.npz")
    write_angular_npz(scene / "rf_angular_images_gaussian/npz/000002.npz")
    write_raw_npz(scene / "rf/000002.npz", n=2)

    ds = RFSceneDataset(common_conf(), scene_roots=str(scene), require_rf=True)
    assert ds.scenes[0]["valid_ids"] == [1, 2]

    conf = common_conf(use_raw_rf_paths=True)
    ds_raw = RFSceneDataset(conf, scene_roots=str(scene), require_rf=True, use_raw_rf_paths=True)
    assert ds_raw.scenes[0]["valid_ids"] == [2]


def tiny_model(use_raw=False):
    return VGGT(
        img_size=(28, 42),
        patch_size=14,
        embed_dim=64,
        aggregator_kwargs={"depth": 1, "num_heads": 4, "patch_embed": "conv"},
        enable_camera=True,
        enable_depth=False,
        enable_point=False,
        enable_track=False,
        enable_rf=True,
        rf_encoder_type="cnn",
        rf_in_chans=8,
        rf_encoder_hidden_dim=32,
        rf_latent_grid=(1, 2),
        rf_img_size=(90, 360),
        rf_fusion_num_heads=4,
        use_raw_rf_paths=use_raw,
        rf_path_feature_dim=RAW_RF_PATH_FEATURE_DIM if use_raw else None,
        rf_global_feature_dim=RAW_RF_GLOBAL_FEATURE_DIM if use_raw else None,
    )


def test_vggt_forward_angular_rf():
    model = tiny_model(use_raw=False)
    images = torch.rand(1, 2, 3, 28, 42)
    rf = torch.rand(1, 2, 8, 90, 360)
    out = model(images, rf=rf)
    assert "pose_enc_list" in out


def test_vggt_forward_raw_path_optional():
    model = tiny_model(use_raw=True)
    images = torch.rand(1, 2, 3, 28, 42)
    rf = torch.rand(1, 2, 8, 90, 360)
    rf_paths = torch.rand(1, 2, 4, RAW_RF_PATH_FEATURE_DIM)
    rf_path_mask = torch.ones(1, 2, 4, dtype=torch.bool)
    rf_global = torch.rand(1, 2, RAW_RF_GLOBAL_FEATURE_DIM)
    out = model(images, rf=rf, rf_paths=rf_paths, rf_path_mask=rf_path_mask, rf_global=rf_global)
    assert "pose_enc_list" in out


def test_camera_only_loss():
    predictions = {"pose_enc_list": [torch.zeros(1, 2, 9, requires_grad=True)]}
    extrinsics = torch.eye(4)[:3].view(1, 1, 3, 4).repeat(1, 2, 1, 1)
    intrinsics = torch.eye(3).view(1, 1, 3, 3).repeat(1, 2, 1, 1)
    intrinsics[:, :, 0, 0] = 21.0
    intrinsics[:, :, 1, 1] = 14.0
    batch = {
        "images": torch.rand(1, 2, 3, 28, 42),
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    loss = compute_camera_loss(predictions, batch)
    assert "loss_camera" in loss


def test_ablation_scene_local(tmp_path):
    scene = make_scene(tmp_path, num_images=1)
    write_angular_npz(scene / "rf_angular_images_gaussian/npz/000000.npz")
    out_json = tmp_path / "ablation.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'vggt'}:{PROJECT_ROOT / 'vggt' / 'training'}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "vggt" / "training" / "run_rf_encoder_ablation.py"),
        "--scene-root",
        str(scene),
        "--methods",
        "rgb_only",
        "--num-frames",
        "1",
        "--image-height",
        "28",
        "--image-width",
        "42",
        "--steps",
        "1",
        "--device",
        "cpu",
        "--output-json",
        str(out_json),
    ]
    result = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    summary = json_load(out_json)
    assert summary["frame_ids"] == [0]


def json_load(path: Path):
    import json

    return json.loads(path.read_text())
