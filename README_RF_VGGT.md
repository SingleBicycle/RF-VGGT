# RF-VGGT

RF-VGGT extends VGGT with an RF-aware geometry conditioning pathway. RGB frames still go through the VGGT visual geometry backbone, while angular RF maps and raw RF multipath measurements are encoded into RF tokens, synchronized across frames, and injected into VGGT token reasoning through gated cross-attention adapters.

The target training objective is:

```text
L_total =
    camera loss
  + depth loss
  + point loss
  + RF angular consistency loss
  + RF path consistency loss
```

If the selected scenes do not provide depth / point ground truth and `skip_missing_geometry_gt: True`, the depth / point supervised terms become zero. In that case, training effectively uses camera supervision plus the RF consistency losses.

## What This Repo Adds

- `vggt/models/rf_modules.py`: the final RF module stack.
  - Angular RF encoder with separate dense / sparse / confidence stems.
  - Circular azimuth padding to respect 0 / 360 degree continuity.
  - Angular positional features.
  - Raw RF path set encoder with masked attention, summary token, and range histogram token.
  - RF global scalar token encoder.
  - Cross-frame RF scene synchronization transformer.
  - Token-type-aware gated RF adapter.
- `training/data/datasets/rf_scene.py`: scene-local RGB + RF dataset.
- `training/data/rf_utils.py`: RF `.npz` packing utilities.
- `training/losses/rf_consistency_losses.py`: RF angular and RF path consistency losses.
- `training/config/rf_vggt_final_full.yaml`: final full RF-VGGT training config.
- `scripts/smoke_rf_vggt_final_full.py`: synthetic forward / loss / gradient smoke test.

## Environment Setup

Use Python 3.10 or newer. Install the CUDA build of PyTorch that matches your machine, then install the repo and training dependencies.

```bash
cd /path/to/rf_vggt/vggt

conda create -n rf-vggt python=3.10 -y
conda activate rf-vggt
pip install --upgrade pip

# Example for CUDA 12.1. Use the PyTorch index URL that matches your CUDA stack.
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements_train.txt
pip install -e .
```

After the environment is installed, run the smoke test:

```bash
python scripts/smoke_rf_vggt_final_full.py --device cuda
```

`--device cpu` is only useful for a very small code-path check. Real training is intended to run on GPUs.

## Training Data Format

`RFSceneDataset` accepts `rf_scene_roots` in three forms:

- a directory containing multiple scene folders;
- a single scene folder;
- a text file with one scene path per line.

Recommended layout:

```text
/path/to/RF_SCENES/
  AI53_001/
    cameras.npz
    images/
      000000.png
      000001.png
    rf_angular_images_gaussian/
      npz/
        000000.npz
        000001.npz
    rf/
      000000.npz
      000001.npz
    depths/              optional
      000000.npy
    world_points/        optional
      000000.npy
    point_masks/         optional
      000000.npy
    depth_masks/         optional
      000000.npy
```

`cameras.npz` must contain:

```text
images:      [N] relative image paths, e.g. images/000000.png
extrinsics:  [N, 3, 4] or [N, 4, 4], OpenCV camera-from-world convention
intrinsics:  [N, 3, 3]
image_size:  [2] optional, [width, height]
```

Angular RF is read from `rf_angular_images_gaussian/npz` by default. Each frame `.npz` should contain:

```text
angular_image:        [90, 360, 3]
sparse_angular_image: [90, 360, 3]
mask_map:             [90, 360]
count_map:            [90, 360]
```

The loader packs these into:

```text
rf: [S, 8, 90, 360]
```

The 8 channels are:

```text
0:3 dense angular RF
3:6 sparse angular RF
6   mask map
7   log1p(count map)
```

Raw RF paths are read from `rf/` by default. Each frame `.npz` should contain:

```text
cir_coefficients
cir_delays
path_loss_db
total_path_gain
per_path_gain_db
pdp_power
aoa
aod
num_paths
frequency_hz
max_depth
```

The loader packs raw RF into:

```text
rf_paths:     [S, K, 17]
rf_path_mask: [S, K]
rf_global:    [S, 7]
```

The default `K` is 64 and is controlled by `raw_rf_top_k`.

## Training

The trainer uses PyTorch DDP, so launch with `torchrun` even for a single GPU. The wrapper script defaults to one local process and accepts Hydra overrides after the script name.

Start with a small debug run:

```bash
cd /path/to/rf_vggt/vggt

RF_SCENE_ROOTS=/path/to/RF_SCENES \
NPROC_PER_NODE=1 \
bash scripts/train_rf_vggt_final_full.sh \
  max_epochs=1 \
  limit_train_batches=2 \
  limit_val_batches=1 \
  num_workers=0 \
  max_img_per_gpu=1 \
  'data.train.common_config.img_nums=[1,2]' \
  'data.val.common_config.img_nums=[1,2]' \
  logging.log_dir=logs/debug_rf_vggt
```

Full multi-GPU fine-tuning:

```bash
RF_SCENE_ROOTS=/path/to/RF_SCENES \
NPROC_PER_NODE=4 \
bash scripts/train_rf_vggt_final_full.sh \
  checkpoint.resume_checkpoint_path=/path/to/pretrained_vggt_checkpoint.pt \
  logging.log_dir=logs/rf_vggt_final_full \
  max_epochs=20 \
  max_img_per_gpu=4
```

`checkpoint.strict` defaults to `False`, so an RGB-only VGGT checkpoint can be loaded while the newly added RF modules are initialized from scratch.

Logs and checkpoints:

```text
logs/rf_vggt_final_full/
logs/rf_vggt_final_full/ckpts/
logs/rf_vggt_final_full/tensorboard/
```

TensorBoard:

```bash
tensorboard --logdir logs
```

## Key Configs

Main config:

```text
training/config/rf_vggt_final_full.yaml
```

Important fields:

```yaml
rf_scene_roots: /path/to/RF_SCENES
img_size: 336
max_img_per_gpu: 4
geometry:
  enable_depth: True
  enable_point: True
loss:
  camera_weight: 1.0
  depth_weight: 0.5
  point_weight: 0.5
  rf_angular_weight: 0.05
  rf_path_weight: 0.02
  skip_missing_geometry_gt: True
model:
  use_rf: True
  rf:
    encoder_type: final_v2
    use_angular: True
    use_paths: True
    use_global: True
```

If the current data has no depth / point GT, keep `skip_missing_geometry_gt: True` and keep at least one geometry prediction head enabled. The RF consistency losses need predicted geometry: they use `world_points` from the point head when available, or backproject the depth head output using intrinsics.

If memory is tight and there is no point GT, you can disable the point head while keeping the depth head:

```bash
bash scripts/train_rf_vggt_final_full.sh \
  geometry.enable_point=False \
  model.enable_point=False \
  loss.point_weight=0.0 \
  data.train.common_config.load_world_points=False \
  data.train.common_config.load_point_masks=False \
  data.val.common_config.load_world_points=False \
  data.val.common_config.load_point_masks=False
```

The older, lighter RF few-shot config is:

```text
training/config/rf_fewshot_final.yaml
```

That config uses the previous hybrid RF encoder path and disables raw RF paths by default.

## Can We Start Training Now?

You can start with debug training once the Python environment is installed and `scripts/smoke_rf_vggt_final_full.py` passes.

In this local workspace, the sample data under `dataset/AI53_001_new` currently has:

```text
100 RGB images
99 angular RF frames
98 raw RF frames
0 depth / point GT folders detected
```

This is enough for camera + RF consistency fine-tuning. It is not enough to exercise the full supervised depth / point terms until `depths/`, `world_points/`, and the corresponding masks are added.

## Common Failure Modes

- `ModuleNotFoundError: torch`: PyTorch is not installed, or the conda environment is not activated.
- `Please set the RANK and LOCAL_RANK environment variables`: training was launched with plain `python`; use `torchrun`.
- `No RF scenes have valid frames after RF filtering`: the path is wrong, or the frame IDs do not overlap across images, angular RF, and raw RF.
- `Missing RF key(s)`: check the angular RF `.npz` keys and shapes.
- CUDA OOM: reduce `max_img_per_gpu`, reduce `data.*.common_config.img_nums`, or increase `accum_steps`.
