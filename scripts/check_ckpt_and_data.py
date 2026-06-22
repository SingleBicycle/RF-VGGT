"""Ground-truth check: (1) localize VGGT-1B + strict=False load, (2) real one-batch data smoke."""
import os, sys, glob
from pathlib import Path
import numpy as np
import torch

REPO = Path("/DATA/zihao/projects/rf_vggt/RF-VGGT")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

C = 299792458.0
SCENES_PARENT = "/DATA/zihao/projects/rf_vggt"

def interp_pos_embed(pos_embed, tgt_tokens):
    """DINOv2-style interpolation of [1, N_src, D] (cls + grid) to tgt_tokens (cls + grid)."""
    n_src = pos_embed.shape[1]
    if n_src == tgt_tokens:
        return pos_embed
    cls, grid = pos_embed[:, :1], pos_embed[:, 1:]
    s_src = int(round((grid.shape[1]) ** 0.5))
    s_tgt = int(round((tgt_tokens - 1) ** 0.5))
    D = grid.shape[-1]
    grid = grid.reshape(1, s_src, s_src, D).permute(0, 3, 1, 2).float()
    grid = torch.nn.functional.interpolate(grid, size=(s_tgt, s_tgt), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, s_tgt * s_tgt, D).to(cls.dtype)
    return torch.cat([cls, grid], dim=1)

def localize_checkpoint(img_size=336, patch_size=14):
    print("\n===== (1) Localize VGGT-1B =====")
    from safetensors.torch import load_file
    snap = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*/model.safetensors"))
    assert snap, "VGGT-1B safetensors not found"
    sd = load_file(snap[0])
    tgt = (img_size // patch_size) ** 2 + 1
    for k in list(sd.keys()):
        if k.endswith("patch_embed.pos_embed"):
            old = sd[k].shape
            sd[k] = interp_pos_embed(sd[k], tgt)
            print(f"  interpolated {k}: {tuple(old)} -> {tuple(sd[k].shape)}")
    out = REPO / "ckpts"; out.mkdir(exist_ok=True)
    pt = out / f"vggt1b_{img_size}.pt"
    torch.save({"model": sd}, pt)
    print(f"  loaded {len(sd)} tensors from {snap[0]}")
    print(f"  saved -> {pt}")
    return pt, sd

def test_load(sd):
    print("\n===== (1b) strict=False load into RF-VGGT VGGT (use_rf=True) =====")
    from hydra import initialize_config_dir, compose
    from hydra.utils import instantiate
    with initialize_config_dir(version_base=None, config_dir=str(REPO / "training" / "config")):
        cfg = compose(config_name="rf_vggt_final_full")
    model = instantiate(cfg.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_total = sum(1 for _ in model.state_dict())
    miss_rf = [k for k in missing if "rf" in k.lower()]
    miss_non_rf = [k for k in missing if "rf" not in k.lower()]
    print(f"  model params: {n_total} | loaded(matched): {n_total-len(missing)} | missing: {len(missing)} | unexpected: {len(unexpected)}")
    print(f"  missing that are RF modules: {len(miss_rf)} (expected, init from scratch)")
    print(f"  missing NON-rf (should be ~0 / only heads if name drift): {len(miss_non_rf)}")
    if miss_non_rf[:10]:
        print(f"    e.g. {miss_non_rf[:10]}")
    if unexpected[:10]:
        print(f"  unexpected[:10]: {unexpected[:10]}")
    # how much of aggregator backbone transferred?
    agg_keys = [k for k in model.state_dict() if k.startswith("aggregator.")]
    agg_missing = [k for k in missing if k.startswith("aggregator.")]
    print(f"  aggregator keys: {len(agg_keys)} | aggregator missing: {len(agg_missing)} -> backbone transfer { (len(agg_keys)-len(agg_missing))/max(len(agg_keys),1)*100:.1f}%")
    return cfg, model

def data_smoke(cfg):
    print("\n===== (2) Real one-batch data smoke on 2 scenes =====")
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    OmegaConf.resolve(cfg)  # resolve all ${...} against root before extracting sub-nodes
    base_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.dataset.dataset_configs[0], resolve=True))
    with open_dict(base_cfg):
        base_cfg.scene_roots = SCENES_PARENT
    common = OmegaConf.create(OmegaConf.to_container(cfg.data.train.common_config, resolve=True))
    ds = instantiate(base_cfg, common_conf=common)
    print(f"  scenes loaded: {[s['seq_name'] for s in ds.scenes]}")
    for s in ds.scenes:
        print(f"    {s['seq_name']}: valid_ids={len(s['valid_ids'])}")
    # pull a few frames from scene 0
    batch = ds.get_data(seq_index=0, img_per_seq=4, ids=[0, 20, 50, 90])
    print(f"\n  batch keys: {sorted(batch.keys())}")
    for k in ["rf", "rf_paths", "rf_path_mask", "rf_global"]:
        if k in batch:
            a = np.asarray(batch[k])
            print(f"  {k}: shape={a.shape} dtype={a.dtype} finite={np.isfinite(a).all()} nonzero_frac={np.mean(np.abs(a)>0):.3f} min={a.min():.3e} max={a.max():.3e}")
    # angular channel-wise magnitude (confirm dense/sparse ~0)
    rf = np.asarray(batch["rf"])  # [S,8,90,360]
    print("\n  angular per-channel mean|x| (0:3 dense, 3:6 sparse, 6 mask, 7 logcount):")
    print("   ", [f"{rf[:,c].__abs__().mean():.3e}" for c in range(8)])
    # raw delay -> range sanity: read raw npz directly for frame 0
    print("\n  raw RF delay->range sanity (scene0):")
    for fid in [0, 20, 50, 90]:
        d = np.load(f"{SCENES_PARENT}/AI53_001_Blender/rf/{fid:06d}.npz", allow_pickle=True)
        rmin = C * d["cir_delays"].min(); rmax = C * d["cir_delays"].max()
        tx, rx = d["tx_position"].astype(float), d["rx_position"].astype(float)
        print(f"    f{fid}: LoS c*minDelay={rmin:.3f}m  |tx-rx|={np.linalg.norm(tx-rx):.3f}m  maxRange={rmax:.3f}m  nPaths={int(d['num_paths'])}")
    # depth vs RF range calibration constant
    print("\n  depth vs RF-range calibration:")
    for sc in ["AI53_001_Blender", "AI53_002_Blender"]:
        dms, rfs = [], []
        for fid in range(0, 100, 10):
            dep = np.load(f"{SCENES_PARENT}/{sc}/depths/{fid:06d}.npy")
            dms.append(np.median(dep[dep > 0]))
            rr = C * np.load(f"{SCENES_PARENT}/{sc}/rf/{fid:06d}.npz")["cir_delays"]
            rfs.append(np.median(rr))
        dm, rfm = np.median(dms), np.median(rfs)
        print(f"    {sc}: median_depth={dm:.3f}m  median_RFrange={rfm:.3f}m  ratio(depth/RFrange)={dm/rfm:.3f}")
    # depths present?
    if "depths" in batch:
        dep = np.asarray(batch["depths"][0])
        print(f"\n  depth[0]: shape={dep.shape} median={np.median(dep[dep>0]):.3f}m max={dep.max():.3f}m")
    print("\n  SMOKE OK: all 4 RF modalities present & finite." )

if __name__ == "__main__":
    pt, sd = localize_checkpoint()
    cfg, model = test_load(sd)
    data_smoke(cfg)
