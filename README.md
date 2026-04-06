# RF-VGGT

This repository packages our RF-augmented VGGT code in a standalone repo for sharing and reproduction.

The codebase starts from the official [facebookresearch/vggt](https://github.com/facebookresearch/vggt) project and adds RF-conditioned camera prediction experiments on top of it.

## What Is Added Here

- RF feature loading in the training dataset through `rf_feature_root` and `rf_feature_key`
- RF encoder modules in `vggt/models/rf_modules.py`
- Gated RF-to-RGB token fusion in `vggt/models/aggregator.py`
- RF-aware model forwarding in `vggt/models/vggt.py`
- RF tensors passed through the training loop in `training/trainer.py`
- Single-scene ablation runner in `training/run_rf_encoder_ablation.py`
- Plotting utility in `training/plot_rf_ablation.py`

The current RF encoder variants are:

- `cnn`
- `shallow_vit`
- `hybrid_vit`

## Install

```bash
git clone https://github.com/SingleBicycle/RF-VGGT.git
cd RF-VGGT
pip install -r requirements.txt
```

If you want to run the plotting script, `matplotlib` is included in `requirements.txt`.

## Expected Data Layout

Scene data is expected to follow the VGGT training format, for example:

```text
scene_root/
├── cameras.npz
└── images/
    ├── 000000.png
    ├── 000001.png
    └── ...
```

RF features are expected under a separate root:

```text
rf_feature_root/
└── AI53_001/
    └── npz/
        ├── 000000.npz
        ├── 000001.npz
        └── ...
```

Each RF `.npz` file should contain the key `angular_image` unless a different key is passed with `--rf-feature-key`.

## Quick RF Ablation Example

Run a quick comparison between RGB-only and RF-enabled variants on one scene:

```bash
python training/run_rf_encoder_ablation.py \
  --scene-root /path/to/dataset/AI53_001 \
  --rf-feature-root /path/to/rf_feature_gaussian \
  --methods rgb_only,rf_cnn,rf_shallow_vit,rf_hybrid_vit \
  --steps 20 \
  --output-json outputs/rf_encoder_ablation_ai53_001.json
```

Plot the resulting loss curves:

```bash
python training/plot_rf_ablation.py \
  --input-json outputs/rf_encoder_ablation_ai53_001.json \
  --output-dir outputs/plots
```

## Notes

- This repo keeps the original VGGT project structure, so most upstream scripts still work unchanged.
- Local datasets and generated outputs are not tracked by git.
- For the original paper, pretrained checkpoints, and upstream documentation, refer to the official VGGT repository.



## Zero-shot Single-view Reconstruction

Our model shows surprisingly good performance on single-view reconstruction, although it was never trained for this task. The model does not need to duplicate the single-view image to a pair, instead, it can directly infer the 3D structure from the tokens of the single view image. Feel free to try it with our demos above, which naturally works for single-view reconstruction.


We did not quantitatively test monocular depth estimation performance ourselves, but [@kabouzeid](https://github.com/kabouzeid) generously provided a comparison of VGGT to recent methods [here](https://github.com/facebookresearch/vggt/issues/36). VGGT shows competitive or better results compared to state-of-the-art monocular approaches such as DepthAnything v2 or MoGe, despite never being explicitly trained for single-view tasks. 



## Runtime and GPU Memory

We benchmark the runtime and GPU memory usage of VGGT's aggregator on a single NVIDIA H100 GPU across various input sizes. 

| **Input Frames** | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|:----------------:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:---:|:---:|
| **Time (s)**     | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| **Memory (GB)**  | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

Note that these results were obtained using Flash Attention 3, which is faster than the default Flash Attention 2 implementation while maintaining almost the same memory usage. Feel free to compile Flash Attention 3 from source to get better performance.


## Research Progression

Our work builds upon a series of previous research projects. If you're interested in understanding how our research evolved, check out our previous works:


<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="left">
      <a href="https://github.com/jytime/Deep-SfM-Revisited">Deep SfM Revisited</a>
    </td>
    <td style="white-space: pre;">──┐</td>
    <td></td>
  </tr>
  <tr>
    <td align="left">
      <a href="https://github.com/facebookresearch/PoseDiffusion">PoseDiffusion</a>
    </td>
    <td style="white-space: pre;">─────►</td>
    <td>
      <a href="https://github.com/facebookresearch/vggsfm">VGGSfM</a> ──►
      <a href="https://github.com/facebookresearch/vggt">VGGT</a>
    </td>
  </tr>
  <tr>
    <td align="left">
      <a href="https://github.com/facebookresearch/co-tracker">CoTracker</a>
    </td>
    <td style="white-space: pre;">──┘</td>
    <td></td>
  </tr>
</table>


## Acknowledgements

Thanks to these great repositories: [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion), [VGGSfM](https://github.com/facebookresearch/vggsfm), [CoTracker](https://github.com/facebookresearch/co-tracker), [DINOv2](https://github.com/facebookresearch/dinov2), [Dust3r](https://github.com/naver/dust3r), [Moge](https://github.com/microsoft/moge), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D](https://github.com/YvanYin/Metric3D) and many other inspiring works in the community.

## Checklist

- [x] Release the training code
- [ ] Release VGGT-500M and VGGT-200M


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

Please note that only this [model checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial) allows commercial usage. This new checkpoint achieves the same performance level (might be slightly better) as the original one, e.g., AUC@30: 90.37 vs. 89.98 on the Co3D dataset.
