from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


ANGULAR_RF_SHAPE = (90, 360)
RAW_RF_PATH_FEATURE_DIM = 17
RAW_RF_GLOBAL_FEATURE_DIM = 7

SPEED_OF_LIGHT = 299792458.0
# dB window for log-compressing raw-linear angular RF power into a [0,1]-ish map.
# Raw angular power spans ~1e-21..1e-7 (W); without this it underflows next to mask/count.
RF_POWER_FLOOR_DB = -200.0
RF_POWER_CEIL_DB = -50.0


def _log_power_to_unit(x: np.ndarray, floor_db: float = RF_POWER_FLOOR_DB, ceil_db: float = RF_POWER_CEIL_DB) -> np.ndarray:
    """Log-compress raw linear RF power -> [0,1] via dB, preserving spatial structure.

    Zero/near-zero power maps to 0; the brightest multipath returns map toward 1.
    """
    x = np.asarray(x, dtype=np.float32)
    db = 10.0 * np.log10(np.maximum(x, 0.0) + 1e-30)
    out = (db - floor_db) / (ceil_db - floor_db)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
RF_DENSE_SLICE = slice(0, 3)
RF_SPARSE_SLICE = slice(3, 6)
RF_MASK_INDEX = 6
RF_COUNT_INDEX = 7


def list_npz_ids(directory: Path) -> set[int]:
    """Return integer frame ids from `*.npz` stems in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return set()

    ids: set[int] = set()
    for path in directory.glob("*.npz"):
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


def _sanitize_float32(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def split_rf_angular_channels(rf):
    """Return dense, sparse, mask, and log-count RF angular components."""
    if getattr(rf, "ndim", 0) >= 4 and rf.shape[-3] == 8:
        return (
            rf[..., RF_DENSE_SLICE, :, :],
            rf[..., RF_SPARSE_SLICE, :, :],
            rf[..., RF_MASK_INDEX, :, :],
            rf[..., RF_COUNT_INDEX, :, :],
        )
    return (
        rf[..., RF_DENSE_SLICE],
        rf[..., RF_SPARSE_SLICE],
        rf[..., RF_MASK_INDEX],
        rf[..., RF_COUNT_INDEX],
    )


def normalize_rf_map_per_frame(x, eps: float = 1e-6):
    """Min/max normalize an RF map independently over the last two spatial axes."""
    if hasattr(x, "amin"):
        min_v = x.amin(dim=(-2, -1), keepdim=True)
        max_v = x.amax(dim=(-2, -1), keepdim=True)
        return (x - min_v) / (max_v - min_v).clamp_min(eps)
    arr = np.asarray(x)
    min_v = np.min(arr, axis=(-2, -1), keepdims=True)
    max_v = np.max(arr, axis=(-2, -1), keepdims=True)
    return (arr - min_v) / np.maximum(max_v - min_v, eps)


def _require_keys(data: np.lib.npyio.NpzFile, npz_path: Path, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing RF key(s) {missing} in {npz_path}")


def _check_angular_shape(rf: np.ndarray, npz_path: Path, channels: int) -> np.ndarray:
    if rf.shape != (*ANGULAR_RF_SHAPE, channels):
        raise ValueError(
            f"Expected angular RF shape {(*ANGULAR_RF_SHAPE, channels)} in {npz_path}, got {rf.shape}"
        )
    return rf


def pack_angular_rf_npz(
    npz_path: Path,
    pack_mode: str = "dense_sparse_mask",
    angular_key: str = "angular_image",
) -> np.ndarray:
    """
    Pack scene-local angular RF npz data into an HWC float32 tensor.

    Supported modes:
      - angular_image: [90, 360, 3]
      - dense_sparse_mask: concat dense/sparse/mask/log-count to [90, 360, 8]
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"RF angular npz not found: {npz_path}")

    with np.load(npz_path) as data:
        if pack_mode == "angular_image":
            _require_keys(data, npz_path, (angular_key,))
            rf = _sanitize_float32(data[angular_key])
            return _check_angular_shape(rf, npz_path, 3)

        if pack_mode == "dense_sparse_mask":
            _require_keys(
                data,
                npz_path,
                ("angular_image", "sparse_angular_image", "mask_map", "count_map"),
            )
            # Log-compress raw-linear power so the dense/sparse channels are not crushed to ~0
            # (they span ~1e-21..1e-7 W; mask/count are O(1), so raw values vanish under norm).
            dense = _log_power_to_unit(_sanitize_float32(data["angular_image"]))
            sparse = _log_power_to_unit(_sanitize_float32(data["sparse_angular_image"]))
            mask = _sanitize_float32(data["mask_map"])[..., None]
            count = np.log1p(np.maximum(_sanitize_float32(data["count_map"]), 0.0))[..., None].astype(np.float32)

            _check_angular_shape(dense, npz_path, 3)
            _check_angular_shape(sparse, npz_path, 3)
            if mask.shape != (*ANGULAR_RF_SHAPE, 1):
                raise ValueError(f"Expected mask_map shape {ANGULAR_RF_SHAPE} in {npz_path}, got {mask.shape[:-1]}")
            if count.shape != (*ANGULAR_RF_SHAPE, 1):
                raise ValueError(f"Expected count_map shape {ANGULAR_RF_SHAPE} in {npz_path}, got {count.shape[:-1]}")

            rf = np.concatenate([dense, sparse, mask, count], axis=-1)
            rf = _sanitize_float32(rf)
            return _check_angular_shape(rf, npz_path, 8)

    raise ValueError(f"Unsupported RF angular pack_mode '{pack_mode}' for {npz_path}")


def _as_1d(data: np.lib.npyio.NpzFile, key: str, npz_path: Path, length: int | None = None) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing raw RF key '{key}' in {npz_path}")
    arr = np.asarray(data[key])
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = arr.reshape(-1)
    if length is not None and arr.shape[0] < length:
        raise ValueError(f"Raw RF key '{key}' in {npz_path} has length {arr.shape[0]}, expected at least {length}")
    return arr


def _as_scalar(data: np.lib.npyio.NpzFile, key: str, default: float = 0.0) -> float:
    if key not in data:
        return default
    value = np.asarray(data[key]).reshape(-1)
    if value.size == 0:
        return default
    value = float(value[0])
    return value if np.isfinite(value) else default


def _scaled_clipped_db(values: np.ndarray) -> np.ndarray:
    return np.clip(values, -200.0, 50.0) / 100.0


def pack_raw_rf_npz(
    npz_path: Path,
    top_k: int = 64,
    sort_by: str = "pdp_power",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack variable-length raw RF paths into padded path and global features.

    Returns:
      path_features: [K, 17] float32
      path_mask: [K] bool
      global_features: [7] float32
      range_m: [K] float32  -- metric bistatic path length per selected path = c * cir_delay
      los_range_m: scalar float32 -- line-of-sight (first-arrival) range = c * min(cir_delay) = |tx - rx|
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Raw RF npz not found: {npz_path}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if sort_by not in {"pdp_power", "per_path_gain_db"}:
        raise ValueError(f"sort_by must be 'pdp_power' or 'per_path_gain_db', got '{sort_by}'")

    features = np.zeros((top_k, RAW_RF_PATH_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((top_k,), dtype=bool)
    range_m = np.zeros((top_k,), dtype=np.float32)
    los_range_m = np.float32(0.0)

    with np.load(npz_path) as data:
        declared_num_paths = int(_as_scalar(data, "num_paths", 0.0))
        cir_coefficients = _as_1d(data, "cir_coefficients", npz_path)
        inferred_num_paths = cir_coefficients.shape[0]
        num_paths = max(0, min(declared_num_paths if declared_num_paths > 0 else inferred_num_paths, inferred_num_paths))

        global_features = np.array(
            [
                np.clip(_as_scalar(data, "path_loss_db", 0.0), -200.0, 50.0) / 100.0,
                np.clip(_as_scalar(data, "total_path_gain", 0.0), -200.0, 50.0) / 100.0,
                np.clip(float(num_paths), 0.0, 4096.0) / 1024.0,
                np.clip(_as_scalar(data, "frequency_hz", 0.0) / 1e9, 0.0, 300.0) / 100.0,
                np.clip(_as_scalar(data, "max_depth", 0.0), 0.0, 100.0) / 100.0,
                np.log1p(max(_as_scalar(data, "samples_per_src", 0.0), 0.0)) / 20.0,
                np.clip(_as_scalar(data, "retry_count", 0.0), 0.0, 100.0) / 100.0,
            ],
            dtype=np.float32,
        )
        global_features = _sanitize_float32(global_features)

        if num_paths == 0:
            return features, mask, global_features, range_m, los_range_m

        cir_delays = _sanitize_float32(_as_1d(data, "cir_delays", npz_path, num_paths)[:num_paths])
        # Line-of-sight (first-arrival) metric range = c * min(delay) = |tx - rx| (bistatic, exact).
        positive_delays = cir_delays[cir_delays > 0]
        if positive_delays.size > 0:
            los_range_m = np.float32(SPEED_OF_LIGHT * positive_delays.min())
        pdp_power = _sanitize_float32(_as_1d(data, "pdp_power", npz_path, num_paths)[:num_paths])
        per_path_gain_db = _sanitize_float32(_as_1d(data, "per_path_gain_db", npz_path, num_paths)[:num_paths])

        _require_keys(data, npz_path, ("aoa", "aod"))
        aoa = _sanitize_float32(data["aoa"])[:num_paths]
        aod = _sanitize_float32(data["aod"])[:num_paths]
        if aoa.shape != (num_paths, 2):
            raise ValueError(f"Expected aoa shape ({num_paths}, 2) in {npz_path}, got {aoa.shape}")
        if aod.shape != (num_paths, 2):
            raise ValueError(f"Expected aod shape ({num_paths}, 2) in {npz_path}, got {aod.shape}")

        cir = np.asarray(cir_coefficients[:num_paths], dtype=np.complex64)
        score = pdp_power if sort_by == "pdp_power" else per_path_gain_db
        valid = np.isfinite(score)
        if not valid.any():
            return features, mask, global_features, range_m, los_range_m

        sorted_valid = np.where(valid)[0][np.argsort(score[valid])[::-1]]
        selected = sorted_valid[:top_k]
        out_n = selected.shape[0]

        delay_ns = cir_delays[selected] * 1e9
        delay_feat = np.log1p(np.maximum(delay_ns, 0.0)) / 10.0
        pdp_power_log = np.log1p(np.maximum(pdp_power[selected], 0.0))
        gain_feat = _scaled_clipped_db(per_path_gain_db[selected])
        cir_selected = cir[selected]
        cir_abs = np.abs(cir_selected)
        cir_phase = np.angle(cir_selected)
        ranks = np.arange(out_n, dtype=np.float32) / float(max(top_k - 1, 1))

        path_features = np.concatenate(
            [
                delay_feat[:, None],
                pdp_power_log[:, None],
                gain_feat[:, None],
                cir_selected.real.astype(np.float32)[:, None],
                cir_selected.imag.astype(np.float32)[:, None],
                np.log1p(cir_abs).astype(np.float32)[:, None],
                np.sin(cir_phase).astype(np.float32)[:, None],
                np.cos(cir_phase).astype(np.float32)[:, None],
                np.sin(aoa[selected]).astype(np.float32),
                np.cos(aoa[selected]).astype(np.float32),
                np.sin(aod[selected]).astype(np.float32),
                np.cos(aod[selected]).astype(np.float32),
                ranks[:, None],
            ],
            axis=-1,
        )
        if path_features.shape[1] != RAW_RF_PATH_FEATURE_DIM:
            raise AssertionError(f"Internal raw RF feature dim mismatch: {path_features.shape[1]}")

        features[:out_n] = _sanitize_float32(path_features)
        mask[:out_n] = True
        # Metric bistatic path length per selected path (meters), aligned with `features`/`mask`.
        range_m[:out_n] = _sanitize_float32(SPEED_OF_LIGHT * cir_delays[selected])
        return features, mask, global_features, range_m, los_range_m
