"""stream_dataset.py — scene-streaming IterableDataset for I/O-bound training.

Why this exists
---------------
The original ``AVSEDataset`` is map-style: ``__getitem__(idx)`` decodes one *window* by opening that
window's 3 files (``*_mixed.wav``, ``*_target.wav``, ``<scene>.npy``) from scratch. With the DataLoader's
global shuffle, consecutive windows come from *different* scenes, so every window pays a fresh
open/seek/header-parse. The train split has ~315k windows over ~34.5k scenes (~9 windows/scene), so each
scene's files get re-opened ~9x per epoch and re-resampled ~9x. On this machine that pegged the dataset
SSD (random small reads, ~40 MB/s, queue ~4) and starved the GPU ~50% of the time.

The fix is structural, not a cache bolt-on: make the unit of work a **scene**, not a window. Each scene's
3 files are opened **once per epoch**, read whole (sequentially) into RAM, resampled once, then all of that
scene's windows are sliced from memory. Shuffling for SGD is preserved with a bounded **sliding scene
pool**: keep ~``scene_buffer`` scenes' raw arrays resident and draw windows at random from their pooled
windows. This decorrelates a scene's ~9 overlapping windows across many batches while holding memory flat
(only raw arrays are buffered; decoded tensors live only for the instant they are yielded).

Per-window decode (slice + the joint ``0.8/max`` normalization) is byte-for-byte identical to
``AVSEDataset.__getitem__`` so training numerics are unchanged — only the *I/O access pattern* differs.
"""
from __future__ import annotations

import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info

from .dataset import AVSEDataset


# ---- pure decode helpers (operate on whole in-memory scene arrays) -------------------------------

def _load_full_audio(path: str, target_sr: int) -> torch.Tensor:
    """Read an entire wav once -> mono [1, T] at ``target_sr`` (resampled once if needed)."""
    wav_np, sr = sf.read(path, dtype="float32")            # [T] or [T, C]
    wav = torch.from_numpy(wav_np).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)                             # [T] -> [1, T]
    else:
        wav = wav.T                                        # [T, C] -> [C, T]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)                # to mono
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav                                             # [1, T] in target-sr coords


def _slice_audio(full: torch.Tensor, start: int, num: int) -> torch.Tensor:
    """Window [start, start+num) from a whole [1, T] waveform; right-pad with zeros if short."""
    seg = full[:, start:start + num]
    if seg.shape[1] < num:
        seg = F.pad(seg, (0, num - seg.shape[1]))
    return seg.contiguous()


def _slice_video(arr: np.ndarray, start: int, num: int, input_size) -> torch.Tensor:
    """Window [start, start+num) from a whole [T, H, W] uint8 array -> float [num, H, W] / 255.

    Matches ``AVSEDataset._load_video_window``: short tails repeat the last frame; an empty array
    zero-pads to ``input_size``.
    """
    total = arr.shape[0]
    end = start + num
    if end <= total:
        window = np.array(arr[start:end])                  # copy out
    else:
        avail = max(0, total - start)
        pad_len = num - avail
        if avail > 0:
            head = np.array(arr[start:total])
            last = np.array(arr[total - 1:total])          # [1, H, W]
            pad = np.repeat(last, pad_len, axis=0)
            window = np.concatenate([head, pad], axis=0)
        elif total > 0:
            last = np.array(arr[total - 1:total])
            window = np.repeat(last, num, axis=0)
        else:
            window = np.zeros((num, *tuple(input_size)), dtype=arr.dtype)
    return torch.from_numpy(np.ascontiguousarray(window)).float() / 255.0


class AVSESceneStreamDataset(IterableDataset):
    """Scene-streaming view over an already-built ``AVSEDataset`` (reuses its cached window scan).

    Each epoch: scenes are sharded across workers (disjoint), optionally shuffled, and streamed through
    a bounded sliding pool. Every scene's 3 files are opened exactly once per epoch.
    """

    def __init__(self, base: AVSEDataset, shuffle: bool, seed: int = 42,
                 scene_buffer: int = 256, pool_mb: float = 160.0, max_windows: int = 0):
        super().__init__()
        self.scenes_dir = base.scenes_dir
        self.video_npy_dir = base.video_npy_dir
        self.window_samples = base.window_samples
        self.window_frames = base.window_frames
        self.target_sr = base.config.audio.sample_rate
        self.input_size = tuple(base.config.video.input_size)

        self.shuffle = shuffle
        self.seed = seed
        # The sliding scene pool is bounded by BYTES, not scene count: LRS3 scene lengths are long-tailed
        # (a 74 s scene is ~25 MB resident: 16 MB video + ~9 MB audio), so a fixed scene count blew up RAM
        # when many long scenes clustered in one epoch's shuffle (per-worker × num_workers). ``pool_mb`` is
        # the per-worker resident budget; ``scene_buffer`` is a secondary count cap. At least one scene is
        # always loaded so progress is guaranteed even if a single scene exceeds the budget.
        self.scene_buffer = max(1, scene_buffer) if shuffle else 1
        self._pool_budget_bytes = int(max(16.0, pool_mb) * 1024 * 1024)

        # Group windows by scene, preserving each scene's start-ordered window list. ``base.windows``
        # is already scene-contiguous and start-ordered from the scan, but group defensively.
        groups: "OrderedDict[str, List[Dict]]" = OrderedDict()
        for w in base.windows:
            groups.setdefault(w["scene_id"], []).append(w)

        # Optional cap (subset / val): keep whole scenes in order until the window budget is hit,
        # truncating the last scene so the total is exactly ``max_windows`` and fully deterministic.
        if max_windows and max_windows > 0:
            trimmed: "OrderedDict[str, List[Dict]]" = OrderedDict()
            remaining = max_windows
            for sid, ws in groups.items():
                if remaining <= 0:
                    break
                take = ws[:remaining]
                trimmed[sid] = take
                remaining -= len(take)
            groups = trimmed

        self.scene_ids: List[str] = list(groups.keys())
        self.scene_windows = groups
        self._total = sum(len(v) for v in groups.values())

        # Per-epoch shuffle seed: workers persist across epochs, so each worker advances its own local
        # epoch counter on every __iter__ (called once per epoch). ``_base_epoch`` (set via set_epoch
        # before the workers spawn) aligns this with --resume so a resumed run shuffles like a fresh one.
        self._base_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call BEFORE the DataLoader spawns workers (e.g. at resume) to align the shuffle schedule."""
        self._base_epoch = int(epoch)

    def __len__(self) -> int:
        return self._total

    def _read_scene_arrays(self, sid: str):
        mixed = _load_full_audio(os.path.join(self.scenes_dir, f"{sid}_mixed.wav"), self.target_sr)
        target = _load_full_audio(os.path.join(self.scenes_dir, f"{sid}_target.wav"), self.target_sr)
        video = np.load(os.path.join(self.video_npy_dir, f"{sid}.npy"))   # full sequential read, uint8
        return mixed, target, video

    @staticmethod
    def _arrays_nbytes(arrays) -> int:
        mixed, target, video = arrays
        return (mixed.element_size() * mixed.nelement()
                + target.element_size() * target.nelement()
                + video.nbytes)

    def _decode(self, sid: str, arrays, w: Dict) -> Dict[str, torch.Tensor]:
        mixed, target, video = arrays
        mw = _slice_audio(mixed, w["start_sample"], self.window_samples)
        tw = _slice_audio(target, w["start_sample"], self.window_samples)
        vf = _slice_video(video, w["start_frame"], self.window_frames, self.input_size)
        # Joint normalization: common scale from mixed's peak (preserves mixed/target amplitude ratio).
        max_val = mw.abs().max()
        if max_val > 0:
            scale = 0.8 / max_val
            mw = mw * scale
            tw = tw * scale
        return {
            "video_frames": vf,            # [T, H, W]
            "mixed_audio": mw,             # [1, T_audio]
            "target_audio": tw,            # [1, T_audio]
            "scene_id": sid,
            "window_start": w["start_sample"],
        }

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info is not None else 0
        nw = info.num_workers if info is not None else 1

        epoch = self._base_epoch + getattr(self, "_local_epoch", 0)
        self._local_epoch = getattr(self, "_local_epoch", 0) + 1

        scenes = list(self.scene_ids)
        if self.shuffle:
            random.Random(self.seed * 100003 + epoch).shuffle(scenes)
        scenes = scenes[wid::nw]                      # disjoint shard -> each scene read once total
        draw_rng = random.Random(self.seed * 7919 + epoch * 31 + wid)

        scene_iter = iter(scenes)
        cached: Dict[str, tuple] = {}                 # sid -> (mixed, target, video) raw arrays
        sizes: Dict[str, int] = {}                    # sid -> resident bytes (to subtract on evict)
        pending: Dict[str, int] = {}                  # sid -> windows not yet emitted
        pool: List[tuple] = []                        # (sid, window_dict)
        cached_bytes = [0]                            # boxed so the nested fill() can mutate it

        def fill():
            # Pull scenes until the count cap OR the byte budget is reached; always keep >=1 resident.
            while len(cached) < self.scene_buffer and (cached_bytes[0] < self._pool_budget_bytes
                                                       or not cached):
                sid = next(scene_iter)                # StopIteration -> caller stops filling
                arr = self._read_scene_arrays(sid)
                cached[sid] = arr
                nb = self._arrays_nbytes(arr)
                sizes[sid] = nb
                cached_bytes[0] += nb
                ws = self.scene_windows[sid]
                pending[sid] = len(ws)
                for w in ws:
                    pool.append((sid, w))

        try:
            fill()
        except StopIteration:
            pass

        while pool:
            if self.shuffle:
                i = draw_rng.randrange(len(pool))
                sid, w = pool[i]
                pool[i] = pool[-1]
                pool.pop()
            else:
                sid, w = pool.pop(0)
            item = self._decode(sid, cached[sid], w)
            pending[sid] -= 1
            if pending[sid] == 0:                     # scene fully emitted -> free it, pull more
                del cached[sid]
                del pending[sid]
                cached_bytes[0] -= sizes.pop(sid)
                try:
                    fill()
                except StopIteration:
                    pass
            yield item
