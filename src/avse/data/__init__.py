"""Data pipeline (Windows-friendly, window-based).

``AVSEDataset`` is the map-style window dataset (used by ``tools/verify_data.py`` and val);
``AVSESceneStreamDataset`` is the scene-streaming IterableDataset used for training (each scene's
files opened once per epoch — see docs/DECISIONS.md D-14).

Reads the LRS3 layout at ``D:\\DataSet\\LRS3``:
    <split>/scenes/<scene_id>_{mixed,target,interferer}.wav   (16 kHz)
    <split>/96/<scene_id>.npy                                  ([T,96,96] uint8)
    scenes.{train,dev}.json                                    (metadata)

Set ``config.data.root_dir = "D:/DataSet/LRS3"``.
"""

from .dataset import AVSEDataset
from .stream_dataset import AVSESceneStreamDataset

__all__ = ["AVSEDataset", "AVSESceneStreamDataset"]
