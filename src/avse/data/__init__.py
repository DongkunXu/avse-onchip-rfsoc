"""Data pipeline (migrated from the reference project, Windows-friendly, window-based).

Reads the LRS3 layout at ``D:\\DataSet\\LRS3``:
    <split>/scenes/<scene_id>_{mixed,target,interferer}.wav   (16 kHz)
    <split>/96/<scene_id>.npy                                  ([T,96,96] uint8)
    scenes.{train,dev}.json                                    (metadata)

Set ``config.data.root_dir = "D:/DataSet/LRS3"``.
"""

from .dataset import AVSEDataset
from .data_module import AVSEDataModule
from .stream_dataset import AVSESceneStreamDataset

__all__ = ["AVSEDataset", "AVSEDataModule", "AVSESceneStreamDataset"]
