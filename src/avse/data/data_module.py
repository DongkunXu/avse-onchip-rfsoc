import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from .dataset import AVSEDataset


class AVSEDataModule(pl.LightningDataModule):
    """AVSE数据模块 - 使用流式窗口数据集"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == "fit" or stage is None:
            self.train_dataset = AVSEDataset(
                root_dir=self.config.data.root_dir,
                split="train",
                config=self.config
            )

            self.val_dataset = AVSEDataset(
                root_dir=self.config.data.root_dir,
                split="dev",
                config=self.config
            )

    def train_dataloader(self):
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=True if self.config.data.num_workers > 0 else False
            # 不指定collate_fn，使用PyTorch默认的default_collate
        )

    def val_dataloader(self):
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            persistent_workers=True if self.config.data.num_workers > 0 else False
            # 不指定collate_fn，使用PyTorch默认的default_collate
        )