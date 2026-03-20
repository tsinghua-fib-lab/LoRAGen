import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import random_split
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric


from zoodatasets.lora_multidatasets import LoRAMultiDataset


import pytorch_lightning as pl
from torch.utils.data import DataLoader




class LoRAZooDataModule_Multi(pl.LightningDataModule):
    def __init__(self, data_dir='vae_input/weights', batch_size=64, num_workers=4, scale=1.0, val_ratio=0.1, max_blocks=288, seed=42, record_split_path=None, split_file=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scale = scale
        self.val_ratio = val_ratio
        self.max_blocks = max_blocks
        self.seed = seed
        self.record_split_path = record_split_path
        self.split_file = split_file

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.trainset = LoRAMultiDataset(
                root_dir=self.data_dir,
                split='train',
                val_ratio=self.val_ratio,
                max_blocks=self.max_blocks,
                seed=self.seed,
                record_split_path=self.record_split_path,
                split_file=self.split_file
            )
            self.valset = LoRAMultiDataset(
                root_dir=self.data_dir,
                split='val',
                val_ratio=self.val_ratio,
                max_blocks=self.max_blocks,
                seed=self.seed,
                record_split_path=self.record_split_path,
                split_file=self.split_file
            )

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        pass

