import os

import librosa
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    def __init__(self, root_dir, data_frame, num_samples, transform=None):
        super(AudioDataset).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = data_frame
        if isinstance(num_samples, tuple):
            num_samples = num_samples[0]
        self.num_samples = num_samples

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = os.path.join(
            self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        audio, _ = librosa.load(audio_path, sr=32000, mono=True)
        audio = torch.tensor(audio)

        if self.transform:
            audio = self.transform(audio)

        if audio.shape[0] > self.num_samples:
            audio = self.crop_audio(audio)

        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)

        # Create padding mask
        padding_mask = torch.zeros(1, audio.shape[0]).bool().squeeze(0)

        return audio, padding_mask, label

    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding)
        return audio

    def crop_audio(self, audio):
        return audio[:self.num_samples]

# BIRDCLEF 2023 DATASET


class BirdDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/kaggle/input/birdclef-split-train-audio/",
        csv_file: str = "/kaggle/working/beats-mine/train_metadata.csv",
        batch_size: int = 8,
        split_ratio=0.8,
        num_samples=32000 * 5,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.num_samples = num_samples,
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        data_frame = pd.read_csv(self.csv_file)
        #.sample(frac=0.01).reset_index(drop=True)
        # data_frame = data_frame.sample(frac=1).reset_index(
        #     drop=True
        # )  # shuffle the data frame
        split_index = int(len(data_frame) * self.split_ratio)
        self.train_set = data_frame.iloc[:split_index, :]
        self.val_set = data_frame.iloc[split_index:, :]

    def train_dataloader(self):
        train_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.train_set, num_samples=self.num_samples, transform=self.transform
        )

        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.val_set, num_samples=self.num_samples, transform=self.transform
        )

        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False)


class ECS50DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/ESC-50-master/audio/",
        csv_file: str = "/ESC-50-master/meta/esc50.csv",
        batch_size: int = 8,
        split_ratio=0.8,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self):
        data_frame = pd.read_csv(self.csv_file)
        data_frame = data_frame.sample(frac=1).reset_index(
            drop=True
        )  # shuffle the data frame
        split_index = int(len(data_frame) * self.split_ratio)
        self.train_set = data_frame.iloc[:split_index, :]
        self.val_set = data_frame.iloc[split_index:, :]

    def train_dataloader(self):
        train_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.train_set, transform=self.transform
        )

        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        val_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.val_set, transform=self.transform
        )

        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False, num_workers=2)
