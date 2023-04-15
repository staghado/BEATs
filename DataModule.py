import librosa
import torch
import torchaudio
import pandas as pd
import os

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#from torchaudio.transforms import Resample
from sklearn.preprocessing import LabelEncoder

from pytorch_lightning import LightningDataModule


class AudioDataset(Dataset):
    def __init__(self, root_dir, data_frame, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_frame = data_frame

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.data_frame["category"])

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]["filename"])
        label = self.data_frame.iloc[idx]["category"]

        # Load audio data and perform any desired transformations
        sig, sr = librosa.load(audio_path, sr=16000, mono=True)
        sig_t = torch.tensor(sig)
        padding_mask = torch.zeros(1, sig_t.shape[0]).bool().squeeze(0)
        if self.transform:
            sig_t = self.transform(sig_t)

        # Encode label as integer
        label = self.label_encoder.transform([label])[0]

        return sig_t, padding_mask, label


class ECS50DataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/home/said/projects/BirdCLEF/Fine-tune-BEATs/ESC-50-master/audio/",
        csv_file: str = "/home/said/projects/BirdCLEF/Fine-tune-BEATs/ESC-50-master/meta/esc50.csv",
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

    def setup(self, stage=None):
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

        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.val_set, transform=self.transform
        )

        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False)

# BIRDCLEF 2023 DATASET
class BirdDataModule(LightningDataModule):
    def __init__(
        self,
        root_dir: str = "/home/said/projects/BirdCLEF/Fine-tune-BEATs/BIRDCLEF-DATASET/",
        data_frame = None,
        batch_size: int = 8,
        split_ratio=0.8,
        transform=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.transform = transform

        self.setup()

    def prepare_data(self):
        pass

    def setup(self):
        data_frame = self.data_frame
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

        return DataLoader(train_df, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_df = AudioDataset(
            root_dir=self.root_dir, data_frame=self.val_set, transform=self.transform
        )

        return DataLoader(val_df, batch_size=self.batch_size, shuffle=False)


# class BirdCLEFDataset(Dataset):
#     def __init__(self, df, target_sample_rate, max_time, image_transforms=None):
#         self.file_paths = df['file_path'].values
#         self.labels = df['primary_label'].values
#         self.target_sample_rate = target_sample_rate
#         num_samples = target_sample_rate * max_time
#         self.num_samples = num_samples
#         self.image_transforms = image_transforms
        
#     def __len__(self):
#         return len(self.file_paths)
    
#     def __getitem__(self, index):
#         filepath = self.file_paths[index]
#         audio, sample_rate = torchaudio.load(filepath)
#         audio = self.to_mono(audio)
        
#         if sample_rate != self.target_sample_rate:
#             resample = Resample(sample_rate, self.target_sample_rate)
#             audio = resample(audio)
        
#         if audio.shape[0] > self.num_samples:
#             audio = self.crop_audio(audio)
            
#         if audio.shape[0] < self.num_samples:
#             audio = self.pad_audio(audio)
            
#         mel_spectogram = MelSpectrogram(sample_rate=self.target_sample_rate, 
#                                         n_mels=CONFIG['n_mels'], 
#                                         n_fft=CONFIG['n_fft'])
#         mel = mel_spectogram(audio)
        
#         label_onehot = torch.zeros(CONFIG['num_classes'])
#         label_onehot[self.labels[index]] = 1
#         label = torch.tensor(self.labels[index])
        
#         # Convert to Image
#         image = torch.stack([mel, mel, mel])
        
#         # Normalize Image
#         max_val = torch.abs(image).max()
#         image = image / max_val
        
#         return {
#             "image": image, 
#             "label": label,
#             "label_onehot": label_onehot
#         }
            
#     def pad_audio(self, audio):
#         pad_length = self.num_samples - audio.shape[0]
#         last_dim_padding = (0, pad_length)
#         audio = F.pad(audio, last_dim_padding)
#         return audio
        
#     def crop_audio(self, audio):
#         return audio[:self.num_samples]
        
#     def to_mono(self, audio):
#         return torch.mean(audio, axis=0)