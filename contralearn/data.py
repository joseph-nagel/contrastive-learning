'''Datamodules.'''

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from lightning import LightningDataModule


class MNISTDataModule(LightningDataModule):
    '''
    DataModule for the MNIST dataset.

    Parameters
    ----------
    data_dir : str
        Directory for storing the data.
    mean : float or None
        Mean for data normalization.
    std : float or None
        Standard deviation for normalization.
    batch_size : int
        Batch size of the data loader.
    num_workers : int
        Number of workers for the loader.

    '''

    def __init__(
        self,
        data_dir: str,
        mean: float | None = None,
        std: float | None = None,
        batch_size: int = 32,
        num_workers: int = 0
    ) -> None:

        super().__init__()

        # set data location
        self.data_dir = data_dir

        # set loader parameters
        self.batch_size = batch_size
        self.num_workers = num_workers

        # create transforms
        train_transforms = [
            transforms.RandomRotation(5),  # TODO: refine data augmentation
            transforms.ToTensor()
        ]

        test_transforms = [transforms.ToTensor()]

        if (mean is not None) and (std is not None):
            normalize_fn = transforms.Normalize(mean=mean, std=std)

            train_transforms.append(normalize_fn)
            test_transforms.append(normalize_fn)

        self.train_transform = transforms.Compose(train_transforms)
        self.test_transform = transforms.Compose(test_transforms)

    def prepare_data(self) -> None:
        '''Download data.'''

        train_set = datasets.MNIST(
            self.data_dir,
            train=True,
            download=True
        )

        test_set = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True
        )

    def setup(self, stage: str) -> None:
        '''Set up train/test/val. datasets.'''

        # create train/val. datasets
        if stage in ('fit', 'validate'):
            train_set = datasets.MNIST(
                self.data_dir,
                train=True,
                transform=self.train_transform
            )

            self.train_set, self.val_set = random_split(
                train_set,
                [50000, 10000],
                generator=torch.Generator().manual_seed(42)
            )

        # create test dataset
        elif stage == 'test':
            self.test_set = datasets.MNIST(
                self.data_dir,
                train=False,
                transform=self.test_transform
            )

    def train_dataloader(self) -> DataLoader:
        '''Create train dataloader.'''
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        '''Create val. dataloader.'''
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        '''Create test dataloader.'''
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.num_workers > 0
        )

