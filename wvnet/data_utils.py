import os
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import v2


class ImagePreprocessor(nn.Module):
    """Preprocessing for images.

    Assumes that images are stored as single-channel greyscale images and roughly 450x450 pixels or smaller.
    Smaller images are padded to standard size 450x450, then center-cropped to 224x224.
    """

    def __init__(
        self,
        image_size: tuple[int] = (450, 450),
        out_size: tuple[int] = (224, 224),
        mu: float = 103.4862417561056,
        sig: float = 56.745013435601074,
    ) -> None:
        """
        Args:
            image_size (tuple[int], optional): Expected input image size. Defaults to (450, 450).
            out_size (tuple[int], optional): Output size. Defaults to (224, 224).
            mu (float, optional): Mean for normalization. Defaults to 103.4862417561056.
            sig (float, optional): Standard deviation for normalization. Defaults to 56.745013435601074.
        """
        super().__init__()
        self.image_size = image_size
        self.out_size = out_size
        self.mu = mu / 255.0
        self.sig = sig / 255.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess an image.

        Args:
            x (torch.Tensor): Unprocessed image.

        Returns:
            torch.Tensor: Processed image.
        """
        load_transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.uint8, scale=True)]
        )
        im = load_transforms(x)

        h_diff = max(0, self.image_size[0] - im.size(dim=-2))
        w_diff = max(0, self.image_size[1] - im.size(dim=-1))
        im = v2.Pad(
            padding=(0, w_diff, 0, h_diff), padding_mode='constant', fill=0.0
        )(im)
        im = v2.CenterCrop(self.image_size)(im)
        im = v2.ToDtype(torch.float32, scale=True)(im)

        # More involved than v2.Resize because original code used kornia with align_corners option set
        im = torch.nn.functional.interpolate(torch.unsqueeze(im, 0), self.out_size, antialias=False, mode='bilinear', align_corners=True)
        im = torch.squeeze(im, 0)

        im = im.repeat(3, 1, 1)
        im = v2.Normalize(mean=[self.mu] * 3, std=[self.sig] * 3)(im)

        return im


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for dataset
    """

    def __init__(self, data_src: str, image_parent:str | None = None, transform: torch.nn.Module | None = None):
        """
        Args:
            data_src (any): Source of the data, varies for different generators
            transform (Optional[torch.nn.Module]): Transformation to be applied to input images
        """
        self.image_parent = image_parent if image_parent else os.getcwd()
        self.transform = transform

        self.df = self._init_data(data_src)

    def __len__(self) -> int:
        """
        Returns:
            int: Length of the dataset
        """
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[np.array, int]:
        """
        Args:
            index (int): Index of the item to be retrieved
        Returns:
            tuple[np.array, int]: Tuple containing the image and its index in the dataframe
        """
        row = self.df.iloc[index]
        img_file = row['filename']
        img_file = os.path.join(self.image_parent, img_file)
        img = Image.open(img_file).convert('L')

        if self.transform:
            img = self.transform(img)

        return img

    def _init_data(self, data_src: any) -> pd.DataFrame:
        """
        Initializes the data

        Args:
            data_src (any): Source of the data, varies for different generators
        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        raise NotImplementedError


class CSVDataset(BaseDataset):
    """
    Dataset class for loading data from a CSV file
    """

    def _init_data(self, data_src: str) -> pd.DataFrame:
        """
        Initializes the data

        Args:
            data_src (str): Path to the CSV file
        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        df = pd.read_csv(data_src)

        return df


class JSONDataset(BaseDataset):
    """
    Dataset class for loading data from a JSON file
    """

    def _init_data(self, data_src: str) -> pd.DataFrame:
        """
        Initializes the data

        Args:
            data_src (str): Path to the JSON file
        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        with open(data_src, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        return df


class SimpleDataset(BaseDataset):
    """
    Dataset class for loading data from a list of files
    """

    def _init_data(self, data_src: list[str]) -> pd.DataFrame:
        """
        Initializes the data

        Args:
            data_src (list[str]): List of file paths
        Returns:
            pd.DataFrame: DataFrame containing the data
        """
        df = pd.DataFrame(data_src, columns=['filename'])

        return df


def init_dataset(source: str, image_parent: str, transform: torch.nn.Module):
    if isinstance(source, str):
        extension = source.split('.')[-1]
        if extension == 'csv':
            dataset = CSVDataset(source, image_parent, transform)
        elif extension == 'json':
            dataset = JSONDataset(source, image_parent, transform)
        else:
            raise ValueError(f'Unsupported file format: {extension}')

    elif isinstance(source, list):
        dataset = SimpleDataset(source, image_parent, transform)

    else:
        raise ValueError(f'Unsupported data source: {source}')

    return dataset
