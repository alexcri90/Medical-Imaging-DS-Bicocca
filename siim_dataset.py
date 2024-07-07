import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm_notebook as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
# from albumentations import (
#     HorizontalFlip,
#     ShiftScaleRotate,
#     Normalize,
#     Resize,
#     Compose,
#     GaussNoise,
# )
#from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")


def run_length_decode(rle, height=1024, width=1024, fill_value=1):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(" ")])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start:end] = fill_value
        start = end
    component = component.reshape(width, height).T
    return component


def run_length_encode(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = " ".join([str(r) for r in rle])
    return rle


class SIIMDataset(Dataset):
    def __init__(self, df, fnames, data_folder, size, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = None #get_transforms(phase, size, mean, std) 
        self.gb = self.df.groupby("ImageId")
        self.fnames = fnames

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df[" EncodedPixels"].tolist()
        image_path = os.path.join(self.root, image_id + ".dcm")
        dcm = pydicom.dcmread(image_path)
        image = dcm.pixel_array
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB
        )  # Assuming grayscale images, convert to RGB
        image = cv2.resize(
            image, (self.size, self.size)
        )  # Resize to the same size as the mask
        mask = np.zeros([1024, 1024])
        if annotations[0] != " -1":
            for rle in annotations:
                mask += run_length_decode(rle)
        mask = (mask >= 1).astype("float32")  # for overlap cases
        # mask = cv2.resize(
        #     mask, (self.size, self.size)
        # )  # Resize mask to match image size
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # augmented = self.transforms(image=image, mask=mask)
        # image = augmented["image"]
        # mask = augmented["mask"]
        # Ensure mask has the correct dimensions
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)  # Add channel dimension if missing

        return image, mask
        # image_id = self.fnames[idx]
        # df = self.gb.get_group(image_id)
        # annotations = df[" EncodedPixels"].tolist()
        # image_path = os.path.join(self.root, image_id + ".dcm")
        # image = cv2.imread(image_path)
        # image = np.array(image)
        # mask = np.zeros([1024, 1024])
        # if annotations[0] != " -1":
        #     for rle in annotations:
        #         mask += run_length_decode(rle)
        # mask = (mask >= 1).astype("float32")  # for overlap cases
        # augmented = self.transforms(image=image, mask=mask)
        # image = augmented["image"]
        # mask = augmented["mask"]
        # return image, mask

    def __len__(self):
        return len(self.fnames)


# def get_transforms(phase, size, mean, std):
#     list_transforms = []
#     if phase == "train":
#         list_transforms.extend(
#             [
#                 #                 HorizontalFlip(),
#                 ShiftScaleRotate(
#                     shift_limit=0,  # no resizing
#                     scale_limit=0.1,
#                     rotate_limit=10,  # rotate
#                     p=0.5,
#                     border_mode=cv2.BORDER_CONSTANT,
#                 ),
#                 #                 GaussNoise(),
#             ]
#         )
#     list_transforms.extend(
#         [
#             Resize(size, size),
#             Normalize(mean=mean, std=std, p=1),
#             ToTensorV2(),
#         ]
#     )

#     list_trfms = Compose(list_transforms)
#     return list_trfms


def provider(
    fold,
    total_folds,
    data_folder,
    df_path,
    phase,
    size,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=4,
):
    df_all = pd.read_csv(df_path)
    # df_all = df_all[0:1000]
    df = df_all.drop_duplicates("ImageId")
    df_with_mask = df[df[" EncodedPixels"] != " -1"]
    df_with_mask["has_mask"] = 1
    df_without_mask = df[df[" EncodedPixels"] == " -1"]
    df_without_mask["has_mask"] = 0
    df_without_mask_sampled = df_without_mask.sample(
        len(df_with_mask), random_state=69
    )  # random state is imp
    df = pd.concat([df_with_mask, df_without_mask_sampled])

    # NOTE: equal number of positive and negative cases are chosen.

    kfold = StratifiedKFold(total_folds, shuffle=True, random_state=69)
    train_idx, val_idx = list(kfold.split(df["ImageId"], df["has_mask"]))[fold]
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    df = train_df if phase == "train" else val_df
    # NOTE: total_folds=5 -> train/val : 80%/20%

    fnames = df["ImageId"].values

    image_dataset = SIIMDataset(df_all, fnames, data_folder, size, mean, std, phase)

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader
