import os
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt

from albumentations import (
    HorizontalFlip,
    ShiftScaleRotate,
    Normalize,
    Resize,
    Compose,
    GaussNoise,
)
from albumentations.pytorch import ToTensorV2

from torchvision import transforms


warnings.filterwarnings("ignore")
import segmentation_models_pytorch as smp

from training_segm import run_length_encode, run_length_decode


class TestDataset(Dataset):
    def __init__(self, root, df, size, mean, std, tta=4):
        self.root = root
        self.df = df
        self.size = size
        self.fnames = list(df["ImageId"])
        self.num_samples = len(self.fnames)
        self.transform = Compose(
            [
                Normalize(mean=mean, std=std, p=1),
                Resize(size, size),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        path = os.path.join(self.root, fname + ".dcm")
        print(path)
        # image = cv2.imread(path)
        dcm = pydicom.dcmread(path)
        image = dcm.pixel_array
        image = cv2.cvtColor(
            image, cv2.COLOR_GRAY2RGB
        )  # Assuming grayscale images, convert to RGB
        image = cv2.resize(
            image, (self.size, self.size)
        )  # Resize to the same size as the mask
        images = self.transform(image=image)["image"]
        return images

    def __len__(self):
        return self.num_samples


def post_process(probability, threshold, min_size):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    num = 0
    for c in range(1, num_component):
        p = component == c
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


if __name__ == "__main__":

    size = 512
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    num_workers = 0
    batch_size = 16
    best_threshold = 0.5
    min_size = 3500
    # device = torch.device("cuda:0")
    device = torch.device("mps")
    # df = pd.read_csv("./results.csv")
    df = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
    testset = DataLoader(
        TestDataset("./train-rle-test-10.csv", df, size, mean, std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    # model = model_trainer.net  # get the model from model_trainer object
    model.eval()
    state = torch.load(
        "./models/segmentation_restnet34_model.pth",
        map_location=lambda storage, loc: storage,
    )
    print(state)
    model.load_state_dict(state["state_dict"])
    encoded_pixels = []
    print("Model loaded")

    for i, batch in enumerate(tqdm(testset)):
        preds = torch.sigmoid(model(batch.to(device)))
        preds = (
            preds.detach().cpu().numpy()[:, 0, :, :]
        )  # (batch_size, 1, size, size) -> (batch_size, size, size)
        for probability in preds:
            if probability.shape != (1024, 1024):
                probability = cv2.resize(
                    probability, dsize=(1024, 1024), interpolation=cv2.INTER_LINEAR
                )
            predict, num_predict = post_process(probability, best_threshold, min_size)
            if num_predict == 0:
                encoded_pixels.append("-1")
            else:
                r = run_length_encode(predict)
                encoded_pixels.append(r)
    df["EncodedPixels"] = encoded_pixels
    df.to_csv("submission.csv", columns=["ImageId", "EncodedPixels"], index=False)
    df.head()
