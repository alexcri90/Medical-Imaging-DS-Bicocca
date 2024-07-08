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
        self.transforms = get_transforms(phase, size, mean, std)
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

        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

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


def get_transforms(phase, size, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                #                 HorizontalFlip(),
                ShiftScaleRotate(
                    shift_limit=0,  # no resizing
                    scale_limit=0.1,
                    rotate_limit=10,  # rotate
                    p=0.5,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
                #                 GaussNoise(),
            ]
        )
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensorV2(),
        ]
    )

    list_trfms = Compose(list_transforms)

    # list_trfms = transforms.Compose(
    #     [
    #         transforms.Resize(size),
    #         transforms.Normalize(mean=mean, std=std),
    #         transforms.ToTensor(),
    #     ]
    # )
    return list_trfms


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
    print("Creating dataset...")
    df_all = pd.read_csv(df_path)
    # df_all = df_all[0:100]
    df = df_all.drop_duplicates("ImageId")
    df_with_mask = df[df[" EncodedPixels"] != " -1"]
    df_with_mask["has_mask"] = 1
    df_without_mask = df[df[" EncodedPixels"] == " -1"]
    df_without_mask["has_mask"] = 0
    df_without_mask_sampled = df_without_mask.sample(
        len(df_with_mask), random_state=69, replace=True
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


def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha * self.focal(input, target) - torch.log(
            dice_loss(input, target)
        )
        return loss.mean()


def predict(X, threshold):
    X_p = np.copy(X)
    preds = (X_p > threshold).astype("uint8")
    return preds


# def metric(probability, truth, threshold=0.5, reduction="none"):
#     """Calculates dice of positive and negative images seperately"""
#     """probability and truth must be torch tensors"""
#     batch_size = len(truth)
#     with torch.no_grad():
#         probability = probability.view(batch_size, -1)
#         truth = truth.view(batch_size, -1)
#         assert probability.shape == truth.shape

#         p = (probability > threshold).float()
#         t = (truth > 0.5).float()

#         t_sum = t.sum(-1)
#         p_sum = p.sum(-1)
#         neg_index = torch.nonzero(t_sum == 0)
#         pos_index = torch.nonzero(t_sum >= 1)

#         dice_neg = (p_sum == 0).float()
#         dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

#         dice_neg = dice_neg[neg_index]
#         dice_pos = dice_pos[pos_index]
#         dice = torch.cat([dice_pos, dice_neg])

#         #         dice_neg = np.nan_to_num(dice_neg.mean().item(), 0)
#         #         dice_pos = np.nan_to_num(dice_pos.mean().item(), 0)
#         #         dice = dice.mean().item()

#         num_neg = len(neg_index)
#         num_pos = len(pos_index)

#     return dice, dice_neg, dice_pos, num_neg, num_pos


def metric(probability, truth, threshold=0.5, reduction="none"):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0).view(-1)
        pos_index = torch.nonzero(t_sum >= 1).view(-1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    return dice.tolist(), dice_neg.tolist(), dice_pos.tolist(), num_neg, num_pos


class Meter:
    """A meter to keep track of iou and dice scores throughout an epoch"""

    def __init__(self, phase, epoch):
        self.base_threshold = 0.5  # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    # def update(self, targets, outputs):
    #     probs = torch.sigmoid(outputs)
    #     dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
    #     self.base_dice_scores.extend(dice)
    #     self.dice_pos_scores.extend(dice_pos)
    #     self.dice_neg_scores.extend(dice_neg)
    #     preds = predict(probs, self.base_threshold)
    #     iou = compute_iou_batch(preds, targets, classes=[1])
    #     self.iou_scores.append(iou)
    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.dice_pos_scores.extend(dice_pos)
        self.dice_neg_scores.extend(dice_neg)
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        print(self.base_dice_scores)
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    """logging the metrics at the end of an epoch"""
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print(
        "Loss: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f | IoU: %0.4f"
        % (epoch_loss, dice, dice_neg, dice_pos, iou)
    )
    return dice, iou


def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    """computes iou for one ground truth mask and predicted mask"""
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


def compute_iou_batch(outputs, labels, classes=None):
    """computes mean iou for a batch of ground truth masks and predicted masks"""
    ious = []
    preds = np.copy(outputs)  # copy is imp
    labels = np.array(labels)  # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou


from tqdm import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import time


class Trainer(object):
    """This class takes care of training and validation of our model"""

    def __init__(self, model):
        self.model = model
        self.fold = 1
        self.total_folds = 5
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size["train"]
        self.lr = 5e-4
        self.num_epochs = 5
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("mps")
        # self.device = torch.device("cuda:0")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = MixedLoss(10.0, 2.0)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, verbose=True
        )
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                fold=1,
                total_folds=5,
                data_folder="./dicom_files",
                df_path="train-rle-train.csv",
                phase=phase,
                size=512,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)  # .unsqueeze(1)  # (batch_size, 1, 512, 512)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches, desc=f"Epoch {epoch} [{phase}]")
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            # print(f"Batch {itr} - targets shape: {targets.shape}, outputs shape: {outputs.shape}")
            # targets = targets.unsqueeze(1)
            meter.update(targets, outputs)
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        # torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])), scores["train"], label=f"train {name}")
    plt.plot(range(len(scores["train"])), scores["val"], label=f"val {name}")
    plt.title(f"{name} plot")
    plt.xlabel("Epoch")
    plt.ylabel(f"{name}")
    plt.legend()
    plt.show()


import segmentation_models_pytorch as smp

if __name__ == "__main__":
    print("test")
    dataloader = provider(
        fold=0,
        total_folds=5,
        data_folder="./dicom_files",
        df_path="./train-rle-train.csv",
        phase="train",
        size=512,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        batch_size=16,
        num_workers=4,
    )
    print("dataloader created")
    batch = next(iter(dataloader))  # get a batch from the dataloader
    images, masks = batch
    # plot some random images in the `batch`
    idx = random.choice(rangxe(16))
    # plt.imshow(images[idx][0], cmap="bone")
    # plt.imshow(masks[idx][0], alpha=0.2, cmap="Reds")
    # plt.show()
    if len(np.unique(masks[idx][0])) == 1:  # only zeros

        print("Chosen image has no ground truth mask, rerun the cell")
    print("success")
    model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None)
    model_trainer = Trainer(model)
    model_trainer.start()
    model_trainer.save_model("models/segmentation_restnet34_model.pth")

    # PLOT TRAINING
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores  # overall dice
    iou_scores = model_trainer.iou_scores

    plot(losses, "BCE loss")
    plot(dice_scores, "Dice score")
    plot(iou_scores, "IoU score")
