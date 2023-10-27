"""
LOAD AND CREATE THE DATASET;
TRAIN AND VALIDATE THE NETWORK
"""
import torch
from torch import nn
import numpy as np
import pandas as pd
import os
from makeTorchDataset import torchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from denUNet3D import denUNet3D
from denUNet2D import denUNet2D
from torchsummary import summary

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING AND TARGET IMAGES MUST BE ORDERED IN THE RESPECTIVE FOLDERS (FOR PAIRING).
TARGET_PATH = 'C:\\Users\\Luisa\\Documents\\master_thesis\\Datasets\\2D_dataset\\PET_15s_EARL1\\002'
# PATH TO THE REFERENCE IMAGES

IM_PATH = 'C:\\Users\\Luisa\\Documents\\master_thesis\\Datasets\\2D_dataset\\PET_70s_EARL1\\002'
# PATH TO THE TRAINING IMAGES

DIM = 2
LOSS = "MSE"

IMG_SIZE = 64

BATCH_SIZE = 2

N_EPOCHS = 1500
LR = 0.0005
DECAY_RATE = 0.90   # Every N epochs the learning rate is multiplied by this factor.
DECAY_EVERY_N_EPOCHS = 50
# ----------------------------------------------------------------------------------------------------------------------
# TRAIN/TEST SPLIT
# T: NUMBER OF IMAGES THAT WILL BE USED TO TEST THE MODEL; FIRST T IMAGES IN THE DATASET FOLDER
# V: NUMBER OF IMAGES THAT WILL BE USED FOR INTERNAL VALIDATION OF THE MODEL; V IMAGES THAT FOLLOW THE TEST SET
# THE REMAINING IMAGES IN THE DATASET WILL BE USED TO TRAIN THE MODEL
T = 20
V = 10
# ----------------------------------------------------------------------------------------------------------------------

DEVICE = 'cuda'

LOSS_FUNCTION = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
}

TARGET_PATH = TARGET_PATH + '\\'
IM_PATH = IM_PATH + '\\'

# -------------------------------------------------------------------------------------------------- DATASET
train_lst = []
valid_lst = []
for i in range(len(os.listdir(IM_PATH))):
    if T <= i < (V+T):
        valid_lst.append([IM_PATH + sorted(os.listdir(IM_PATH))[i],
                          TARGET_PATH + sorted(os.listdir(TARGET_PATH))[i]])

    elif i >= (V+T):

        train_lst.append([IM_PATH + sorted(os.listdir(IM_PATH))[i],
                          TARGET_PATH + sorted(os.listdir(TARGET_PATH))[i]])

COLUMNS = ['images', 'targets']
train_df = pd.DataFrame(train_lst, columns=COLUMNS)
valid_df = pd.DataFrame(valid_lst, columns=COLUMNS)


trainingSet = torchDataset(train_df, PATCH_SIZE=IMG_SIZE)
validationSet = torchDataset(valid_df, PATCH_SIZE=IMG_SIZE)


tLoader = DataLoader(trainingSet, batch_size=BATCH_SIZE, shuffle=True)
vLoader = DataLoader(validationSet, batch_size=BATCH_SIZE)

# -------------------------------------------------------------------------------------------------- MODEL TRAINING
if DIM == 3:
    model = denUNet3D()
else:
    model = denUNet2D()

model.to(DEVICE)

if DIM == 3:
    summary(model, (1, IMG_SIZE, IMG_SIZE, IMG_SIZE))
else:
    summary(model, (1, IMG_SIZE, IMG_SIZE))


def initial_loss(data_loader, device):
    total_loss = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        loss = LOSS_FUNCTION[LOSS](images, targets)

        total_loss += loss.item()

    return total_loss / len(data_loader)


train_loss = initial_loss(tLoader, DEVICE)
valid_loss = initial_loss(vLoader, DEVICE)


def train_fn(data_loader, model, optimizer, device):
    model.train()  # Tells the model it's going to be trained, this prepares
    # layers that are present in training but not in validation,
    # for instance.
    total_loss = 0.0

    for images, targets in tqdm(data_loader):

        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = LOSS_FUNCTION[LOSS](out, targets)
        loss.backward()  # loss.backward() computes dloss/dx for every parameter x
        # which has requires_grad=True. These are accumulated into x.grad.
        optimizer.step()  # optimizer.step() updates the value of x using the
        # gradient x.grad

        total_loss += loss.item()  # loss.item() extracts the loss's value as a
        # python float. At each epoch loss for each image in the batch is
        # calculated and, therefore, at each iteration the losses are summed
        # and then divided by the number of images in the batch, in order to get
        # the average batch loss of that epoch.

    return total_loss / len(data_loader)


def valid_fn(data_loader, model, device):
    model.eval()  # Tells the model it's going to be validated.
    total_loss = 0.0

    with torch.no_grad():  # Context-manager that disabled gradient calculation.
        # Disabling gradient calculation is useful for inference,
        # when you are sure that you will not call Tensor.backward().
        # It will reduce memory consumption for computations that
        # would otherwise have requires_grad=True.
        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = targets.to(device)

            out = model(images)
            loss = LOSS_FUNCTION[LOSS](out, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)


if not os.path.isdir('Models/'):
    os.mkdir('Models/')

EPOCH = 0
MODEL_NAME = f"denUNET{DIM}D_{LOSS}_{str(EPOCH).zfill(3)}_PET.pt"
LAST_MODEL = MODEL_NAME

best_valid_loss = np.Inf
for i in range(1, N_EPOCHS+1):

    if i % DECAY_EVERY_N_EPOCHS == 0:
        LR = LR * DECAY_RATE

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f'[{i}]')

    train_loss = train_fn(tLoader, model, optimizer, DEVICE)
    valid_loss = valid_fn(vLoader, model, DEVICE)

    if LAST_MODEL != MODEL_NAME:
        try:
            os.remove('Models/' + LAST_MODEL)
        except OSError:
            pass
    LAST_MODEL = MODEL_NAME.replace(str(EPOCH).zfill(3), str(i).zfill(3))
    torch.save(model.state_dict(), 'Models/' + LAST_MODEL)

    if valid_loss < best_valid_loss:
        try:
            os.remove('Models/'+MODEL_NAME)
        except OSError:
            pass
        MODEL_NAME = MODEL_NAME.replace(str(EPOCH).zfill(3), str(i).zfill(3))
        torch.save(model.state_dict(), 'Models/'+MODEL_NAME)
        EPOCH = i
        print("MODEL SAVED")
        best_valid_loss = valid_loss

    print(f"[Epoch {i}] Train loss: {train_loss} | Validation loss: {valid_loss}")
