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
from fastPET_UNet3D import UNet3D
from fastPET_UNet2D import UNet2D
import torch.multiprocessing
#from torchsummary import summary

# ----------------------------------------------------------------------------------------------------------------------
# TRAINING AND TARGET IMAGES MUST BE ORDERED IN THE RESPECTIVE FOLDERS (FOR PAIRING).
TARGET_PATH = '/Users/alex/Documents/GitHub/UNET4denoising/Dataset/Target' # Penser à changer le chemin
# PATH TO THE REFERENCE IMAGES

IM_PATH = '/Users/alex/Documents/GitHub/UNET4denoising/Dataset/Training' # Penser à changer le chemin
# PATH TO THE TRAINING IMAGES

DIM = 2    # EITHER 2 (if the images are 2D --> 2D U-Net) or 3 (if the images are 3D --> 3D U-Net)
LOSS = "MSE"

IMG_SIZE = 480 # Ancienne exécution: 128

BATCH_SIZE = 16

N_EPOCHS = 150
INITIAL_LR = 0.001  # Renommé pour plus de clarté
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

DEVICE = 'mps'

LOSS_FUNCTION = {
    "MSE": nn.MSELoss(),
    "L1": nn.L1Loss(),
}

def initial_loss(data_loader, device):
    total_loss = 0.0

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)

        loss = LOSS_FUNCTION[LOSS](images, targets)

        total_loss += loss.item()

    return total_loss / len(data_loader)


def train_fn(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(data_loader):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        out = model(images)
        loss = LOSS_FUNCTION[LOSS](out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def valid_fn(data_loader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = targets.to(device)

            out = model(images)
            loss = LOSS_FUNCTION[LOSS](out, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)

def save_checkpoint(epoch, model, optimizer, loss, current_lr, checkpoint_dir='checkpoints'):
    """Sauvegarde l'état complet de l'entraînement"""
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{str(epoch).zfill(3)}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'learning_rate': current_lr
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint sauvegardé: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Charge un checkpoint précédemment sauvegardé"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return (
        checkpoint['epoch'],
        model,
        optimizer,
        checkpoint['loss'],
        checkpoint['learning_rate']
    )

def main(resume_from_checkpoint=None):
    # Utilisation de LR comme variable locale
    current_lr = INITIAL_LR

    # -------------------------------------------------------------------------------------------------- DATASET
    train_lst = []
    valid_lst = []

    # Récupérer et trier les fichiers
    target_files = sorted([f for f in os.listdir(TARGET_PATH) if f.endswith('.png')])
    noisy_files = sorted([f for f in os.listdir(IM_PATH) if f.endswith('.png')])

    # Pour chaque image originale
    for i, target_file in enumerate(target_files):
        target_id = target_file.split('.')[0]  # ex: "0001"
        target_path = os.path.join(TARGET_PATH, target_file)
        
        # Trouver toutes les versions bruitées correspondantes
        matching_noisy = [f for f in noisy_files if f.startswith(target_id + "_")]
        
        if not matching_noisy:
            print(f"Attention: Aucune image bruitée trouvée pour {target_file}")
            continue
            
        # Pour chaque version bruitée de cette image
        for noisy_file in matching_noisy:
            noisy_path = os.path.join(IM_PATH, noisy_file)
            pair = [noisy_path, target_path]
            
            # Répartition dans train ou validation
            if T <= i < (V+T):
                valid_lst.append(pair)
            elif i >= (V+T):
                train_lst.append(pair)

    print(f"Nombre de paires d'entraînement: {len(train_lst)}")
    print(f"Nombre de paires de validation: {len(valid_lst)}")

    COLUMNS = ['images', 'targets']
    train_df = pd.DataFrame(train_lst, columns=COLUMNS)
    valid_df = pd.DataFrame(valid_lst, columns=COLUMNS)

    # Utilisation de la classe torchDataset existante
    trainingSet = torchDataset(train_df, PATCH_SIZE=IMG_SIZE)
    validationSet = torchDataset(valid_df, PATCH_SIZE=IMG_SIZE)

    # Configuration des DataLoaders avec multi-processing
    tLoader = DataLoader(trainingSet, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=torch.multiprocessing.cpu_count(), pin_memory=True)
    vLoader = DataLoader(validationSet, batch_size=BATCH_SIZE, 
                        num_workers=torch.multiprocessing.cpu_count(), pin_memory=True)

    # -------------------------------------------------------------------------------------------------- MODEL TRAINING
    if DIM == 3:
        model = UNet3D()
    else:
        model = UNet2D()

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=current_lr)

    # Initialisation des variables d'entraînement
    start_epoch = 0
    best_valid_loss = np.Inf
    
    # Chargement du checkpoint si demandé
    if resume_from_checkpoint:
        print(f"Reprise de l'entraînement depuis {resume_from_checkpoint}")
        start_epoch, model, optimizer, best_valid_loss, current_lr = load_checkpoint(
            resume_from_checkpoint, model, optimizer
        )
        print(f"Reprise depuis l'époque {start_epoch} avec lr={current_lr}")
    else:
        # Calcul des pertes initiales si nouveau départ
        train_loss = initial_loss(tLoader, DEVICE)
        valid_loss = initial_loss(vLoader, DEVICE)
        print(f"Pertes initiales - Train: {train_loss} | Validation: {valid_loss}")

    if not os.path.isdir('Models/'):
        os.mkdir('Models/')

    EPOCH = start_epoch
    MODEL_NAME = f"model_fastPET_UNET{DIM}D_{LOSS}_{str(EPOCH).zfill(3)}.pt"
    LAST_MODEL = MODEL_NAME

    for i in range(start_epoch + 1, N_EPOCHS + 1):
        if i % DECAY_EVERY_N_EPOCHS == 0:
            current_lr = current_lr * DECAY_RATE
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        print(f'[{i}]')

        train_loss = train_fn(tLoader, model, optimizer, DEVICE)
        valid_loss = valid_fn(vLoader, model, DEVICE)

        # Sauvegarde du dernier modèle
        if LAST_MODEL != MODEL_NAME:
            try:
                os.remove('Models/' + LAST_MODEL)
            except OSError:
                pass
        LAST_MODEL = MODEL_NAME.replace(str(EPOCH).zfill(3), str(i).zfill(3))
        torch.save(model.state_dict(), 'Models/' + LAST_MODEL)

        # Sauvegarde du meilleur modèle
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

        # Sauvegarde du checkpoint tous les 10 epochs
        if i % 1 == 0:
            save_checkpoint(
                epoch=i,
                model=model,
                optimizer=optimizer,
                loss=valid_loss,
                current_lr=current_lr
            )

        print(f"[Epoch {i}] Train loss: {train_loss} | Validation loss: {valid_loss}")


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    # Pour reprendre depuis un checkpoint spécifique:
    main(resume_from_checkpoint='checkpoints/checkpoint_epoch_034.pt')
    # Pour commencer un nouvel entraînement:
    #main()