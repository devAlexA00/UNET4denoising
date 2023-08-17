import random
import torch
from torch.utils.data import Dataset
import numpy as np
import itk


class torchDataset(Dataset):
    """
    CLASS THAT RECEIVES A PD.DATAFRAME IN WHICH THE FIRST COLUMN CONTAINS THE PATHS TO THE INPUT IMAGES AND THE
    SECOND CONTAINS THE PATHS TO THE TARGET IMAGES, AND RETURNS THE IMAGES AS TORCH TENSORS.
    """

    def __init__(self, df, PATCH_SIZE=None):

        if PATCH_SIZE is None:
            PATCH_SIZE = 128

        self.PATCH_SIZE = PATCH_SIZE

        # pd.DataFrame
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        PATCH_SIZE = self.PATCH_SIZE

        # Training pair: (input, target)
        # Column 0: paths to the input images
        image_path = self.df.iloc[idx, 0]
        # Column 1: paths do the target images
        target_path = self.df.iloc[idx, 1]

        # Input image
        image = np.asarray(itk.imread(image_path))  # ATTENTION: path must not contain accents
        # Target image
        target = np.asarray(itk.imread(target_path))  # ATTENTION: path must not contain accents

        if len(image.shape) == 3:

            z = random.randint(0, image.shape[0]-PATCH_SIZE)
            y = random.randint(0, image.shape[1]-PATCH_SIZE)
            x = random.randint(0, image.shape[2]-PATCH_SIZE)

            image = image[z:z+PATCH_SIZE, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
            image = np.expand_dims(image, axis=-1)

            target = target[z:z+PATCH_SIZE, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            # Transforming size (D, W, H) to (D, W, H, C) where C is the number of channels
            target = np.expand_dims(target, axis=-1)

            # (D, W, H, C) to (C, D, W, H) <=> (0, 1, 2, 3) to (3, 0, 1, 2)
            image = np.transpose(image, (3, 0, 1, 2)).astype(float)
            target = np.transpose(target, (3, 0, 1, 2)).astype(float)

        else:
            x = random.randint(0, image.shape[0] - PATCH_SIZE)
            y = random.randint(0, image.shape[1] - PATCH_SIZE)

            image = image[x:x + PATCH_SIZE, y:y + PATCH_SIZE]
            # Transforming size (W, H) to (W, H, C) where C is the number of channels
            image = np.expand_dims(image, axis=-1)

            target = target[x:x + PATCH_SIZE, y:y + PATCH_SIZE]
            # Transforming size (W, H) to (W, H, C) where C is the number of channels
            target = np.expand_dims(target, axis=-1)

            # (W, H, C) to (C, W, H) <=> (0, 1, 2) to (2, 0, 1)
            image = np.transpose(image, (2, 0, 1)).astype(float)
            target = np.transpose(target, (2, 0, 1)).astype(float)

        # Numpy to torch tensor
        image = torch.Tensor(image)
        target = torch.Tensor(target)

        return image, target
