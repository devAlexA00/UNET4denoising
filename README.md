# U-Net 4 denoising
Adaptation of the U-Net architecture for **denoising medical images**.

## CNN Architecture
The U-Net is a CNN architecture designed by Ronneberger, Fischer and Brox (2015), for the purpose of biomedical image segmentation. Here, an adaptation of this architecture was built for biomedical image denoising. In particular, this project was developed for restoring fast-acquisition PET images to their standard quality. 2D and 3D implementations are provided.

### 3D U-Net

![u-net_3d_inv](https://github.com/luisacbscs/UNET4denoising/assets/100357143/881a0b5e-9a22-4cd5-b4d3-64f1b820c1a8)

The network's input is, by default, a 128×128×128 vx volume. The number of encoder/decoder blocks was reduced to three to reduce computational complexity, as we are dealing with 3D images. Each encoder block is composed by two convolutional layers followed by a downsampling layer. Each convolutional layer contains a **convolution** that doubles the number of feature maps (except in the input block, where, by default, 64 feature maps are extracted from the 1-channel input) and a **ReLU activation**. Batch normalisation was removed, as the scale of each PET image, in particular, is not fixed-range and contains important physiological information which musn't be altered. Downsampling is done through **average pooling**, instead of max pooling, as denoising is not a classification problem.

The implementation of this CNN on PyTorch is in `fastPET_UNet3D.py`.

### 2D U-Net

![u-net_inv](https://github.com/luisacbscs/UNET4denoising/assets/100357143/474fb986-5c5a-48c6-ae83-fe9d4718bd07)

The network's input is, by default, a 144×144 px image. To do this, as the images are 3D arrays, each volume must be divided in its axial slices, building a 2D dataset. The composition of the different blocks is similar to that of the 3D version.

The implementation of this CNN on PyTorch is in `fastPET_UNet2D.py`.

## Training the network

`fastPET_UNet_training.py` is the training script. The first block contains the customisable parameters:

- `TARGET_PATH` and `IM_PATH` will store the paths to the folders that contain the target and training images, respectively.
- `DIM` must be set to `2` if the 2D U-Net is to be used (2D dataset) or to `3` if the 3D U-Net is to be used (3D dataset).
- `LOSS` refers to the loss function to be used during training: `MSE` and `L1`

`makeTorchDataset.py` is an auxiliary file that receives a dataframe with the training pairs' paths and returns the corresponding images as torch tensors.

## Model Application

Work in progress...


