# denUNET
Adaptation of the U-Net architecture for denoising medical images.

The U-Net is a CNN architecture designed by Ronneberger, Fischer and Brox (2015), for the purpose of biomedical image segmentation. Here, an adaptation of this architecture was built for biomedical image denoising.

- denUNet2D.py and denUNet3D.py contain the PyTorch implementation of the 2D and 3D networks.
- makeTorchDataset.py is an auxiliary file that receives a dataframe with the training pairs' paths and returns the corresponding images as torch tensors.
- denUNet_train.py is the customisable file used to train the network. It expects the paths to the images and targets (assuming they're ordered) and the training parameters.
