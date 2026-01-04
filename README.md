# TEM Segmentation using U-Net
<img width="1087" height="413" alt="image" src="https://github.com/user-attachments/assets/5f3ee551-64d9-4f59-91e9-c97fb850b837" />

This repository contains a PyTorch implementation of a deep learning pipeline for segmenting Transmission Electron Microscopy
(TEM) images using a modified U-Net architecture. The project has a large focus on handling high-resolution biomedical imagery,
strong class imbalance, and limited labeled data.

The model is trained to perform multi-class semantic segmentation, assigning each pixel in a TEM image to one of five biological
classes.


## Project Overview
Transmission Electron Microscopy (TEM) produces extremely high-resolution grayscale images that reveal cellular ultrastructure.
Manual annotation of these images is time-consuming and subjective, motivating automated segmentation methods.

This project implements a patch-based U-Net with:

- Residual blocks
- A self-attention module at the bottleneck
- Skip connections for spatial detail preservation
- A combined Cross-Entropy + Dice loss to address class imbalance

The trained model produces pixel-wise segmentation masks for both labeled training images and unlabeled test images.


## Model Architecture
The network is based on a U-Net encoder–decoder architecture with the following modifications:

- Encoder
  - Initial convolution to 32 channels
  - Three downsampling stages
  - Each stage contains two residual blocks
- Bottleneck
  - Two residual blocks
  - Self-attention module to capture global context
- Decoder
  - Nearest-neighbor upsampling
  - Skip connections from the encoder
  - Convolutional refinement
- Output
  - 1×1 convolution
  - Five output channels (one per class)

This design balances local spatial accuracy with global contextual awareness, which is critical for TEM segmentation.

<img width="1550" height="1349" alt="image" src="https://github.com/user-attachments/assets/137dc173-5e19-43a7-afe8-0fe1539dfd94" />


## Dataset Description
### Training Data

- High-resolution TEM images with corresponding label masks
- Labels encoded using five discrete grayscale values:
  - 0 – Background
  - 50 – Blood vessels
  - 100 – Myelinated axons
  - 150 – Unmyelinated axons
  - 200 – Schwann cells

Labels are remapped to integer class IDs [0–4] during preprocessing.

### Testing Data

- TEM images without labels
- Used for qualitative evaluation only


## Preprocessing Pipeline
Because TEM images are extremely large and vary in size, a patch-based approach is used:
- Images are normalized for consistent intensity scaling
- Each training image is randomly sampled into 256×256 patches
- 80 patches per image are sampled during training
- Deterministic center crops are used during validation
- Joint augmentation of images and labels:
  - Random rotations
  - Horizontal/vertical flips

This approach reduces memory usage, improves generalization, and mitigates overfitting.


## Loss Function
Training uses a combined loss:

                                          𝐿 = CrossEntropy + 𝜆 * SoftDice,    𝜆 = 0.5

- Cross-entropy loss encourages correct pixel classification
- Dice loss improves performance on underrepresented classes

This combination provides more stable training than cross-entropy alone.


## Evaluation Metrics
The model is evaluated using:
- Dice score
- Intersection over Union (IoU)

Metrics are computed per class and averaged across all five classes.

Because the test set is unlabeled, final evaluation on test images is qualitative, based on visual inspection of predicted
segmentation masks.
