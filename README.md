# Scaling Laws for ConvNeXt on ECG Data

## Introduction
This project aims to determine the scaling laws for the ConvNeXt model when applied to ECG (Electrocardiogram) data. Scaling laws help in understanding how the performance of a model changes with respect to its size, the amount of data, and available compute.

## Dataset

We use the [PhysioNet 2021](https://physionet.org/content/challenge-2021/1.0.3/) dataset.
The dataset include >88.000 annotated twelve-lead ECG recordings from six sources in four countries across three continents.
1. CPSC Database and CPSC-Extra Database
2. INCART Database
3. PTB and PTB-XL Database
4. The Georgia 12-lead ECG Challenge (G12EC) Database
5. Chapman-Shaoxing and Ningbo Database
6. The University of Michigan (UMich) Database

## Model
We utilize the ConvNeXt model, a convolutional neural network architecture, known for its efficiency and performance in image and signal processing tasks.
Key features are:
- Stochastic depth
- LayerScale
- LayerNorm
- Inverted bottleneck
- GELU
- Depth wise convolution
- ... and more
