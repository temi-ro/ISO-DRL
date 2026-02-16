# Deep Reinforcement Learning for Brain Tumour Segmentation

This repository contains the implementation of **PixelDRL-MG**, a Multi-Agent Deep Reinforcement Learning (MARL) framework for 2D brain tumour segmentation, developed as part of an **Independent Study Option (ISO)** at Imperial College London.

The project evaluates the robustness of DRL agents against a standard **2D nnU-Net** baseline on the **BraTS 2021** dataset, specifically focusing on performance under Gaussian noise corruption.

## Key Features
* **PixelDRL-MG Implementation:** A multi-agent actor-critic architecture where each pixel acts as an independent agent.
* **Supervised Anchor:** Hybrid loss function combining RL rewards with Cross-Entropy to stabilise training.
* **Robustness Analysis:** Evaluation of model performance under varying levels of Gaussian noise ($\sigma=0.01$ to $0.5$).
* **Reduced BraTS 2021:** Data pipeline for filtering and processing 2D slices with specific tumour ratios.

## Results Summary
While U-Net outperforms DRL on clean data, **PixelDRL-MG demonstrates superior robustness** to image noise, maintaining functional segmentation where CNNs fail.

| Noise Level ($\sigma$) | PixelDRL-MG (DSC) | U-Net 2D (DSC) |
| :--- | :--- | :--- |
| **0.00 (Clean)** | 0.8302 | **0.8543** |
| **0.01** | **0.8300** | 0.5330 |
| **0.10** | **0.8295** | 0.5060 |
| **0.30** | **0.7990** | 0.4652 |
| **0.50** | **0.6634** | 0.4288 |

## Usage

### Prerequisites
* Python 3.8+
* PyTorch (tested on NVIDIA A100)
* BraTS 2021 Dataset

### Training
To train the PixelDRL-MG agent with the supervised anchor:
```bash
python training.py --dataset /path/to/brats
```

To evaluate the PixelDRL-MG agent:
```bash
python evaluation.py --data_dir path/to/brats --checkpoint path/to/model.pth
```

