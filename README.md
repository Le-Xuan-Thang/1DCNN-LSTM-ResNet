# 1DCNN-LSTM-ResNet

A hybrid deep learning architecture combining **1D Convolutional Neural Networks (1DCNN)**, **Long Short-Term Memory (LSTM)** networks, and **Residual Networks (ResNet)** for structural damage detection.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## 📋 Overview

This repository contains the implementation of the **1DCNN-LSTM-ResNet** model for structural health monitoring (SHM) and damage detection. The model is designed to classify structural damage states using vibration signals from structural monitoring systems.

### Key Features

- **1DCNN** for automatic feature extraction from raw time-series vibration data
- **LSTM** layers to capture long-term temporal dependencies in sequential data
- **ResNet** with dilated convolutions to address the vanishing gradient problem and enable deeper network training
- **Double skip connections** for improved gradient flow and feature preservation
- **Global pooling** combining average and max pooling for robust feature aggregation

## 🏗️ Model Architecture

```
Input (5 channels × 8000 timesteps)
         │
    Conv1D (7×1, stride=2)
         │
    BatchNorm + ReLU
         │
    ┌────┴────┐
    │         │
 ResNet    ResNet (Dilated)
 Block      Block
    │         │
    └────┬────┘
         │ (Add)
         │
     LSTM (128 units)
         │
    ┌────┼────┐
    │    │    │
  Skip  Skip  LSTM
    1    2   Output
    │    │    │
    └────┼────┘
         │ (Concatenate)
         │
    GlobalAvgPool + GlobalMaxPool
         │
    Dense (128) + ReLU
         │
    Dense (num_classes) + Softmax
         │
       Output
```

## 🗃️ Dataset

This model is validated on the **Z24 Bridge Dataset**, a benchmark dataset widely used in structural health monitoring research.

### About Z24 Bridge

The Z24 bridge was a post-tensioned concrete highway bridge in Switzerland that was progressively damaged before demolition in 1998. Extensive vibration measurements were collected under various damage scenarios, making it an ideal dataset for validating damage detection algorithms.

### Data Format

- **Input**: 5 acceleration channels × 8000 time samples
- **Output**: 15-class damage classification labels
- **Files Expected**:
  - `new_data/train_Z24_08_08.p` - Training data (pickle format)
  - `new_data/valid_Z24_08_08.p` - Validation data (pickle format)

## 📊 Performance

| Model | Accuracy |
|-------|----------|
| 1DCNN | 78.7% |
| LSTM | 79.3% |
| 1DCNN-LSTM | 80.6% |
| **1DCNN-LSTM-ResNet (Ours)** | **81.5%** |

*Results on Z24 Bridge 15-class damage detection problem*

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- scikit-learn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Le-Xuan-Thang/1DCNN-LSTM-ResNet.git
cd 1DCNN-LSTM-ResNet
```

2. Install dependencies:
```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

3. Prepare your data in the `new_data/` directory.

### Usage

Run the training script:

```bash
python DCNN-LSTM-ResNet.py
```

The script will:
1. Load and preprocess the Z24 bridge dataset
2. Split data into train/validation/test sets (60%/20%/20%)
3. Train the 1DCNN-LSTM-ResNet model with early stopping
4. Generate confusion matrices and classification reports
5. Save the best model to `model_1DCNN_LSTM_ResNet.h5`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Epochs | 100 (max) |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |
| Early Stopping | Patience = 30 epochs |

## 📁 Project Structure

```
1DCNN-LSTM-ResNet/
├── DCNN-LSTM-ResNet.py          # Main training script
├── README.md                     # This file
├── 1DCNN-LSTM-ResNet 2023.pdf   # Research paper
├── model_1DCNN_LSTM_ResNet.h5   # Trained model (generated)
└── new_data/                     # Dataset directory
    ├── train_Z24_08_08.p        # Training data
    └── valid_Z24_08_08.p        # Validation data
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@article{1dcnn_lstm_resnet_2023,
  title={A novel approach model design for signal data using 1DCNN combing with LSTM and ResNet for damaged detection problem},
  author={Le-Xuan, Thang and Bui-Tien, Thanh and Tran-Ngoc, Hoa},
  booktitle={Structures},
  volume={59},
  pages={105784},
  year={2024},
  organization={Elsevier},
  doi={10.1016/j.istruc.2023.105784}
}
```

## 📬 Contact

- **Author**: Le Xuan Thang
- **GitHub**: [@Le-Xuan-Thang](https://github.com/Le-Xuan-Thang)
- **Email**: lexuanthang.official@gmail.com; lexuanthang.official@outlook.com


## 🙏 Acknowledgments

- Z24 Bridge benchmark dataset administrators
- TensorFlow and Keras development teams
- Structural Health Monitoring research community
