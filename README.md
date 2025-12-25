# FMD-DWTSUNet: Seismic Data Denoising using Feature Mode Decomposition and Dual-Wavelet Transform Spatial Attention UNet

## Overview

This repository implements a deep learning-based seismic data denoising method that combines **Feature Mode Decomposition (FMD)** with a **Dual-Wavelet Transform Spatial Attention UNet (DWT-S-UNet)** architecture. The method effectively removes noise from seismic data while preserving important signal features.

## Key Features

- **FMD Preprocessing**: Feature Mode Decomposition for signal decomposition and noise reduction
- **Dual-Branch Architecture**: Combines Discrete Wavelet Transform (DWT) and traditional UNet branches
- **Spatial Attention Mechanism**: Integrates spatial attention gates for enhanced feature extraction
- **Multiple Model Implementations**: Includes various baseline models (UNet, DWTUNet, ASCNet, SMCNN, etc.)
- **Comprehensive Evaluation**: Supports evaluation on synthetic and field seismic data
- **Batch Processing**: Efficient batch processing for large-scale seismic data

## Project Structure

```
FMD-DWTSUNet/
├── data/                          # Data directory
│   ├── data_mat_npy_sgy/          # Synthetic seismic data in multiple formats
│   │   ├── mat/                   # MATLAB format data files
│   │   ├── npy/                   # NumPy format data files (SNR variants)
│   │   └── part4/                 # Part 4 of 2007BP dataset
│   ├── field_data/                # Real-world field seismic data
│   └── record_result/             # Experimental results and outputs
│       ├── part4/                 # Results on part4 synthetic data
│       ├── real3/                 # Results on real3 field data
│       ├── sea/                   # Results on sea field data
│       └── train_result/          # Training results and model checkpoints
├── __pycache__/                   # Python cache files
├── *.py                           # Python source files (see detailed descriptions below)
└── *.npy                          # Preprocessed FMD data files
```

## File Descriptions

### Core Model Files

- **`DWT_S_UNet.py`**: Main implementation of the DWT-S-UNet model with dual-branch architecture (DWT branch + UNet branch) and spatial attention fusion
- **`DWT_S_UNet_ReLU.py`**: Variant of DWT-S-UNet using ReLU activation instead of ELU
- **`DWTUNet.py`**: UNet variant incorporating Discrete Wavelet Transform
- **`DWTIUNet.py`**: Improved DWT-UNet variant
- **`UNet.py`**: Standard UNet implementation for 1D seismic data
- **`ASCNet.py`**: Attention-based Seismic Convolutional Neural Network implementation
- **`SMCNN.py`**: Seismic Multi-scale Convolutional Neural Network
- **`DnCNN.py`**: Denoising Convolutional Neural Network baseline

### Feature Mode Decomposition

- **`FMD.py`**: Implementation of Feature Mode Decomposition algorithm using Maximum Correlated Kurtosis Deconvolution (MCKD) for signal decomposition
- **`Get_FMD_patches.py`**: Batch processing utilities for FMD decomposition on seismic data patches

### Data Processing

- **`get_patches.py`**: Utilities for extracting and processing seismic data patches, including:
  - Reading SEGY format seismic data
  - Patch extraction with sliding window
  - Data normalization and preprocessing
  - SNR and RMSE calculation
  - Seismic data visualization (time-domain and f-k domain)

### Training and Inference

- **`train.py`**: Main training script for training denoising models with:
  - Data loading and preprocessing
  - Model training with validation
  - Loss tracking and SNR monitoring
  - Model checkpoint saving
  - Training curve visualization

- **`denoise.py`**: Inference script for denoising seismic data:
  - Loads trained model weights
  - Processes seismic data (synthetic or field data)
  - Reconstructs denoised seismic sections
  - Visualizes results (time-domain and f-k domain)

- **`denoise_snrs.py`**: Batch evaluation script for testing models on multiple SNR levels:
  - Processes multiple noise levels (SNR -10 to +6 dB)
  - Generates performance metrics (RMSE, SNR, inference time)
  - Exports results to Excel format
  - Creates performance curves

### Utility Files

- **`temp.py`**: Temporary testing/experimental scripts

## Directory Descriptions

### `data/data_mat_npy_sgy/`
Contains synthetic seismic data in multiple formats:
- **`mat/`**: MATLAB format files with different SNR levels (snr_-10.mat to snr_6.mat)
- **`npy/`**: NumPy format files with different SNR levels (snr_-10.npy to snr_6.npy)
- **`part4/`**: Part 4 of the 2007BP synthetic dataset (MAT, NPY, and SEGY formats)

### `data/field_data/`
Contains real-world field seismic data:
- **`real3.mat`**: Real seismic data in MATLAB format
- **`Sea_0_1_shot.mat/npy/sgy`**: Marine seismic shot data

### `data/record_result/`
Stores experimental results and outputs:
- **`part4/`**: Denoising results on part4 synthetic data, including:
  - Results from different methods (ASCNet, FMD-DWTS-UNet, SMCNN, VMD, Wavelet)
  - Ablation study results (DWTSUNet, DWTTUNet, DWTUNet, FMD-UNet, UNet)
  - Visualization images (time-domain: d.png, dn.png; f-k domain: fkd.png, fkdn.png)
- **`real3/`**: Results on real3 field data
- **`sea/`**: Results on sea field data
- **`train_result/`**: Training results including:
  - Trained model weights (`model.pth`)
  - Training/validation loss curves (`train_loss.txt`, `val_loss.txt`)
  - SNR metrics (`snr_train_after.txt`, `snr_val_after.txt`)
  - Performance curves (`result_curves.png`)
  - Evaluation metrics (`result.xlsx`)

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy scipy matplotlib
pip install segyio
pip install pywavelets
pip install pytorch-wavelets
pip install pandas openpyxl
```

### Dependencies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **SciPy**: Scientific computing (signal processing, I/O)
- **Matplotlib**: Visualization
- **segyio**: SEGY file format support
- **PyWavelets**: Wavelet transform operations
- **pytorch-wavelets**: PyTorch-compatible wavelet transforms
- **Pandas**: Data analysis and Excel export

## Usage

### Training

1. Prepare your seismic data in SEGY format and place it in the `data/` directory
2. Modify hyperparameters in `train.py`:
   ```python
   Data_Size = 256      # Patch size
   EPOCH = 200          # Number of training epochs
   BATCH_SIZE_s = 100   # Training batch size
   LR = 0.0001          # Learning rate
   ```
3. Run training:
   ```bash
   python train.py
   ```

### Inference

1. Load a trained model and process seismic data:
   ```bash
   python denoise.py
   ```
   Modify the model path and data path in the script as needed.

2. Batch evaluation on multiple SNR levels:
   ```bash
   python denoise_snrs.py
   ```

### Using FMD Preprocessing

To enable FMD preprocessing, set `fmd = True` in the training or inference scripts. This will:
- Decompose input signals into multiple modes using FMD
- Concatenate FMD modes with original data as input features
- Improve denoising performance, especially for low SNR data

## Model Architecture

The DWT-S-UNet architecture consists of:

1. **Dual Encoder Branches**:
   - **DWT Branch**: Uses Discrete Wavelet Transform for downsampling
   - **UNet Branch**: Traditional UNet encoder with max pooling

2. **Bottleneck Fusion**: Combines features from both branches

3. **Hybrid Decoder**: Four-channel fusion upsampling that combines:
   - Spatial attention-enhanced DWT features
   - Original DWT features
   - Spatial attention-enhanced UNet features
   - Original UNet features

4. **Spatial Attention Gates**: Applied to both branches for enhanced feature extraction

## Results

Results are saved in `data/record_result/` including:
- Denoised seismic sections (time-domain)
- F-K domain spectra
- Performance metrics (SNR, RMSE)
- Training curves

## Citation

If you use this code in your research, please cite:

```bibtex
@article{fmd_dwtsunet,
  title={FMD-DWTSUNet: Seismic Data Denoising using Feature Mode Decomposition and Dual-Wavelet Transform Spatial Attention UNet},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- 2007BP synthetic dataset
- PyTorch community
- Contributors to open-source seismic processing libraries

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

