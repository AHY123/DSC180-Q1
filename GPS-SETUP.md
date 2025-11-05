# GraphGPS Setup Instructions

## Overview

This guide covers setting up GraphGPS on modern systems with PyTorch 2.9+ and CUDA 12.8.

## Prerequisites

- Python 3.10
- CUDA-capable GPU (tested with CUDA 12.8)
- Conda (for environment management)

## Installation

### 1. Create Conda Environment

```bash
conda create -n graphgps python=3.10
conda activate graphgps
```

### 2. Install PyTorch with CUDA

```bash
# Install PyTorch 2.9+ with CUDA 12.8
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
```

### 3. Install PyTorch Geometric

```bash
# Install PyG 2.2
conda install pyg=2.2 -c pyg -c conda-forge

# Install PyG extension libraries (matching your PyTorch version)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cu128.html
```

### 4. Install Additional Dependencies

```bash
pip install pytorch-lightning yacs torchmetrics performer-pytorch tensorboardX ogb wandb
```

**Note:** For compatibility with older code, use specific versions:
```bash
pip install 'pytorch-lightning<2.0' 'torchmetrics<1.0'
```

### 5. Verify Installation

```bash
python -c "import torch; import torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Compatibility Fixes (Already Applied)

This repository includes fixes for PyTorch 2.6+ compatibility. The following changes are already in the codebase:

### Automatic torch.load Fix

The code automatically handles PyTorch 2.6+'s new `weights_only=True` default for `torch.load()`. This ensures all dataset loading works correctly without requiring changes to your config files.

### Metric Computation Fix

Regression metrics (RMSE, MAE, MSE) are computed using NumPy for compatibility across different scikit-learn versions.

## Running Experiments

### Basic Training

```bash
conda activate graphgps
cd /path/to/GraphGPS

# Run ZINC experiment
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 3 wandb.use False

# Run Peptides Functional experiment
python main.py --cfg configs/GPS/peptides-func-GPS.yaml --repeat 3 wandb.use False
```

### With Weights & Biases

```bash
# First, login to W&B
wandb login

# Run with W&B logging (using your entity and project)
python main.py --cfg configs/GPS/zinc-GPS+RWSE.yaml --repeat 1 \
  wandb.use True wandb.entity androdro wandb.project zinc_gpu

# Run Peptides Functional with W&B
python main.py --cfg configs/GPS/peptides-func-GPS.yaml --repeat 1 \
  wandb.use True wandb.entity androdro wandb.project graphgps
```

## Troubleshooting

### Dataset Loading Errors

If you encounter `Weights only load failed` errors:
- The fixes are already applied in the codebase
- Make sure you're using the latest version of `main.py` and dataset loaders
- If issues persist, check that all PyG extension libraries are correctly installed

### CUDA Errors

If you see CUDA-related errors:
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check CUDA version matches: `python -c "import torch; print(torch.version.cuda)"`
- Ensure PyG extension libraries match your PyTorch/CUDA version

### W&B Permission Errors

If you get `403: permission denied` from Weights & Biases:
- Ensure the project `zinc_gpu` exists in the `androdro` workspace (create it via W&B web interface if needed)
- Verify you're logged in: `wandb whoami`
- Try omitting `wandb.entity` to use your personal workspace instead

## Environment Notes

- **Tested on:** WSL2 (Linux 5.15.167.4), PyTorch 2.9.0, CUDA 12.8
- **GPU:** Any CUDA-capable GPU should work
- **Compatibility:** These fixes ensure compatibility with PyTorch 2.6+ while maintaining identical training behavior to PyTorch 1.13

## Next Steps

1. Review the config files in `configs/GPS/` to customize experiments
2. Check the main README.md for dataset-specific instructions
3. Start training with your chosen configuration

## Support

For issues specific to GraphGPS, refer to the original repository. For setup issues, ensure all dependencies match the versions specified above.

