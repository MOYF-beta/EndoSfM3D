# Attention Encoder DoRA Experiment

## Setup Instructions

## Implementation Details

This experiment implements an attention encoder with DoRA (Weight-Decomposed Low-Rank Adaptation) for endoscopic depth estimation:

- **Main Entry Point**: `train_attn_encoder_dora.py` - Runs the complete experiment
- **Core Components**:
  - `src/trainer_attn_encoder.py` - Main training logic with attention mechanisms
  - `dares/networks/dares_peft.py` - DARES model with PEFT/DoRA support
  - `endo3dac/utils/layers.py` - SSI Loss implementation

## Quick Start

### 1. Install Dependencies

Create a Python environment and install dependencies:

```bash
conda create -n attn_encoder_dora python=3.9
conda activate attn_encoder_dora

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio

# Install other dependencies
pip install numpy matplotlib pillow opencv-python
pip install tensorboard torchmetrics
pip install peft  # For parameter-efficient fine-tuning
```

### 2. Dataset Setup

You need to set up the SCARED and C3VD datasets:

1. **SCARED Dataset**: Place the SCARED dataset in `./data/SCARED_Images_Resized/`
2. **C3VD Dataset**: Place the C3VD dataset in `./data/C3VD_as_SCARED/`

The expected directory structure for datasets:

```
data/
├── SCARED_Images_Resized/
│   ├── splits/
│   │   ├── train_files.txt
│   │   ├── val_files.txt
│   │   └── test_files.txt
│   └── [dataset images and depth files]
└── C3VD_as_SCARED/
    ├── splits/
    │   ├── train_files.txt
    │   └── test_files.txt
    └── [dataset images and depth files]
```

### 3. Setup Pre-trained Weights (Optional)

Place any pre-trained DARES weights in the `pretrained_weights/` directory:
- `./pretrained_weights/best/` (for DARES af_sfmlearner weights)

## Running the Experiment

```bash
python train_attn_encoder_dora.py
```

The script will automatically:
1. Run SCARED dataset training and evaluation
2. Run C3VD dataset training and evaluation

## Configuration

To customize dataset paths, edit `src/exp_setup.py`:

```python
# Update these paths in exp_setup.py
DEFAULT_SCARED_PATH = './data/SCARED_Images_Resized'
DEFAULT_C3VD_PATH = './data/C3VD_as_SCARED'
```
