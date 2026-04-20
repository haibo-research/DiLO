#!/bin/bash

# FNO Model Optimized Training Configuration Script - Fourier Neural Operator for solving PDEs
# Data-driven solver based on the Darcy Flow equation
# Implements efficient operator learning using frequency-domain convolutions
# 
# Optimization Objectives:
# - Reduce model parameters by ~70% (from ~1.2M down to ~350K)
# - Improve training speed by 2-3x
# - Maintain similar prediction accuracy

# ==========================================
# Environment Configuration
# ==========================================
# Set to use GPU index 6
export CUDA_VISIBLE_DEVICES=6

# Activate conda environment
source /amax/haibo/miniconda3/etc/profile.d/conda.sh
conda activate fenics

# ==========================================
# Optimized FNO Config - Faster Training Speed
# ==========================================
N_GRID=255              # Grid resolution (maintaining original) - Note: FNO prefers power-of-2 dimensions
BATCH_SIZE=128           # Batch size (increased to improve GPU utilization)
EPOCHS=2000              # Training epochs (reduced further for more efficient training)

# ==========================================
# Optimizer Parameters
# ==========================================
LEARNING_RATE=0.001     # Learning rate (slightly increased to accelerate convergence)

# ==========================================
# FNO Architecture Parameters - Controls Model Scale and Complexity
# ==========================================
WIDTH=20                # Network width (number of channels, affects model capacity)
MODES1=8                # Fourier modes for 1st dimension (frequency truncation, affects precision)
MODES2=8                # Fourier modes for 2nd dimension (frequency truncation, affects precision)
LAYERS=4                # Number of FNO layers (network depth, affects expressive power)

# ==========================================
# Network Scale Configuration Options (Selectable configurations)
# ==========================================
# Small Network (Fast training, ~200K parameters):
# WIDTH=16, MODES1=6, MODES2=6, LAYERS=3

# Medium Network (Balanced performance, ~350K parameters):
# WIDTH=20, MODES1=8, MODES2=8, LAYERS=4

# Large Network (High precision, ~800K parameters):
# WIDTH=32, MODES1=12, MODES2=12, LAYERS=5


python train_fno.py \
    --N_grid "$N_GRID" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --width "$WIDTH" \
    --modes1 "$MODES1" \
    --modes2 "$MODES2" \
    --layers "$LAYERS" \
    --mode test \
    --test_dir /amax/haibo/Diffusion4IP/work_dir/fno/20251012_023152