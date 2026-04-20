#!/bin/bash

# EIT Reconstruction Script - Electrical Impedance Tomography Reconstruction using Diffusion Models
# Function: Run EIT reconstruction algorithm based on diffusion models

# ==========================================
# Environment Configuration
# ==========================================
# Set GPU (Defaulting to GPU 1 as per export)
export CUDA_VISIBLE_DEVICES=1

# Activate conda environment
source /amax/haibo/miniconda3/etc/profile.d/conda.sh
conda activate fenics

# ==========================================
# Data Configuration
# ==========================================
DATA_PATH="data/test_data.npy"  # Path to test data

# ==========================================
# Model Configuration
# ==========================================
MESH_SIZE=256                    # Mesh size
DEVICE="cuda"                    # Computing device (cuda/cpu)
SEED=${SEED:-$RANDOM}            # Random seed (defaults to random generation)
SEED=7752

# ==========================================
# Model Path Configuration
# ==========================================
CHECKPOINT_PATH="work_dir/diffusion/checkpoints/checkpoint_60000.pth"  # Diffusion model checkpoint
FNO_MODEL_PATH="work_dir/fno/20251012_023152/best_model.pth"               # FNO model path
ADJOINT_FNO_PATH="work_dir/adjoint_fno/20251011_063346/checkpoints/best_adjoint_fno_model.pth" # Adjoint FNO model path


# ==========================================
# Reconstruction Algorithm Configuration
# ==========================================
MAX_ITERATIONS=3000             # Maximum iterations
LEARNING_RATE=0.005              # Learning rate
GRADIENT_METHOD="autodiff"      # Gradient calculation method: adjoint_fno or autodiff

# === NEW: Select Ablation Mode (dilo | stochastic | ddis) ===
ABLATION_MODE="ddis"

# ==========================================
# Execute Reconstruction
# ==========================================
echo "Running in ABLATION MODE: ${ABLATION_MODE}"

python reconstruction.py \
    --data_path "$DATA_PATH" \
    --mesh_size "$MESH_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --max_iterations "$MAX_ITERATIONS" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_method "$GRADIENT_METHOD" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --fno_model_path "$FNO_MODEL_PATH" \
    --adjoint_fno_path "$ADJOINT_FNO_PATH" \
    --ablation_mode "$ABLATION_MODE"