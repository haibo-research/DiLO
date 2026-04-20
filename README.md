# DiLO: Decoupling Generative Priors and Neural Operators via Diffusion Latent Optimization for Inverse Problems

![example](https://github.com/haibo-research/DiLO/blob/main/utils/pipeline_overview.png) 

## Abstract

In this work, we propose DiLO, an algorithm that can solve PDE-constrained inverse problems. Our algorithm explicitly decoupling the diffusion prior distribution of physical parameters from the forward physical model. Using fast neural operators to accelerate inference within such decoupled frameworks. To eliminate out-of-distribution (OOD), we transforms the stochastic sampling process into a deterministic latent trajectory, enabling stable backpropagation of measurement gradients to the initial latent state.

## Getting Started

### 1) Clone the repository

```
git clone https://github.com/haibo-research/DiLO.git
cd DiLO
```
<br /> 

### 2) Download pretrained checkpoints (autoencoders and model)
Our work utilizes the pre-trained Latent Diffusion Models trained on the FFHQ dataset, as introduced in High-resolution image synthesis with latent diffusion models (CVPR 2022).

Please run the following commands to download the necessary checkpoints:

```
mkdir -p models/ldm
wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
```
<br />

### 3) Set environment
Install the required dependencies via conda:

```
conda env create -f environment.yaml
conda activate dilo
```
(Note: Please ensure you have the appropriate CUDA Toolkit installed matching your PyTorch version).

<br />

### 4) Train Neural Operator
To train the fast neural operator serving as the surrogate solver for the forward physical model, run:

```
bash run_fno.sh
```
<br />

### 5) Inference / Reconstruction
To perform the inference and solve the inverse problems using the proposed Diffusion Latent Optimization (DiLO), run:

```
bash run_reconstruction.sh
```

The current code supports PDE-constrained inverse problems including Electrical Impedance Tomography, Inverse Scattering, and Inverse Navier-Stokes. You can modify the configuration files or arguments to switch between different physical tasks.

<br />

## Citation
If you find our work or this code useful for your research, please consider citing our paper:

```
@article{liu2026dilo,
  title={DiLO: Decoupling Generative Priors and Neural Operators via Diffusion Latent Optimization for Inverse Problems},
  author={Liu, Haibo and Lin, Guang},
  journal={arXiv preprint arXiv:2604.11375},
  year={2026}
}
```

Acknowledgements
This repository is built upon the Latent Diffusion Models codebase. We thank the authors for their open-source contributions.