# DiLO: Decoupling Generative Priors and Neural Operators via Diffusion Latent Optimization for Inverse Problems

![example](https://github.com/haibo-research/DiLO/blob/main/utils/pipeline_overview.png) 

## Abstract

Diffusion models have emerged as powerful generative priors for solving PDE-constrained inverse problems. Compared to end-to-end approaches relying on massive paired datasets, explicitly decoupling the prior distribution of physical parameters from the forward physical model, a paradigm often formalized as Plug-and-Play (PnP) priors, offers enhanced flexibility and generalization. To accelerate inference within such decoupled frameworks, fast neural operators are employed as surrogate solvers. However, directly integrating them into standard diffusion sampling introduces a critical bottleneck: evaluating neural surrogates on partially denoised, non-physical intermediate states forces them into out-of-distribution (OOD) regimes. To eliminate this, the physical surrogate must be evaluated exclusively on the fully denoised parameter, a principle we formalize as the Manifold Consistency Requirement. To satisfy this requirement, we present Diffusion Latent Optimization (DiLO), which transforms the stochastic sampling process into a deterministic latent trajectory, enabling stable backpropagation of measurement gradients to the initial latent state. By keeping the trajectory on the physical manifold, it ensures physically valid updates and improves reconstruction accuracy. We provide theoretical guarantees for the convergence of this optimization trajectory. Extensive experiments across Electrical Impedance Tomography, Inverse Scattering, and Inverse Navier-Stokes problems demonstrate DiLO's accuracy, efficiency, and robustness to noise.

## Getting Started

### 1) Clone the repository

```bash
git clone [https://github.com/haibo-research/DiLO.git](https://github.com/haibo-research/DiLO.git)
cd DiLO
<br />

2) Download pretrained checkpoints (autoencoders and model)
Our work utilizes the pre-trained Latent Diffusion Models trained on the FFHQ dataset, as introduced in High-resolution image synthesis with latent diffusion models (CVPR 2022).

Please run the following commands to download the necessary checkpoints:

Bash
mkdir -p models/ldm
wget [https://ommer-lab.com/files/latent-diffusion/ffhq.zip](https://ommer-lab.com/files/latent-diffusion/ffhq.zip) -P ./models/ldm
unzip models/ldm/ffhq.zip -d ./models/ldm

mkdir -p models/first_stage_models/vq-f4
wget [https://ommer-lab.com/files/latent-diffusion/vq-f4.zip](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) -P ./models/first_stage_models/vq-f4
unzip models/first_stage_models/vq-f4/vq-f4.zip -d ./models/first_stage_models/vq-f4
<br />

3) Set environment
Install the required dependencies via conda:

Bash
conda env create -f environment.yaml
conda activate dilo
(Note: Please ensure you have the appropriate CUDA Toolkit installed matching your PyTorch version).

<br />

4) Train Neural Operator
To train the fast neural operator serving as the surrogate solver for the forward physical model, run:

Bash
python run_fno.py
<br />

5) Inference / Reconstruction
To perform the inference and solve the inverse problems using the proposed Diffusion Latent Optimization (DiLO), run:

Bash
python run_reconstruction.py
The current code supports PDE-constrained inverse problems including Electrical Impedance Tomography, Inverse Scattering, and Inverse Navier-Stokes. You can modify the configuration files or arguments to switch between different physical tasks.

<br />

Citation
If you find our work or this code useful for your research, please consider citing our paper:

Code snippet
@article{liu2026dilo,
  title={DiLO: Decoupling Generative Priors and Neural Operators via Diffusion Latent Optimization for Inverse Problems},
  author={Liu, Haibo and Lin, Guang},
  journal={arXiv preprint arXiv:2604.11375},
  year={2026}
}
Acknowledgements
This repository is built upon the Latent Diffusion Models codebase. We thank the authors for their open-source contributions.