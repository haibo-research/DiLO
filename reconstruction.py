import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
matplotlib.rcParams['figure.max_open_warning'] = 50
import os
import sys
import argparse
import random
import time
import json
from datetime import datetime
import importlib.util
import yaml


from dolfin import *
set_log_level(LogLevel.ERROR)


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


sys.path.append(os.path.join(current_dir, "model"))
sys.path.append(os.path.join(current_dir, "model/ldm"))


from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


from utils.load import NpyReader, GaussianNormalizer
from utils.Loss import LpLoss


def get_finetuned_model(checkpoint_path, device='cuda', config_path=None):
    try:
        print(f"Loading finetuned model from {checkpoint_path}")
        
        if config_path is None:
            config_path = "work_dir/diffusion/configs/latent-diffusion/ffhq-ldm-vq-4.yaml"
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        class SimpleConfig:
            def __init__(self, d):
                self._dict = d.copy()
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, SimpleConfig(v))
                    else:
                        setattr(self, k, v)
            
            def __contains__(self, key):
                return key in self._dict
            
            def get(self, key, default=None):
                if hasattr(self, key):
                    attr = getattr(self, key)
                    if isinstance(attr, SimpleConfig):
                        return attr._dict
                    return attr
                return default
            
            def __getitem__(self, key):
                return getattr(self, key)
            
            def __setattr__(self, key, value):
                super().__setattr__(key, value)
                if key != '_dict' and hasattr(self, '_dict'):
                    if isinstance(value, SimpleConfig):
                        self._dict[key] = value._dict
                    else:
                        self._dict[key] = value
        
        if ('model' in config_dict and 'params' in config_dict['model'] and 
            'first_stage_config' in config_dict['model']['params'] and
            'params' in config_dict['model']['params']['first_stage_config'] and
            'ckpt_path' in config_dict['model']['params']['first_stage_config']['params']):
            old_path = config_dict['model']['params']['first_stage_config']['params']['ckpt_path']
            new_path = "work_dir/diffusion/models/first_stage_models/vq-f4/model.ckpt"
            config_dict['model']['params']['first_stage_config']['params']['ckpt_path'] = new_path
            print(f"Modified ckpt_path from {old_path} to {new_path}")
        
        config = SimpleConfig(config_dict)
        
        model = instantiate_from_config(config.model)
        checkpoint = torch.load(checkpoint_path)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print("Successfully loaded finetuned model")
        return model

    except Exception as e:
        print(f"Error loading finetuned model: {e}")
        raise


def save_intermediate_results(iteration, conductivity, loss_val, save_dir, voltage_pred=None, voltage_true=None, ground_truth=None):
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        conductivity_np = conductivity.detach().cpu().numpy()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        if conductivity_np.ndim > 2:
            conductivity_np = conductivity_np.squeeze()
        
        if ground_truth is not None:
            ground_truth_np = ground_truth.detach().cpu().numpy() if hasattr(ground_truth, 'detach') else ground_truth
            if ground_truth_np.ndim > 2:
                ground_truth_np = ground_truth_np.squeeze()

            vmin_conductivity = ground_truth_np.min()
            vmax_conductivity = ground_truth_np.max()
            
            im1 = ax1.imshow(ground_truth_np, cmap='viridis', vmin=vmin_conductivity, vmax=vmax_conductivity)
            ax1.set_title('Ground Truth Target', fontsize=14, fontweight='bold')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, label='Conductivity (S/m)', shrink=0.8)
            
            im2 = ax2.imshow(conductivity_np, cmap='viridis', vmin=vmin_conductivity, vmax=vmax_conductivity)
            ax2.set_title(f'Current Reconstruction (Iter {iteration})\nLoss: {loss_val:.6f}', 
                         fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Conductivity (S/m)', shrink=0.8)
            
            difference = np.abs(ground_truth_np - conductivity_np)
            im3 = ax3.imshow(difference, cmap='Reds')
            ax3.set_title(f'Absolute Difference\nMAE: {difference.mean():.6f}', 
                         fontsize=14, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, label='|GT - Recon|', shrink=0.8)
            
        else:
            ax1.text(0.5, 0.5, 'Ground Truth\nNot Available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.axis('off')
            
            im2 = ax2.imshow(conductivity_np, cmap='viridis')
            ax2.set_title(f'Current Reconstruction (Iter {iteration})\nLoss: {loss_val:.6f}', 
                         fontsize=14, fontweight='bold')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, label='Conductivity (S/m)', shrink=0.8)
            
            ax3.text(0.5, 0.5, 'Difference\nNot Available\n(No Ground Truth)', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.axis('off')
        
        plt.tight_layout()
        
        conductivity_img_path = os.path.join(save_dir, f"comparison_iter_{iteration:04d}.png")
        plt.savefig(conductivity_img_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
        loss_log_path = os.path.join(save_dir, "loss_log.txt")
        with open(loss_log_path, "a") as f:
            f.write(f"Iteration {iteration:4d}, Loss: {loss_val:.8f}\n")
        
        stats_log_path = os.path.join(save_dir, "conductivity_stats.txt")
        with open(stats_log_path, "a") as f:
            mean_val = float(conductivity_np.mean())
            std_val = float(conductivity_np.std())
            min_val = float(conductivity_np.min())
            max_val = float(conductivity_np.max())
            f.write(f"Iteration {iteration:4d}: Mean={mean_val:.4f}, Std={std_val:.4f}, "
                   f"Range=[{min_val:.4f}, {max_val:.4f}] S/m\n")
            
            if ground_truth is not None:
                ground_truth_np = ground_truth.detach().cpu().numpy() if hasattr(ground_truth, 'detach') else ground_truth
                if ground_truth_np.ndim > 2:
                    ground_truth_np = ground_truth_np.squeeze()
                difference = np.abs(ground_truth_np - conductivity_np)
                mae = float(difference.mean())
                mse = float(np.mean((ground_truth_np - conductivity_np)**2))
                f.write(f"           Difference: MAE={mae:.6f}, MSE={mse:.6f}\n")
            
    except Exception as e:
        print(f"Error saving intermediate results for iteration {iteration}: {e}")
        plt.close('all')


def plot_real_time_loss(losses, output_dir, iteration=None):
    try:
        if not losses:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        ax1.plot(losses, 'b-', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Curve (Linear Scale) - Iter {len(losses)-1}')
        ax1.grid(True, alpha=0.3)
        if losses:
            ax1.set_ylim(0, max(losses) * 1.1)
        
        ax2.plot(losses, 'r-', linewidth=2, alpha=0.8)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title(f'Loss Curve (Log Scale) - Iter {len(losses)-1}')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        recent_losses = losses[-100:] if len(losses) > 100 else losses
        recent_iterations = list(range(max(0, len(losses)-100), len(losses)))
        ax3.plot(recent_iterations, recent_losses, 'g-', linewidth=2, alpha=0.8)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Loss')
        ax3.set_title('Recent 100 Iterations')
        ax3.grid(True, alpha=0.3)
        
        if len(losses) > 10:
            window_size = min(10, len(losses))
            moving_avg = []
            for i in range(len(losses)):
                start_idx = max(0, i - window_size + 1)
                moving_avg.append(np.mean(losses[start_idx:i+1]))
            
            ax4.plot(losses, 'b-', alpha=0.5, linewidth=1, label='Original')
            ax4.plot(moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window_size})')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Loss')
            ax4.set_title('Loss with Moving Average')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.7, f'Current Loss: {losses[-1]:.6f}', 
                    transform=ax4.transAxes, ha='center', fontsize=12)
            ax4.text(0.5, 0.5, f'Min Loss: {min(losses):.6f}', 
                    transform=ax4.transAxes, ha='center', fontsize=12)
            ax4.text(0.5, 0.3, f'Iterations: {len(losses)}', 
                    transform=ax4.transAxes, ha='center', fontsize=12)
            ax4.set_title('Loss Statistics')
            ax4.axis('off')
        
        plt.tight_layout()
        
        loss_plot_path = os.path.join(output_dir, 'real_time_loss_curve.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig) 
        
        loss_data_path = os.path.join(output_dir, 'loss_data.txt')
        with open(loss_data_path, 'w') as f:
            f.write("Iteration\tLoss\n")
            for i, loss in enumerate(losses):
                f.write(f"{i}\t{loss:.8f}\n")
           
    except Exception as e:
        print(f"Error plotting real-time loss: {e}")
        plt.close('all') 


class EITLatentDiffusionReconstructorCorrected:
    def __init__(self, mesh_size=64, device='cpu', checkpoint_path=None,
                 fno_model_path=None, adjoint_fno_model_path=None, num_electrodes=8, gradient_method='autodiff'):
        self.mesh_size = mesh_size
        self.device = device
        self.num_electrodes = num_electrodes
        
        self.fno_model = None
        self.adjoint_fno_model = None
        self.fno_expected_input_channels = None  # Store expected input channels for FNO model
        
        self.checkpoint_path = checkpoint_path
        self.diffusion_model = None
        self.sampler = None
        
        if fno_model_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fno_path = os.path.join(current_dir, "model", "FNO.py")
            spec = importlib.util.spec_from_file_location("fno_module", fno_path)
            fno_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fno_module)
            FNO2d = fno_module.FNO2d
            
            # Load checkpoint first to inspect architecture
            checkpoint = torch.load(fno_model_path, map_location=device, weights_only=False)
            checkpoint_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Infer architecture from checkpoint shapes
            # Checkpoint has: fc0.weight [32, 3], conv_layers [32, 32, 12, 12], fc2.weight [640, 64]
            modes1 = checkpoint_dict['conv_layers.0.weights1'].shape[2]
            modes2 = checkpoint_dict['conv_layers.0.weights1'].shape[3]
            width = checkpoint_dict['fc0.weight'].shape[0]
            final_channels = checkpoint_dict['fc0.weight'].shape[1]
            out_channels = checkpoint_dict['fc2.weight'].shape[0]
            
            # Determine in_channels and include_grid based on fc0 input size
            # If include_grid=True: final_channels = in_channels + 2
            # If include_grid=False: final_channels = in_channels
            # Common cases:
            # - final_channels=3: could be (in_channels=3, include_grid=False) OR (in_channels=1, include_grid=True)
            # - final_channels=4: typically (in_channels=2, include_grid=True) - most common for EIT
            # - final_channels=5: typically (in_channels=3, include_grid=True)
            # For EIT, we typically use 2 input channels (sigma, g_lift), so prefer include_grid=True
            if final_channels == 4:
                # Most common case: 2 input channels + 2 grid channels
                in_channels = 2
                include_grid = True
            elif final_channels == 3:
                # Could be 3 channels without grid, or 1 channel with grid
                # For EIT, if final_channels=3, it's more likely to be (in_channels=1, include_grid=True)
                # because EIT typically uses grid coordinates, and 1 channel (sigma) + 2 grid = 3
                # However, we need to handle both cases. Let's try in_channels=1 first.
                # If that doesn't work, we can fall back to in_channels=3
                in_channels = 1
                include_grid = True
            elif final_channels == 5:
                in_channels = 3
                include_grid = True
            else:
                # Default fallback: try to infer
                # If final_channels > 2, likely has grid
                if final_channels >= 4:
                    in_channels = final_channels - 2
                    include_grid = True
                else:
                    # For final_channels <= 2, try without grid first
                    in_channels = final_channels
                    include_grid = False
            
            print(f"Loading FNO model with inferred architecture:")
            print(f"  modes1={modes1}, modes2={modes2}, width={width}")
            print(f"  in_channels={in_channels}, include_grid={include_grid}, out_channels={out_channels}")
            
            self.fno_model = FNO2d(
                modes1=modes1, 
                modes2=modes2, 
                width=width, 
                in_channels=in_channels,
                out_channels=out_channels,
                include_grid=include_grid
            )
            
            # Store expected input channels (before grid addition if include_grid=True)
            self.fno_expected_input_channels = in_channels
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.fno_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.fno_model.load_state_dict(checkpoint)
            self.fno_model.to(device)
            self.fno_model.eval()
            print("FNO model loaded successfully")
        
        if adjoint_fno_model_path and gradient_method == 'adjoint_fno':
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adjoint_fno_path = os.path.join(current_dir, "train_adjoint_fno.py")
            spec = importlib.util.spec_from_file_location("adjoint_fno_module", adjoint_fno_path)
            adjoint_fno_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adjoint_fno_module)
            AdjointFNO2d = adjoint_fno_module.FNO2d
            
            self.adjoint_fno_model = AdjointFNO2d(
                modes1=8,
                modes2=8,
                width=20
            ).to(device)
            
            checkpoint = torch.load(adjoint_fno_model_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.adjoint_fno_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.adjoint_fno_model.load_state_dict(checkpoint)
            self.adjoint_fno_model.eval()
            print("Adjoint FNO model loaded successfully")
        else:
            self.adjoint_fno_model = None
        
        if checkpoint_path:
            self.diffusion_model = get_finetuned_model(checkpoint_path, device=device)
            self.sampler = DDIMSampler(self.diffusion_model)
            self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=0.0, verbose=False)
            print("扩散模型初始化完成")
        
        fno_norm_path = 'data/normalization_params.npy'
        fno_params = np.load(fno_norm_path, allow_pickle=True).item()
        self.normalizers = {
            'sigma': GaussianNormalizer(mean=torch.tensor(fno_params['sigma']['mean']), 
                                        std=torch.tensor(fno_params['sigma']['std'])),
            'g_lift': GaussianNormalizer(mean=torch.tensor(fno_params['g_lift']['mean']), 
                                        std=torch.tensor(fno_params['g_lift']['std']))
        }

        for key in self.normalizers:
            self.normalizers[key].to(device)
        print(f"FNO normalizers loaded from: {fno_norm_path}")
        
        adjoint_norm_path = 'data/adjoint_normalization_params.npy'
        adjoint_params = np.load(adjoint_norm_path, allow_pickle=True).item()
        self.adjoint_normalizers = {
            'sigma': GaussianNormalizer(mean=torch.tensor(adjoint_params['sigma']['mean']), 
                                        std=torch.tensor(adjoint_params['sigma']['std'])),
            'u_diff_lift': GaussianNormalizer(mean=torch.tensor(adjoint_params['u_diff_lift']['mean']), 
                                                std=torch.tensor(adjoint_params['u_diff_lift']['std']))
        }
        for key in self.adjoint_normalizers:
            self.adjoint_normalizers[key].to(device)
        print(f"Adjoint normalizers loaded from: {adjoint_norm_path}")
    
    def prepare_fno_input(self, sigma, g_lift):
        """
        Prepare input for FNO model, handling channel mismatch.
        
        Args:
            sigma: Conductivity tensor [batch, height, width] or [height, width]
            g_lift: Boundary condition tensor [batch, height, width] or [height, width]
        
        Returns:
            Input tensor with correct number of channels for FNO model
        """
        # Ensure batch dimension
        if sigma.dim() == 2:
            sigma = sigma.unsqueeze(0)
        if g_lift.dim() == 2:
            g_lift = g_lift.unsqueeze(0)
        
        # Stack sigma and g_lift
        x_in = torch.stack([sigma, g_lift], dim=-1)  # [batch, height, width, 2]
        
        # If model expects different number of channels, adjust accordingly
        # Note: This should only happen if the checkpoint was trained with different input format
        # For standard EIT FNO models, we use 2 channels (sigma, g_lift) with include_grid=True
        if self.fno_expected_input_channels is not None:
            current_channels = x_in.shape[-1]
            if current_channels < self.fno_expected_input_channels:
                # Pad with zeros (only warn once to avoid spam)
                if not hasattr(self, '_padding_warned'):
                    print(f"Note: FNO model expects {self.fno_expected_input_channels} input channels, "
                          f"padding from {current_channels} channels. "
                          f"This may indicate a checkpoint trained with different input format.")
                    self._padding_warned = True
                padding_shape = list(x_in.shape)
                padding_shape[-1] = self.fno_expected_input_channels - current_channels
                padding = torch.zeros(padding_shape, device=x_in.device, dtype=x_in.dtype)
                x_in = torch.cat([x_in, padding], dim=-1)
            elif current_channels > self.fno_expected_input_channels:
                # Truncate if needed
                # If model expects 1 channel, use only sigma (first channel)
                if not hasattr(self, '_truncation_warned'):
                    if self.fno_expected_input_channels == 1:
                        print(f"Note: FNO model expects 1 input channel, using sigma only (g_lift will be ignored)")
                    else:
                        print(f"Warning: Truncating FNO input from {current_channels} to {self.fno_expected_input_channels} channels")
                    self._truncation_warned = True
                x_in = x_in[..., :self.fno_expected_input_channels]
        
        return x_in
    
    def image_to_conductivity(self, decoded_img):
        conductivity = decoded_img[:,0:1,:,:] 
        conductivity = torch.clamp(conductivity, -1.0, 1.0)
        conductivity = (conductivity + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        min_value = 0.01
        max_value = 1.0
        conductivity = min_value + conductivity * (max_value - min_value)  # [0,1] -> [0.1,100]
        
        conductivity = torch.clamp(conductivity, min_value, max_value)
        conductivity = F.interpolate(conductivity, size=(256, 256), mode='bilinear', align_corners=False)
        conductivity = conductivity.squeeze(1) 
        
        return conductivity
    
    def compute_measurements(self, u_target):
        u = torch.as_tensor(u_target, dtype=torch.float32)
        u_np = u.detach().cpu().numpy()
        
        if u_np.ndim == 4:  # [batch, height, width, channels]
            u_np = u_np[0, :, :, 0]  
        elif u_np.ndim == 3:  # [batch, height, width] 或 [height, width, channels]
            if u_np.shape[-1] == 1:  # [height, width, 1]
                u_np = u_np[:, :, 0]
            else:  # [batch, height, width]
                u_np = u_np[0, :, :]
        elif u_np.ndim == 2:  # [height, width]
            pass
        else:
            raise ValueError(f"Unexpected u_target dimension: {u_np.shape}")
            
        measurements = []
        measurements.extend(u_np[0, :])    
        measurements.extend(u_np[-1, :])   
        measurements.extend(u_np[:, 0])    
        measurements.extend(u_np[:, -1])  
        measurements = np.array(measurements)
            
        return measurements
    
    def compute_measurements_differentiable(self, u_target):
        u = u_target
        
        if u.ndim == 4:  # [batch, height, width, channels]
            u = u[0, :, :, 0] 
        elif u.ndim == 3:  # [batch, height, width] 或 [height, width, channels]
            if u.shape[-1] == 1:  # [height, width, 1]
                u = u[:, :, 0]
            else:  # [batch, height, width]
                u = u[0, :, :]
        elif u.ndim == 2:  # [height, width] 
            pass
        else:
            raise ValueError(f"Unexpected u_target dimension: {u.shape}")
            
        measurements = torch.cat([
            u[0, :],   
            u[-1, :],   
            u[:, 0],   
            u[:, -1]     
        ])
            
        return measurements       

    def tensor_to_fenics_function(self, tensor, V, N_grid):
        class TensorExpression(UserExpression):
            def __init__(self, tensor_data, **kwargs):
                super().__init__(**kwargs)
                self.tensor_data = tensor_data.detach().cpu().numpy()
                self.N_grid = N_grid
                
            def eval(self, value, x):
                i_float = (x[0] + 1) / 2 * self.N_grid
                j_float = (x[1] + 1) / 2 * self.N_grid
                
                i_low = max(0, min(int(i_float), self.N_grid - 1))
                i_high = min(i_low + 1, self.N_grid)
                j_low = max(0, min(int(j_float), self.N_grid - 1))
                j_high = min(j_low + 1, self.N_grid)
                    
                w_i = i_float - i_low
                w_j = j_float - j_low
                    
                value[0] = (self.tensor_data[j_low, i_low] * (1 - w_i) * (1 - w_j) +
                            self.tensor_data[j_low, i_high] * w_i * (1 - w_j) +
                            self.tensor_data[j_high, i_low] * (1 - w_i) * w_j +
                            self.tensor_data[j_high, i_high] * w_i * w_j)
                
            def value_shape(self):
                return ()
        
        expr = TensorExpression(tensor, degree=1)
        func = Function(V)
        func.interpolate(expr)
        return func
    
    def process_boundary_diff_direct(self, u_current, target_measurements, N_grid):

        spec = importlib.util.spec_from_file_location("generate_adjoint_data", 
                                                        os.path.join(os.path.dirname(__file__), "generate_adjoint_data.py"))
        generate_adjoint_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generate_adjoint_data)
        expr_boundary_values = generate_adjoint_data.expr_boundary_values
        solve_laplace_lift = generate_adjoint_data.solve_laplace_lift

        mesh = RectangleMesh(MPI.comm_self, Point(-1, -1), Point(1, 1), N_grid, N_grid)
        P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, P1)
        
 
        u_current_fenics = self.tensor_to_fenics_function(u_current, V, N_grid)
        

        boundary_current = expr_boundary_values(u_current_fenics, degree=1)
        

        target_measurements = np.array(target_measurements)
        n_points_per_edge = N_grid + 1

        top_boundary = target_measurements[:n_points_per_edge]        
        bottom_boundary = target_measurements[n_points_per_edge:2*n_points_per_edge]  
        left_boundary = target_measurements[2*n_points_per_edge:3*n_points_per_edge]  
        right_boundary = target_measurements[3*n_points_per_edge:4*n_points_per_edge]  
        

        class DirectBoundaryTargetExpression(UserExpression):
            def __init__(self, top_vals, bottom_vals, left_vals, right_vals, **kwargs):
                super().__init__(**kwargs)
                self.top_vals = top_vals
                self.bottom_vals = bottom_vals
                self.left_vals = left_vals
                self.right_vals = right_vals
                self.N_grid = N_grid
            
            def eval(self, value, x):

                i = int(round((x[0] + 1) / 2 * self.N_grid))
                j = int(round((x[1] + 1) / 2 * self.N_grid))
                

                if abs(x[1] + 1) < 1e-10:  # 下边界 (y = -1)
                    idx = max(0, min(i, self.N_grid))
                    value[0] = self.bottom_vals[idx]
                elif abs(x[1] - 1) < 1e-10:  # 上边界 (y = 1)  
                    idx = max(0, min(i, self.N_grid))
                    value[0] = self.top_vals[idx]
                elif abs(x[0] + 1) < 1e-10:  # 左边界 (x = -1)
                    idx = max(0, min(j, self.N_grid))
                    value[0] = self.left_vals[idx]
                elif abs(x[0] - 1) < 1e-10:  # 右边界 (x = 1)
                    idx = max(0, min(j, self.N_grid))
                    value[0] = self.right_vals[idx]
                else:
                    value[0] = 0.0  
            
            def value_shape(self):
                return ()
        
        boundary_target = DirectBoundaryTargetExpression(
            top_boundary, bottom_boundary, left_boundary, right_boundary, degree=1
        )
        
        class BoundaryDiffExpression(UserExpression):
            def __init__(self, boundary_current, boundary_target, **kwargs):
                super().__init__(**kwargs)
                self.boundary_current = boundary_current
                self.boundary_target = boundary_target
            
            def eval(self, value, x):
                val_current = np.zeros(1)
                val_target = np.zeros(1)
                self.boundary_current.eval(val_current, x)
                self.boundary_target.eval(val_target, x)
                value[0] = val_current[0] - val_target[0]
            
            def value_shape(self):
                return ()
        
        boundary_diff = BoundaryDiffExpression(boundary_current, boundary_target, degree=1)
            

        u_diff_lift_fenics = solve_laplace_lift(mesh, boundary_diff)
            

        u_diff_lift_values = u_diff_lift_fenics.compute_vertex_values().reshape((N_grid + 1, N_grid + 1))
        u_diff_lift_tensor = torch.from_numpy(u_diff_lift_values).float().to(self.device)
        
        return u_diff_lift_tensor   

    def compute_gradient_adjoint_fno_direct(self, sigma_initial, u_current, target_measurements):

        u_diff_lift_inference = self.process_boundary_diff_direct(
            u_current, target_measurements, N_grid=u_current.shape[0]-1
        )
            
        sigma_mean = self.adjoint_normalizers['sigma'].mean
        sigma_std = self.adjoint_normalizers['sigma'].std
        u_diff_lift_mean = self.adjoint_normalizers['u_diff_lift'].mean
        u_diff_lift_std = self.adjoint_normalizers['u_diff_lift'].std
        
        sigma_norm_adj = (sigma_initial - sigma_mean) / sigma_std
        u_diff_lift_norm = (u_diff_lift_inference - u_diff_lift_mean) / u_diff_lift_std
        
        x_in_adj = torch.stack([sigma_norm_adj, u_diff_lift_norm], dim=-1).unsqueeze(0)

        with torch.no_grad():
            lambda_pred_norm = self.adjoint_fno_model(x_in_adj).squeeze(0)
        
        lambda_mean = self.adjoint_normalizers['lambda'].mean
        lambda_std = self.adjoint_normalizers['lambda'].std
        lambda_pred = lambda_pred_norm * lambda_std + lambda_mean
        
        u_current_2d = u_current.detach().cpu().numpy()
        lambda_pred_2d = lambda_pred.detach().cpu().numpy()
        
        grad_u_x, grad_u_y = self.compute_gradient_fd(u_current_2d, dx=2.0/u_current_2d.shape[0])
        grad_lambda_x, grad_lambda_y = self.compute_gradient_fd(lambda_pred_2d, dx=2.0/lambda_pred_2d.shape[0])
        
        gradient_adjoint_fno_np = -(grad_u_x * grad_lambda_x + grad_u_y * grad_lambda_y)
        gradient_adjoint_fno = torch.from_numpy(gradient_adjoint_fno_np).float().to(self.device)
        
        return gradient_adjoint_fno, lambda_pred

    def compute_gradient_fd(self, u, dx):

        grad_u_x = np.zeros_like(u)
        grad_u_y = np.zeros_like(u)
        
        if u.shape[0] <= 1 or u.shape[1] <= 1:
            print(f"警告：数组大小 {u.shape} 太小，无法计算梯度，返回零数组")
            return grad_u_x, grad_u_y
        
        if u.shape[0] > 2:
            grad_u_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
        if u.shape[1] > 2:
            grad_u_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
        
        if u.shape[0] > 1:
            grad_u_x[0, :] = (u[1, :] - u[0, :]) / dx
            grad_u_x[-1, :] = (u[-1, :] - u[-2, :]) / dx
        if u.shape[1] > 1:
            grad_u_y[:, 0] = (u[:, 1] - u[:, 0]) / dx
            grad_u_y[:, -1] = (u[:, -1] - u[:, -2]) / dx
        
        return grad_u_x, grad_u_y
    
    def reconstruct_corrected(self, boundary_conditions=None, 
                                max_iterations=2000, learning_rate=5e-3,
                                save_interval=1, output_dir=None, gradient_method='adjoint_fno',
                                ablation_mode='dilo'):  
        print(f"Starting CORRECTED EIT reconstruction with Latent Diffusion + {gradient_method.upper()} method...")
        print(f"Ablation Mode: {ablation_mode.upper()}") 
    
        print(f"Starting CORRECTED EIT reconstruction with Latent Diffusion + {gradient_method.upper()} method...")
        print(f"Max iterations: {max_iterations}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gradient method: {gradient_method}")
        
        
        latent_shape = (1, 3, 64, 64)  
        noise_target = torch.randn(latent_shape, device=self.device)
        
        with torch.no_grad():
            decoded_z_target = self.sampler.ddecode(
                noise_target,
                t_start=100,
                temp=0,
                unconditional_guidance_scale=1.0
            )
            decoded_img_target = self.diffusion_model.differentiable_decode_first_stage(decoded_z_target)
            sigma_target = self.image_to_conductivity(decoded_img_target)
            g_lift_target = boundary_conditions.clone()
            
            sigma_target_norm = self.normalizers['sigma'].encode(sigma_target)
            g_lift_target_batch = g_lift_target.unsqueeze(0)  
            g_lift_target_norm = self.normalizers['g_lift'].encode(g_lift_target_batch)
            
            if sigma_target_norm.dim() == 2:
                sigma_target_norm = sigma_target_norm.unsqueeze(0)
            if g_lift_target_norm.dim() == 2:
                g_lift_target_norm = g_lift_target_norm.unsqueeze(0)
            x_in_target = self.prepare_fno_input(sigma_target_norm, g_lift_target_norm)
            
            u_target = self.fno_model(x_in_target)
            
        target_measurements = self.compute_measurements(u_target)
        latent_shape = (1, 3, 64, 64)
        

        if ablation_mode in ['dilo', 'stochastic']:

            opt_t_start = 100
            opt_temp = 0.0 if ablation_mode == 'dilo' else 1.0
            

            noise_latent = torch.randn(latent_shape, device=self.device, requires_grad=True)
            optimizer = torch.optim.AdamW([noise_latent], lr=learning_rate)
            losses = []
            
            print(f"Starting Optimization Loop (Mode: {ablation_mode.upper()}, t_start={opt_t_start}, temp={opt_temp})")
            
            for iteration in range(max_iterations):
                optimizer.zero_grad()
                

                if ablation_mode == 'stochastic':
                    torch.manual_seed(42)
                

                decoded_z = self.sampler.ddecode(
                    noise_latent,
                    t_start=opt_t_start,
                    temp=opt_temp,
                    unconditional_guidance_scale=1.0
                )
                        
                # 2. 通过VAE decoder得到图像
                decoded_img = self.diffusion_model.differentiable_decode_first_stage(decoded_z)
                sigma_input = self.image_to_conductivity(decoded_img)
                sigma_norm = self.normalizers['sigma'].encode(sigma_input)
                if sigma_norm.dim() == 2: sigma_norm = sigma_norm.unsqueeze(0)
                x_in = self.prepare_fno_input(sigma_norm, g_lift_target_norm)
                    
                # 3. FNO前向预测
                u_pred = self.fno_model(x_in)
                    
                # 4. 计算数据损失
                if gradient_method == 'autodiff':
                    current_measurements_tensor = self.compute_measurements_differentiable(u_pred)
                else:
                    current_measurements = self.compute_measurements(u_pred)
                    current_measurements_tensor = torch.tensor(current_measurements, dtype=torch.float32, device=self.device)
                
                target_measurements_tensor = torch.tensor(target_measurements, dtype=torch.float32, device=self.device)
                total_loss = F.mse_loss(current_measurements_tensor, target_measurements_tensor)
                
                # 5. 梯度反传
                if gradient_method == 'adjoint_fno':
                    u_current = u_pred.squeeze().detach()
                    grad_J_sigma, lambda_pred = self.compute_gradient_adjoint_fno_direct(
                        sigma_input.squeeze(0), u_current, target_measurements
                    )
                    grad_norm = torch.norm(grad_J_sigma)
                    if grad_norm > 1e-8:
                        loss_scale = total_loss.item() / grad_norm.item()
                        grad_J_sigma_scaled = grad_J_sigma * loss_scale
                    else:
                        grad_J_sigma_scaled = grad_J_sigma
                                
                    virtual_loss = torch.sum(sigma_input.squeeze() * grad_J_sigma_scaled)
                    virtual_loss.backward()
                elif gradient_method == 'autodiff':
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_([noise_latent], max_norm=1.0)
                        
                optimizer.step()
                        
                loss_val = total_loss.detach().cpu().numpy()
                losses.append(loss_val)
                        
                if iteration % 10 == 0:
                    print(f"Iter {iteration:4d}: Total Loss = {loss_val:.6f}")
                    plot_real_time_loss(losses, output_dir, iteration)
                    
                    intermediate_dir = os.path.join(output_dir, "intermediate_results")
                    voltage_pred_save = current_measurements_tensor.detach() if gradient_method == 'autodiff' else current_measurements_tensor
                        
                    save_intermediate_results(
                        iteration, sigma_input.squeeze(0), loss_val, intermediate_dir,
                        voltage_pred=voltage_pred_save, voltage_true=target_measurements_tensor, ground_truth=sigma_target.squeeze(0)
                    )

        elif ablation_mode == 'ddis':
            # ---------------------------------------------------------
            # 模式 3: 真正的 DDIS/DPS 逐步引导采样法 (Sampling-based Baseline)
            # ---------------------------------------------------------
            print(f"Starting Step-wise Guidance Sampling (Mode: True DDIS/DPS Baseline)")
            
            # 初始化 z_T
            z_t = torch.randn(latent_shape, device=self.device)
            
            # =========================================================
            # 1. 参照 ddecode 的逻辑设置时间步
            # =========================================================
            opt_t_start = 100
            timesteps = np.arange(self.sampler.ddpm_num_timesteps) if False else self.sampler.ddim_timesteps
            timesteps = timesteps[:opt_t_start]
            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]
            
            losses = []
            dps_scale = 0.05  # DPS自适应缩放步长，保持 z_t 不崩溃
            
            # 直接遍历时间步
            for i, step in enumerate(time_range):
                index = total_steps - i - 1
                ts = torch.full((1,), step, device=self.device, dtype=torch.long)
                
                # 开启对当前隐变量 z_t 的梯度追踪
                z_t = z_t.detach().requires_grad_(True)
                
                # =========================================================
                # 2. 参照 ddecode 调用底层单步采样
                # =========================================================
                p_sample_result = self.sampler.p_sample_ddim(
                    z_t,
                    c=None,
                    t=ts,
                    index=index,
                    use_original_steps=False,
                    temperature=0.0, # 保持 ODE 轨迹，温度为 0
                    unconditional_guidance_scale=1.0,
                    unconditional_conditioning=None
                )
                
                # 解析返回值：z_prev 对应下一步状态, pred_z0 对应 Tweedie 估计的干净隐状态 \hat{z}_0
                if isinstance(p_sample_result, tuple):
                    z_prev = p_sample_result[0]
                    pred_z0 = p_sample_result[1]
                else:
                    z_prev = p_sample_result
                    pred_z0 = p_sample_result # fallback，理论上代码肯定会返回 tuple
                
                # =========================================================
                # 3. Latent Decode -> 物理算子引导 (核心解耦部分)
                # =========================================================
                # 用 Decoder 将 \hat{z}_0 解码回物理空间图像
                decoded_img = self.diffusion_model.differentiable_decode_first_stage(pred_z0)
                
                sigma_input = self.image_to_conductivity(decoded_img)
                sigma_norm = self.normalizers['sigma'].encode(sigma_input)
                if sigma_norm.dim() == 2: sigma_norm = sigma_norm.unsqueeze(0)
                
                x_in = self.prepare_fno_input(sigma_norm, g_lift_target_norm)
                u_pred = self.fno_model(x_in)
                
                # 算物理 Loss
                if gradient_method == 'autodiff':
                    current_measurements_tensor = self.compute_measurements_differentiable(u_pred)
                else:
                    current_measurements = self.compute_measurements(u_pred)
                    current_measurements_tensor = torch.tensor(current_measurements, dtype=torch.float32, device=self.device)
                
                target_measurements_tensor = torch.tensor(target_measurements, dtype=torch.float32, device=self.device)
                total_loss = F.mse_loss(current_measurements_tensor, target_measurements_tensor)
                
                # 计算引导梯度
                if gradient_method == 'adjoint_fno':
                    u_current = u_pred.squeeze().detach()
                    grad_J_sigma, _ = self.compute_gradient_adjoint_fno_direct(
                        sigma_input.squeeze(0), u_current, target_measurements
                    )
                    virtual_loss = torch.sum(sigma_input.squeeze() * grad_J_sigma)
                    grad_z = torch.autograd.grad(virtual_loss, z_t)[0]
                else:
                    grad_z = torch.autograd.grad(total_loss, z_t)[0]
                
                # =========================================================
                # 4. 执行引导采样更新 (引入 DPS 自适应缩放)
                # =========================================================
                norm_grad = torch.norm(grad_z)
                norm_loss = torch.norm(current_measurements_tensor - target_measurements_tensor)
                
                if norm_grad > 1e-8:
                    adjusted_grad = grad_z * (dps_scale * norm_loss / norm_grad)
                else:
                    adjusted_grad = torch.zeros_like(grad_z)
                
                # 用单步去噪的底子 z_prev 减去修正后的物理梯度，完成隐空间采样更新
                z_t = z_prev.detach() - adjusted_grad
                
                # --- 日志记录 ---
                loss_val = total_loss.detach().cpu().numpy()
                losses.append(loss_val)
                
                if i % 10 == 0 or i == total_steps - 1:
                    print(f"Sampling Step {i:4d}/{total_steps} (t={step}): DPS Guidance Loss = {loss_val:.6f} | Grad Norm = {norm_grad:.6f}")
                    plot_real_time_loss(losses, output_dir, i)
                    
                    intermediate_dir = os.path.join(output_dir, "intermediate_results")
                    voltage_pred_save = current_measurements_tensor.detach() if gradient_method == 'autodiff' else current_measurements_tensor
                    save_intermediate_results(
                        i, sigma_input.squeeze(0).detach(), loss_val, intermediate_dir,
                        voltage_pred=voltage_pred_save, voltage_true=target_measurements_tensor, ground_truth=sigma_target.squeeze(0)
                    )
                    
        return


def prepare_sample(reader, idx, device):
    """从reader中准备指定索引的样本数据。"""
    sigma = reader.read_field('sigma')[idx]
    g_lift = reader.read_field('g_lift')[idx] 
    u = reader.read_field('u')[idx]
    return sigma, g_lift, u


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='EIT重建使用扩散模型')
    
    # 数据参数 - 参考06_compare_gradients.py的配置
    parser.add_argument('--data_path', type=str, 
                       default='data/test_data.npy',
                       help='测试数据路径')
    
    # 模型参数
    parser.add_argument('--mesh_size', type=int, default=256, help='网格大小')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 重建参数
    parser.add_argument('--max_iterations', type=int, default=1000, help='最大迭代次数')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='学习率')
    parser.add_argument('--gradient_method', type=str, default='autodiff', 
                       choices=['adjoint_fno', 'autodiff'], 
                       help='梯度计算方法: adjoint_fno (使用伴随FNO) 或 autodiff (直接自动微分)')
    # === 新增消融实验参数 ===
    parser.add_argument('--ablation_mode', type=str, default='dilo', 
                       choices=['dilo', 'stochastic', 'ddis'],  # <--- 替换这里
                       help='消融实验模式: dilo(我们的), stochastic(验证梯度方差), ddis(DPS逐步引导法)')
    
    # 模型路径参数
    parser.add_argument('--checkpoint_path', type=str, 
                       default='work_dir/diffusion/checkpoints/checkpoint_60000.pth',
                       help='扩散模型检查点路径')
    parser.add_argument('--fno_model_path', type=str, 
                       default='work_dir/fno/best_model.pth',
                       help='FNO模型路径')
    parser.add_argument('--adjoint_fno_path', type=str, 
                       default='work_dir/adjoint_fno/20250903_151032/checkpoints/best_adjoint_fno_model.pth',
                       help='伴随FNO模型路径')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建统一的时间戳目录（在开始时就创建）
    base_output_dir = "logs"
    os.makedirs(base_output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"eit_reconstruction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 保存运行参数到JSON文件
    config_params = {
        "timestamp": timestamp,
        "output_dir": output_dir,
        "data_params": {
            "data_path": args.data_path,
        },
        "model_params": {
            "mesh_size": args.mesh_size,
            "device": args.device,
            "seed": args.seed,
        },
        "reconstruction_params": {
            "max_iterations": args.max_iterations,
            "learning_rate": args.learning_rate,
            "gradient_method": args.gradient_method,
            "ablation_mode": args.ablation_mode,  # <--- 保存进 config
            "save_interval": 10,
        },
        "model_paths": {
            "checkpoint_path": args.checkpoint_path,
            "fno_model_path": args.fno_model_path,
            "adjoint_fno_path": args.adjoint_fno_path,
        },
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "pytorch_version": torch.__version__,
        }
    }
    
    config_file = os.path.join(output_dir, "run_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_params, f, indent=4)
    print(f"Configuration saved to: {config_file}")
    
    # 初始化重建器
    print("初始化EIT重建器...")
    reconstructor = EITLatentDiffusionReconstructorCorrected(
        mesh_size=args.mesh_size,
        device=device,
        checkpoint_path=args.checkpoint_path,  # 传递checkpoint路径让重建器自己初始化
        fno_model_path=args.fno_model_path,  # 添加FNO模型路径
        adjoint_fno_model_path=args.adjoint_fno_path,
        num_electrodes=8,
        gradient_method=args.gradient_method
    )
    
    # 创建数据读取器来获取边界条件
    reader = NpyReader(args.data_path, to_cuda=(device.type == 'cuda'))
    
    sigma, g_lift, u = prepare_sample(reader, 0, device)
    boundary_conditions = g_lift
    
    print("\n" + "="*80)
    print("Starting EIT Reconstruction Test")
    print("="*80)
    
    reconstructor.reconstruct_corrected(
        boundary_conditions=boundary_conditions,
        max_iterations=args.max_iterations,
        learning_rate=args.learning_rate,
        save_interval=10,  
        output_dir=output_dir,  
        gradient_method=args.gradient_method,
        ablation_mode=args.ablation_mode 
    )


if __name__ == "__main__":
    main()