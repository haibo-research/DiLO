"""
Diffusion model helper functions module
Contains utility functions for loading and managing diffusion models
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """
    Load model from configuration file and checkpoint
    
    Args:
        config: Model configuration
        ckpt: Checkpoint path
        device: Device (cpu/cuda)
        verbose: Whether to print detailed information
    
    Returns:
        Loaded model
    """
    print(f"Loading model from {ckpt}")
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
    else:
        sd = None

    model = instantiate_from_config(config.model)
    
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) > 0 and verbose:
            print("missing keys:")
            print(missing)
        if len(unexpected) > 0 and verbose:
            print("unexpected keys:")
            print(unexpected)

    model.to(device)
    model.eval()
    return model


def load_diffusion_model(config_path, checkpoint_path, device="cuda"):
    """
    Load diffusion model
    
    Args:
        config_path: Configuration file path
        checkpoint_path: Checkpoint path
        device: Device
    
    Returns:
        tuple: (model, sampler)
    """
    config = OmegaConf.load(config_path)
    model = load_model_from_config(config, checkpoint_path, device=device)
    
    sampler = DDIMSampler(model)
    
    return model, sampler


def sample_from_model(model, sampler, shape, conditioning=None, 
                     ddim_steps=50, ddim_eta=0.0, scale=1.0):
    """
    从模型采样
    
    Args:
        model: 扩散模型
        sampler: DDIM采样器
        shape: 采样形状
        conditioning: 条件信息
        ddim_steps: DDIM步数
        ddim_eta: DDIM eta参数
        scale: 缩放因子
    
    Returns:
        采样结果
    """
    with torch.no_grad():
        samples, _ = sampler.sample(
            S=ddim_steps,
            conditioning=conditioning,
            batch_size=shape[0],
            shape=shape[1:],
            verbose=False,
            eta=ddim_eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=None
        )
    return samples


def encode_to_latent(model, x):
    """
    将输入编码到潜在空间
    
    Args:
        model: 扩散模型
        x: 输入数据
    
    Returns:
        潜在编码
    """
    with torch.no_grad():
        posterior = model.encode_first_stage(x)
        z = model.get_first_stage_encoding(posterior).detach()
    return z


def decode_from_latent(model, z):
    """
    从潜在空间解码
    
    Args:
        model: 扩散模型
        z: 潜在编码
    
    Returns:
        解码结果
    """
    with torch.no_grad():
        x_hat = model.decode_first_stage(z)
    return x_hat


def prepare_conditioning(model, conditioning_data):
    """
    准备条件信息
    
    Args:
        model: 扩散模型
        conditioning_data: 条件数据
    
    Returns:
        处理后的条件信息
    """
    if conditioning_data is None:
        return None
    
    with torch.no_grad():
        if hasattr(model, 'get_learned_conditioning'):
            c = model.get_learned_conditioning(conditioning_data)
        else:
            c = conditioning_data
    return c


def normalize_tensor(x, mode='0to1'):
    """
    标准化张量
    
    Args:
        x: 输入张量
        mode: 标准化模式 ('0to1', '-1to1', 'zscore')
    
    Returns:
        标准化后的张量
    """
    if mode == '0to1':
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-8)
    elif mode == '-1to1':
        x_min = x.min()
        x_max = x.max()
        return 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
    elif mode == 'zscore':
        return (x - x.mean()) / (x.std() + 1e-8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def denormalize_tensor(x, x_min, x_max, mode='0to1'):
    """
    反标准化张量
    
    Args:
        x: 标准化的张量
        x_min: 原始最小值
        x_max: 原始最大值
        mode: 标准化模式
    
    Returns:
        反标准化后的张量
    """
    if mode == '0to1':
        return x * (x_max - x_min) + x_min
    elif mode == '-1to1':
        return (x + 1) / 2 * (x_max - x_min) + x_min
    else:
        raise ValueError(f"Unknown denormalization mode: {mode}")


def setup_diffusion_paths(project_root):
    """
    设置扩散模型相关路径
    
    Args:
        project_root: 项目根目录
    
    Returns:
        dict: 包含各种路径的字典
    """
    paths = {
        'taming_transformers': os.path.join(project_root, "taming-transformers"),
        'resample_scripts': os.path.join(project_root, "resample/scripts"),
        'configs': os.path.join(project_root, "configs"),
        'checkpoints': os.path.join(project_root, "checkpoints"),
        'logs': os.path.join(project_root, "logs")
    }
    
    # 添加到系统路径
    for path in [paths['taming_transformers'], paths['resample_scripts']]:
        if path not in sys.path and os.path.exists(path):
            sys.path.append(path)
    
    return paths


def check_diffusion_dependencies():
    """
    检查扩散模型依赖是否可用
    
    Returns:
        bool: 依赖是否可用
    """
    try:
        from taming.models import vqgan
        from ldm.models.diffusion.ddim import DDIMSampler
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        return True
    except ImportError as e:
        print(f"Diffusion model dependencies not available: {e}")
        return False


def create_sampling_config(ddim_steps=50, ddim_eta=0.0, scale=1.0):
    """
    创建采样配置
    
    Args:
        ddim_steps: DDIM步数
        ddim_eta: DDIM eta参数
        scale: 无条件引导缩放
    
    Returns:
        采样配置字典
    """
    return {
        'ddim_steps': ddim_steps,
        'ddim_eta': ddim_eta,
        'scale': scale
    }


def batch_process_latents(model, latents, batch_size=4):
    """
    批量处理潜在编码
    
    Args:
        model: 扩散模型
        latents: 潜在编码张量
        batch_size: 批次大小
    
    Returns:
        处理后的结果
    """
    results = []
    n_samples = latents.shape[0]
    
    for i in range(0, n_samples, batch_size):
        batch_latents = latents[i:i+batch_size]
        with torch.no_grad():
            batch_results = decode_from_latent(model, batch_latents)
        results.append(batch_results)
    
    return torch.cat(results, dim=0)


def save_latent_codes(latents, save_path, metadata=None):
    """
    保存潜在编码
    
    Args:
        latents: 潜在编码张量
        save_path: 保存路径
        metadata: 元数据
    """
    save_dict = {
        'latents': latents.cpu().numpy(),
        'shape': latents.shape
    }
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    np.save(save_path, save_dict)


def load_latent_codes(load_path, device='cuda'):
    """
    加载潜在编码
    
    Args:
        load_path: 加载路径
        device: 设备
    
    Returns:
        tuple: (latents, metadata)
    """
    data = np.load(load_path, allow_pickle=True).item()
    latents = torch.from_numpy(data['latents']).to(device)
    metadata = data.get('metadata', None)
    
    return latents, metadata 