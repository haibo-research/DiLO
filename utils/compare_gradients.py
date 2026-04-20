import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
import argparse
import random
import time # 修正: 导入标准的 time 模块
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from dotmap import DotMap

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入项目模块
# from diffusion_ip.reconstruction import load_fno_model
import importlib.util
import sys
import os

# 动态导入utils模块
utils_spec = importlib.util.spec_from_file_location("load", os.path.join(os.path.dirname(__file__), "utils/load.py"))
load_module = importlib.util.module_from_spec(utils_spec)
utils_spec.loader.exec_module(load_module)

loss_spec = importlib.util.spec_from_file_location("Loss", os.path.join(os.path.dirname(__file__), "utils/Loss.py"))
loss_module = importlib.util.module_from_spec(loss_spec)
loss_spec.loader.exec_module(loss_module)

# 动态导入FNO模型
fno_spec = importlib.util.spec_from_file_location("FNO", os.path.join(os.path.dirname(__file__), "model/FNO.py"))
fno_module = importlib.util.module_from_spec(fno_spec)
fno_spec.loader.exec_module(fno_module)

# 获取需要的类和函数
NpyReader = load_module.NpyReader
GaussianNormalizer = load_module.GaussianNormalizer
LpLoss = loss_module.LpLoss
FNO2d = fno_module.FNO2d


def load_fno_model(model_path, model_params, device):
    """加载FNO模型。
    
    Args:
        model_path: 模型权重文件路径
        model_params: 模型参数字典
        device: 计算设备
    
    Returns:
        加载完成的FNO模型
    """
    # 使用与03_verify_fno_accuracy.py完全相同的参数和加载方式
    model = FNO2d(
        modes1=model_params.get('modes1', 12),
        modes2=model_params.get('modes2', 12),
        width=model_params.get('width', 32),
        in_channels=model_params.get('in_channels', 2)
    ).to(device)
    
    try:
        # 显式设置weights_only=False以兼容新版PyTorch
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
        print(f"FNO模型已从 {model_path} 成功加载")
        return model
    except Exception as e:
        print(f"加载FNO模型时出错: {e}")
        print(f"模型路径: {model_path}")
        print(f"模型参数: {model_params}")
        raise

# 导入 FEniCS
try:
    from dolfin import *
    # 从04_generate_adjoint_data.py导入所需的函数
    # 由于Python模块名不能以数字开头，我们需要使用importlib导入
    import importlib.util
    import sys
    import os

    # 动态导入04_generate_adjoint_data.py
    spec = importlib.util.spec_from_file_location("generate_adjoint_data", 
                                                  os.path.join(os.path.dirname(__file__), "generate_adjoint_data.py"))
    generate_adjoint_data = importlib.util.module_from_spec(spec)
    sys.modules["generate_adjoint_data"] = generate_adjoint_data
    spec.loader.exec_module(generate_adjoint_data)

    # 导入需要的函数
    expr_boundary_values = generate_adjoint_data.expr_boundary_values
    solve_laplace_lift = generate_adjoint_data.solve_laplace_lift
except ImportError:
    print("FEniCS/dolfin is not installed. FEM solver will not be available.")
    dolfin_installed = False
else:
    # 设置 FEniCS 日志级别，减少不必要的输出
    set_log_level(LogLevel.WARNING)
    dolfin_installed = True

# 导入伴随FNO相关模块 - 直接从05_train_adjoint_fno.py导入
try:
    import importlib.util
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    adjoint_fno_path = os.path.join(current_dir, "train_adjoint_fno.py")
    spec = importlib.util.spec_from_file_location("adjoint_fno_module", adjoint_fno_path)
    adjoint_fno_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adjoint_fno_module)
    AdjointFNO2d = adjoint_fno_module.FNO2d
    adjoint_fno_available = True
    print("成功导入AdjointFNO2d from 05_train_adjoint_fno.py")
except ImportError:
    print("Warning: AdjointFNO2d not available. Will try to use regular FNO2d for adjoint model.")
    adjoint_fno_available = False

# --- FEniCS 辅助类 ---
class NumpyUserExpression(UserExpression):
    """一个通用的 FEniCS UserExpression，用于从 Numpy 数组中插值。"""
    def __init__(self, np_array, **kwargs):
        self.np_array = np_array
        self.H, self.W = np_array.shape
        super().__init__(**kwargs)
    
    def eval(self, value, x):
        i = int(round((x[0] + 1) / 2 * (self.W - 1)))
        j = int(round((x[1] + 1) / 2 * (self.H - 1)))
        i = max(0, min(i, self.W - 1))
        j = max(0, min(j, self.H - 1))
        value[0] = float(self.np_array[j, i])
        
    def value_shape(self):
        return ()

# --- 核心梯度计算函数 ---

def compute_gradient_fem_adjoint(sigma, u_target, g, mesh_size):
    """
    使用有限元伴随法计算梯度 (Ground Truth)。
    此版本使用 UserExpression 和 interpolate，以精确匹配数据生成过程。
    """
    if not dolfin_installed:
        raise RuntimeError("FEniCS is not installed.")
    
    print("Computing gradient with FEM Adjoint method (Ground Truth)...")
    start_time = time.time()
    
    sigma_np = sigma.squeeze().detach().cpu().numpy()
    u_target_np = u_target.squeeze().detach().cpu().numpy()
    g_np = g.squeeze().detach().cpu().numpy()

    # 1. 设置网格和函数空间
    mesh = RectangleMesh(Point(-1, -1), Point(1, 1), mesh_size - 1, mesh_size - 1)
    P1 = FiniteElement('CG', mesh.ufl_cell(), 1)
    R = FiniteElement('Real', mesh.ufl_cell(), 0)
    W = FunctionSpace(mesh, P1 * R)
    V = FunctionSpace(mesh, P1)

    # 2. 创建 FEniCS 函数
    sigma_expr = NumpyUserExpression(sigma_np, degree=1)
    sigma_func = Function(V)
    sigma_func.interpolate(sigma_expr)

    g_expr = NumpyUserExpression(g_np, degree=1)
    g_func = Function(V)
    g_func.interpolate(g_expr)

    # 3. 正向求解 u - 与01_generate_fno_data.py完全一致
    (u, c) = TrialFunctions(W)
    (v, d) = TestFunctions(W)
    a_fwd = (inner(sigma_func*grad(u), grad(v))) * dx + c * v * dx + u * d * dx
    L_fwd = g_func * v * ds
    w_fwd = Function(W)
    solve(a_fwd == L_fwd, w_fwd)
    u_sol, _ = w_fwd.split(deepcopy=True)

    # 4. 伴随求解 p - 与04_generate_adjoint_data.py完全一致
    u_target_expr = NumpyUserExpression(u_target_np, degree=1)
    u_target_func = Function(V)
    u_target_func.interpolate(u_target_expr)
    
    # 创建边界差值函数：(u_sol - u_target)|_∂Ω，与训练数据生成保持一致
    # 注意：训练数据中使用的是 (u_current - u_target)|_∂Ω
    boundary_diff = u_sol - u_target_func
    
    (p, c_adj) = TrialFunctions(W)
    (q, d_adj) = TestFunctions(W)
    # adjoint 方程的变分形式：∇·(σ∇p) = 0 in Ω, σ(∂p/∂n) = (u_target - u_sol)|_∂Ω on ∂Ω
    a_adj = (inner(sigma_func*grad(p), grad(q)) + c_adj*q + p*d_adj)*dx
    L_adj = boundary_diff*q*ds
    w_adj = Function(W)
    solve(a_adj == L_adj, w_adj)
    p_sol, _ = w_adj.split(deepcopy=True)

    # 5. 计算梯度
    grad_fem_func = project(-dot(grad(u_sol), grad(p_sol)), V)
    
    # 6. 将结果转换回 Numpy
    u_np = u_sol.compute_vertex_values(mesh).reshape(mesh_size, mesh_size)
    p_np = p_sol.compute_vertex_values(mesh).reshape(mesh_size, mesh_size)
    grad_np = grad_fem_func.compute_vertex_values(mesh).reshape(mesh_size, mesh_size)
    
    # 诊断：检查FEM伴随解的统计信息
    print(f"FEM adjoint solution statistics - mean: {np.mean(p_np):.6f}, std: {np.std(p_np):.6f}")

    elapsed_time = time.time() - start_time
    print(f"FEM Adjoint solve time: {elapsed_time:.4f}s")
    
    return grad_np, u_np, p_np, elapsed_time


def compute_gradient_fno_autodiff(model, normalizers, sigma_initial, g_lift_initial, u_target, device, apply_zero_mean=False):
    """
    使用FNO模型和自动微分计算梯度
    
    Args:
        model: 训练好的FNO模型
        normalizers: 归一化器字典
        sigma_initial: 初始电导率
        g_lift_initial: 初始边界提升
        u_target: 目标解
        device: 计算设备
        apply_zero_mean: 是否应用零均值约束（已废弃，保持为False）
    
    Returns:
        grad_fno_ad: 梯度
        u_pred: 预测解
        elapsed_time: 计算时间
    """
    print("Computing gradient with FNO Autodiff method...")
    start_time = time.time()

    s = sigma_initial.shape[-1]
    sigma_initial.requires_grad_(True)

    # 归一化输入
    sigma_norm = normalizers['sigma'].encode(sigma_initial.unsqueeze(0))
    glift_norm = normalizers['g_lift'].encode(g_lift_initial.unsqueeze(0))
    x_in = torch.stack([sigma_norm.squeeze(0), glift_norm.squeeze(0)], dim=-1).unsqueeze(0)

    # FNO 正向预测 - 输出归一化后的数据
    u_pred_output = model(x_in)
    s = sigma_initial.shape[-1]
    u_pred = u_pred_output.reshape(u_pred_output.shape[0], s, s).squeeze()
    
    # 计算损失（不需要任何后处理，因为模型已经输出正确尺度的数据）
    loss = 0.5 * torch.sum((u_pred - u_target) ** 2)
    
    # 反向传播计算梯度
    loss.backward()
    grad_fno_ad = sigma_initial.grad.detach().cpu().numpy()
    
    elapsed_time = time.time() - start_time
    print(f"FNO Autodiff solve time: {elapsed_time:.4f}s")

    return grad_fno_ad, u_pred.detach().cpu().numpy(), elapsed_time


def compute_gradient_fd(u, dx):
    """
    使用有限差分计算梯度
    
    Args:
        u: 输入场 (numpy array)
        dx: 网格间距
    
    Returns:
        grad_u_x, grad_u_y: x和y方向的梯度
    """
    grad_u_x = np.zeros_like(u)
    grad_u_y = np.zeros_like(u)
    
    # 检查数组大小，如果某个维度只有1个元素，则无法计算梯度
    if u.shape[0] <= 1 or u.shape[1] <= 1:
        print(f"警告：数组大小 {u.shape} 太小，无法计算梯度，返回零数组")
        return grad_u_x, grad_u_y
    
    # 中心差分
    if u.shape[0] > 2:
        grad_u_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2 * dx)
    if u.shape[1] > 2:
        grad_u_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * dx)
    
    # 边界处使用前向/后向差分
    if u.shape[0] > 1:
        grad_u_x[0, :] = (u[1, :] - u[0, :]) / dx
        grad_u_x[-1, :] = (u[-1, :] - u[-2, :]) / dx
    if u.shape[1] > 1:
        grad_u_y[:, 0] = (u[:, 1] - u[:, 0]) / dx
        grad_u_y[:, -1] = (u[:, -1] - u[:, -2]) / dx
    
    return grad_u_x, grad_u_y


def process_boundary_diff_for_adjoint_inference(u_current, u_target, N_grid, device='cpu'):
    """
    为伴随FNO推理处理边界差值：从解差值中提取边界值并进行拉普拉斯提升
    使用04_generate_adjoint_data.py中的函数来确保与训练数据生成一致
    
    Args:
        u_current: 当前解 (torch tensor)
        u_target: 目标解 (torch tensor)
        N_grid: 网格大小
        device: 设备 (cpu或cuda)
    
    Returns:
        u_diff_lift_inference: 边界差值的拉普拉斯提升 (torch tensor)
    """
    # 创建FEniCS网格和函数空间
    mesh = RectangleMesh(MPI.comm_self, Point(-1, -1), Point(1, 1), N_grid, N_grid)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)
    
    # 将PyTorch张量转换为FEniCS函数
    u_current_fenics = tensor_to_fenics_function(u_current, V, N_grid)
    u_target_fenics = tensor_to_fenics_function(u_target, V, N_grid)
    
    # 使用04_generate_adjoint_data.py中的方法提取边界值
    boundary_current = expr_boundary_values(u_current_fenics, degree=1)
    boundary_target = expr_boundary_values(u_target_fenics, degree=1)

    # 创建边界差值表达式（与训练数据生成完全一致）
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
        
    # 使用04_generate_adjoint_data.py中的拉普拉斯提升函数
    u_diff_lift_fenics = solve_laplace_lift(mesh, boundary_diff)
        
    # 将FEniCS函数转换回PyTorch张量，并确保在正确的设备上
    u_diff_lift_values = u_diff_lift_fenics.compute_vertex_values().reshape((N_grid + 1, N_grid + 1))
    u_diff_lift_tensor = torch.from_numpy(u_diff_lift_values).float().to(device)
    
    return u_diff_lift_tensor


def tensor_to_fenics_function(tensor, V, N_grid):
    """将PyTorch张量转换为FEniCS函数"""
    # 创建一个自定义的UserExpression来从张量中插值
    class TensorExpression(UserExpression):
        def __init__(self, tensor_data, **kwargs):
            super().__init__(**kwargs)
            self.tensor_data = tensor_data.detach().cpu().numpy()
            self.N_grid = N_grid
        
        def eval(self, value, x):
            # 将[-1,1]的坐标映射到[0,N_grid]，使用更精确的插值
            i_float = (x[0] + 1) / 2 * self.N_grid
            j_float = (x[1] + 1) / 2 * self.N_grid
            
            # 双线性插值
            i_low = max(0, min(int(i_float), self.N_grid - 1))
            i_high = min(i_low + 1, self.N_grid)
            j_low = max(0, min(int(j_float), self.N_grid - 1))
            j_high = min(j_low + 1, self.N_grid)
            
            # 插值权重
            w_i = i_float - i_low
            w_j = j_float - j_low
            
            # 双线性插值
            value[0] = (self.tensor_data[j_low, i_low] * (1 - w_i) * (1 - w_j) +
                       self.tensor_data[j_low, i_high] * w_i * (1 - w_j) +
                       self.tensor_data[j_high, i_low] * (1 - w_i) * w_j +
                       self.tensor_data[j_high, i_high] * w_i * w_j)
        
        def value_shape(self):
            return ()
    
    # 创建表达式并插值到函数空间
    expr = TensorExpression(tensor, degree=1)
    func = Function(V)
    func.interpolate(expr)
    return func


def compute_gradient_adjoint_fno(adjoint_model, adjoint_normalizers, adjoint_params, sigma_initial, u_current, u_target, device, apply_zero_mean=False):
    """
    使用伴随FNO模型计算梯度
    
    Args:
        adjoint_model: 训练好的伴随FNO模型
        adjoint_normalizers: 伴随问题的归一化器
        adjoint_params: 伴随问题的归一化参数（包含mean和std）
        sigma_initial: 初始电导率（伴随FNO模型需要作为输入）
        u_current: 当前解（从正向FNO计算得到，避免重复计算）
        u_target: 目标解
        device: 计算设备
        apply_zero_mean: 是否应用零均值约束（已废弃，保持为False）
    
    Returns:
        gradient_adjoint_fno: 梯度
        lambda_pred: 伴随变量
        elapsed_time: 计算时间
    """
    print("Computing gradient with Adjoint FNO method...")
    start_time = time.time()
    
    # 计算当前解与目标解的差值（全域）
    u_diff_for_adjoint = u_current - u_target
    
    # 使用从04_generate_adjoint_data.py导入的函数来正确处理边界值和拉普拉斯提升
    u_diff_lift_inference = process_boundary_diff_for_adjoint_inference(
        u_current, u_target, N_grid=u_diff_for_adjoint.shape[0]-1, device=device
    )
        
        
    # --- 3. 使用伴随FNO模型求解伴随问题 ---
    # 使用与训练时完全相同的归一化方式
    sigma_mean = adjoint_params['sigma']['mean']
    sigma_std = adjoint_params['sigma']['std']
    u_diff_lift_mean = adjoint_params['u_diff_lift']['mean']
    u_diff_lift_std = adjoint_params['u_diff_lift']['std']
    
    # 手动归一化，与训练时完全一致
    sigma_norm_adj = (sigma_initial - sigma_mean) / sigma_std
    u_diff_lift_norm = (u_diff_lift_inference - u_diff_lift_mean) / u_diff_lift_std
    
    # 构建输入张量
    x_in_adj = torch.stack([sigma_norm_adj, u_diff_lift_norm], dim=-1).unsqueeze(0)
        
    # 伴随FNO预测伴随变量（归一化的输出）
    lambda_pred = adjoint_model(x_in_adj).reshape(x_in_adj.shape[0], sigma_initial.shape[0], sigma_initial.shape[1])
        

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Adjoint FNO solve time: {elapsed_time:.4f}s")

    # --- 4. 计算梯度 ∂J/∂σ = -∇u·∇λ ---
    # 计算梯度（使用有限差分）
    u_current_2d = u_current.detach().cpu().numpy()
    lambda_pred_2d = lambda_pred.squeeze(0).detach().cpu().numpy()  # 去掉batch维度
    
    grad_u_x, grad_u_y = compute_gradient_fd(u_current_2d, dx=2.0/u_current_2d.shape[0])
    grad_lambda_x, grad_lambda_y = compute_gradient_fd(lambda_pred_2d, dx=2.0/lambda_pred_2d.shape[0])
    
    # 梯度计算：∂J/∂σ = -∇u·∇λ
    gradient_adjoint_fno = -(grad_u_x * grad_lambda_x + grad_u_y * grad_lambda_y)
    
    # 返回4个值以匹配调用代码的期望，确保所有数组都是2D
    return gradient_adjoint_fno, lambda_pred_2d, u_current_2d, elapsed_time


def check_and_generate_normalization_params(config):
    """检查归一化参数文件是否存在，如果不存在则自动生成"""
    print("检查归一化参数文件...")
    
    # 检查原始FNO归一化参数文件
    if not os.path.exists(config.paths.fno_norm_params):
        print(f"原始FNO归一化参数文件不存在: {config.paths.fno_norm_params}")
        print("正在自动生成...")
        
        # 检查训练数据是否存在
        train_data_path = './data/train_data.npy'
        if not os.path.exists(train_data_path):
            print(f"错误：训练数据文件不存在: {train_data_path}")
            print("请先运行训练脚本生成训练数据")
            return False
        
        # 加载训练数据并计算归一化参数
        try:
            train_data = np.load(train_data_path, allow_pickle=True).item()
            print("数据字段:", list(train_data.keys()))
            
            # 计算归一化参数
            sigma_mean = np.mean(train_data['sigma'])
            sigma_std = np.std(train_data['sigma'])
            g_lift_mean = np.mean(train_data['g_lift'])
            g_lift_std = np.std(train_data['g_lift'])
            
            # 创建归一化参数字典
            norm_params = {
                'sigma': {'mean': sigma_mean, 'std': sigma_std},
                'g_lift': {'mean': g_lift_mean, 'std': g_lift_std}
            }
            
            # 保存到指定路径
            os.makedirs(os.path.dirname(config.paths.fno_norm_params), exist_ok=True)
            np.save(config.paths.fno_norm_params, norm_params)
            print(f"原始FNO归一化参数已保存到: {config.paths.fno_norm_params}")
            print(f"sigma: mean={sigma_mean:.6f}, std={sigma_std:.6f}")
            print(f"g_lift: mean={g_lift_mean:.6f}, std={g_lift_std:.6f}")
            
        except Exception as e:
            print(f"生成原始FNO归一化参数时出错: {e}")
            return False
    else:
        print(f"原始FNO归一化参数文件已存在: {config.paths.fno_norm_params}")
    
    # 检查伴随FNO归一化参数文件
    if not os.path.exists(config.paths.adjoint_norm_params):
        print(f"伴随FNO归一化参数文件不存在: {config.paths.adjoint_norm_params}")
        print("正在自动生成...")
        
        # 检查伴随训练数据是否存在
        adjoint_train_data_path = './data/adjoint_train_data.npy'
        if not os.path.exists(adjoint_train_data_path):
            print(f"错误：伴随训练数据文件不存在: {adjoint_train_data_path}")
            print("请先运行伴随训练脚本生成训练数据")
            return False
        
        # 加载伴随训练数据并计算归一化参数
        try:
            train_data = np.load(adjoint_train_data_path, allow_pickle=True).item()
            print("伴随数据字段:", list(train_data.keys()))
            
            # 计算归一化参数
            sigma_mean = np.mean(train_data['sigma'])
            sigma_std = np.std(train_data['sigma'])
            u_diff_lift_mean = np.mean(train_data['u_diff_lift'])
            u_diff_lift_std = np.std(train_data['u_diff_lift'])
            
            # 创建归一化参数字典
            norm_params = {
                'sigma': {'mean': sigma_mean, 'std': sigma_std},
                'u_diff_lift': {'mean': u_diff_lift_mean, 'std': u_diff_lift_std}
            }
            
            # 保存到指定路径
            os.makedirs(os.path.dirname(config.paths.adjoint_norm_params), exist_ok=True)
            np.save(config.paths.adjoint_norm_params, norm_params)
            print(f"伴随FNO归一化参数已保存到: {config.paths.adjoint_norm_params}")
            print(f"sigma: mean={sigma_mean:.6f}, std={sigma_std:.6f}")
            print(f"u_diff_lift: mean={u_diff_lift_mean:.6f}, std={u_diff_lift_std:.6f}")
            print(f"注意：lambda输出不进行归一化，保持物理量级")
            
        except Exception as e:
            print(f"生成伴随FNO归一化参数时出错: {e}")
            return False
    else:
        print(f"伴随FNO归一化参数文件已存在: {config.paths.adjoint_norm_params}")
    
    print("所有归一化参数文件检查完成！")
    return True


def load_models_and_data(config, device):
    """加载模型、归一化器和数据读取器"""
    
    # 加载原始FNO模型
    fno_model_params = {
        'modes1': config.model.modes1,
        'modes2': config.model.modes2,
        'width': config.model.fno_width,
        'in_channels': config.model.in_channels
    }
    fno_model = load_fno_model(config.paths.fno_model, fno_model_params, device)

    # 加载伴随FNO模型
    if adjoint_fno_available:
        adjoint_model = AdjointFNO2d(
            modes1=config.model.modes1,
            modes2=config.model.modes2,
            width=config.model.adjoint_width
        ).to(device)
    else:
        # 如果没有专门的伴随FNO类，使用普通FNO2d
        adjoint_model = FNO2d(
            modes1=config.model.modes1,
            modes2=config.model.modes2,
            width=config.model.adjoint_width,
            in_channels=2  # 伴随模型输入是2通道 (sigma, u_diff_lift)，网格坐标会自动添加
        ).to(device)
    
    # 加载adjoint模型checkpoint，设置weights_only=False以支持包含numpy对象的文件
    checkpoint = torch.load(config.paths.adjoint_model, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        adjoint_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        adjoint_model.load_state_dict(checkpoint)
    adjoint_model.eval()
    print("Adjoint FNO model loaded successfully.")

    # 加载原始FNO的归一化参数
    fno_params = np.load(config.paths.fno_norm_params, allow_pickle=True).item()
    fno_normalizers = {
        'sigma': GaussianNormalizer(mean=torch.tensor(fno_params['sigma']['mean']), std=torch.tensor(fno_params['sigma']['std'])),
        'u': GaussianNormalizer(mean=torch.tensor(fno_params['u']['mean']), std=torch.tensor(fno_params['u']['std'])),
        'g_lift': GaussianNormalizer(mean=torch.tensor(fno_params['g_lift']['mean']), std=torch.tensor(fno_params['g_lift']['std']))
    }
    
    # 加载伴随FNO的归一化参数
    adjoint_params = np.load(config.paths.adjoint_norm_params, allow_pickle=True).item()
    adjoint_normalizers = {
        'sigma': GaussianNormalizer(mean=torch.tensor(adjoint_params['sigma']['mean']), std=torch.tensor(adjoint_params['sigma']['std'])),
        'u_diff_lift': GaussianNormalizer(mean=torch.tensor(adjoint_params['u_diff_lift']['mean']), std=torch.tensor(adjoint_params['u_diff_lift']['std']))
    }
    
    # 将归一化器移到指定设备
    for key in fno_normalizers:
        fno_normalizers[key].to(device)
    for key in adjoint_normalizers:
        adjoint_normalizers[key].to(device)
        
    print("Normalizers loaded successfully.")

    # 创建数据读取器
    reader = NpyReader(config.paths.data, to_cuda=(device.type == 'cuda'))
    print("Data reader created successfully.")

    return fno_model, adjoint_model, fno_normalizers, adjoint_normalizers, reader


def prepare_sample(reader, idx, device):
    """从reader中准备指定索引的样本数据。"""
    sigma = reader.read_field('sigma')[idx]
    g_lift = reader.read_field('g_lift')[idx] 
    u = reader.read_field('u')[idx]
    return sigma, g_lift, u


def print_error_statistics(results):
    """计算和打印各种梯度方法相对于FEM的误差统计。"""
    print("\n" + "="*60)
    print("梯度精度误差统计 (相对于FEM Ground Truth)")
    print("="*60)
    
    grad_fem = results['grad_fem']
    fem_norm = np.linalg.norm(grad_fem)
    
    methods = [
        ('原始FNO 自动微分', results['grad_fno_ad']),
        ('伴随FNO 方法', results['grad_adjoint_fno_ad'])
    ]
    
    for method_name, grad in methods:
        # L2相对误差
        l2_error = np.linalg.norm(grad - grad_fem) / fem_norm
        
        # L∞相对误差
        linf_error = np.max(np.abs(grad - grad_fem)) / np.max(np.abs(grad_fem))
        
        # 相关系数
        correlation = np.corrcoef(grad.flatten(), grad_fem.flatten())[0, 1]
        
        print(f"{method_name}:")
        print(f"  L2 相对误差: {l2_error:.6f}")
        print(f"  L∞ 相对误差: {linf_error:.6f}")
        print(f"  相关系数: {correlation:.6f}")
        print()


def plot_results(results, config, idx_initial, idx_target):
    """绘制所有结果的对比图。"""
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    
    # 提取数据
    sigma_i, sigma_t, u_t = results['sigma_initial'], results['sigma_target'], results['u_target']
    grad_fem, u_fem, p_fem, t_fem = results['grad_fem'], results['u_fem'], results['p_fem'], results['t_fem']
    grad_fno_ad, u_fno_ad, t_fno_ad = results['grad_fno_ad'], results['u_fno_ad'], results['t_fno_ad']
    grad_adjoint_fno_ad, lambda_adjoint_fno, t_adjoint_fno_ad = results['grad_adjoint_fno_ad'], results['lambda_adjoint_fno'], results['t_adjoint_fno_ad']
    g_true, g_lift = results['g_true'], results['g_lift']
    
    # 绘图辅助函数
    def plot_field(ax, data, title, cmap='viridis', is_grad=False, vmin=None, vmax=None):
        if is_grad:
            # 对于梯度，使用对称的colorbar
            vmax_grad = np.max(np.abs(data))
            vmin_grad = -vmax_grad
            im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin_grad, vmax=vmax_grad)
        else:
            # 使用提供的vmin/vmax，如果没有则自动确定
            im = ax.imshow(data, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 计算统一的colorbar范围
    # 电导率：使用sigma_t（目标）作为参考
    sigma_vmin, sigma_vmax = sigma_t.min(), sigma_t.max()
    # 前向解：使用FEM结果作为参考
    u_vmin, u_vmax = u_fem.min(), u_fem.max()
    # 伴随解：使用FEM结果作为参考
    p_vmin, p_vmax = p_fem.min(), p_fem.max()
    
    # --- 第一行: 输入、目标和边界条件 ---
    plot_field(axes[0, 0], sigma_i, 'Initial Conductivity (σ_initial)', vmin=sigma_vmin, vmax=sigma_vmax)
    plot_field(axes[0, 1], sigma_t, 'Target Conductivity (σ_target)', vmin=sigma_vmin, vmax=sigma_vmax)
    plot_field(axes[0, 2], g_true, 'Boundary Condition (j)')
    plot_field(axes[0, 3], g_lift, 'Lifted Boundary Condition (j_lift)')
    axes[0, 4].axis('off')

    # --- 第二行: FEM 方法 ---
    plot_field(axes[1, 0], u_fem, 'FEM Forward Solution', vmin=u_vmin, vmax=u_vmax)
    plot_field(axes[1, 1], p_fem, 'FEM Adjoint Solution', vmin=p_vmin, vmax=p_vmax)
    plot_field(axes[1, 2], grad_fem, f'FEM Gradient\n(Time: {t_fem:.3f}s)', cmap='bwr', is_grad=True)
    # 删除 Target Potential 子图
    axes[1, 3].axis('off')
    axes[1, 4].axis('off')

    # --- 第三行: FNO 方法对比（使用统一的colorbar范围）---
    plot_field(axes[2, 0], u_fno_ad, 'FNO Forward Solution', vmin=u_vmin, vmax=u_vmax)
    plot_field(axes[2, 1], lambda_adjoint_fno, 'Adjoint FNO λ Output', vmin=p_vmin, vmax=p_vmax)
    plot_field(axes[2, 2], grad_fno_ad, f'FNO Autodiff Gradient\n(Time: {t_fno_ad:.4f}s)', cmap='bwr', is_grad=True)
    plot_field(axes[2, 3], grad_adjoint_fno_ad, f'Adjoint FNO Gradient\n(Time: {t_adjoint_fno_ad:.4f}s)', cmap='bwr', is_grad=True)
    axes[2, 4].axis('off')
    
    # --- 第四行: 绝对误差对比 ---
    # 调试信息：检查数据
    print(f"\n=== DEBUG: Forward Solution Comparison ===")
    print(f"u_fem shape: {u_fem.shape}, type: {type(u_fem)}")
    print(f"u_fno_ad shape: {u_fno_ad.shape}, type: {type(u_fno_ad)}")
    print(f"u_fem stats - min: {np.min(u_fem):.6f}, max: {np.max(u_fem):.6f}, mean: {np.mean(u_fem):.6f}, std: {np.std(u_fem):.6f}")
    print(f"u_fno_ad stats - min: {np.min(u_fno_ad):.6f}, max: {np.max(u_fno_ad):.6f}, mean: {np.mean(u_fno_ad):.6f}, std: {np.std(u_fno_ad):.6f}")
    
    # FEM Forward vs FNO Forward 绝对误差
    forward_error = np.abs(u_fem - u_fno_ad)
    print(f"forward_error stats - min: {np.min(forward_error):.6f}, max: {np.max(forward_error):.6f}, mean: {np.mean(forward_error):.6f}")
    print(f"Expected max error: {np.max(np.abs(u_fem - u_fno_ad)):.6f}")
    print("=" * 50)
    
    plot_field(axes[3, 0], forward_error, f'|FEM Forward - FNO Forward|\nMax Error: {np.max(forward_error):.4f}', cmap='Reds')
    
    # FEM Adjoint vs Adjoint FNO λ 绝对误差
    adjoint_error = np.abs(p_fem - lambda_adjoint_fno)
    plot_field(axes[3, 1], adjoint_error, f'|FEM Adjoint - Adjoint FNO λ|\nMax Error: {np.max(adjoint_error):.4f}', cmap='Reds')
    
    # 其余子图关闭
    for i in range(2, 5):
        axes[3, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = config.paths.get('output_dir', 'diffusion_ip/logs/gradient_comparison')
    os.makedirs(output_dir, exist_ok=True)
    # 使用更精确的时间戳，包含微秒，确保文件名唯一
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{int(time.time()*1000000) % 1000000:06d}"
    filename = f'gradient_comparison_{timestamp}_s{idx_initial}_t{idx_target}.png'
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults image saved to: {save_path}")


def run_multiple_samples_comparison(config):
    """运行多个样本对的梯度比较分析。"""
    
    print("="*80)
    print("开始多样本梯度比较分析")
    print("="*80)
    
    
    for i, (idx_initial, idx_target) in enumerate(config.data.sample_pairs):
        print(f"\n{'='*60}")
        print(f"样本对 {i+1}/{len(config.data.sample_pairs)}: 初始样本 {idx_initial} -> 目标样本 {idx_target}")
        print(f"{'='*60}")
        
        # 临时修改config为当前样本对
        config.data.sample_idx_initial = idx_initial
        config.data.sample_idx_target = idx_target
        
        run_single_sample_comparison(config, show_plots=True)


def run_single_sample_comparison(config, show_plots=True):
    """运行单个样本对的比较，返回误差统计结果。"""
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 首先检查并生成归一化参数文件
    if not check_and_generate_normalization_params(config):
        print("归一化参数文件生成失败，程序退出")
        return
    
    # 加载模型和数据（与原始代码相同的逻辑）
    fno_model, adjoint_model, fno_normalizers, adjoint_normalizers, reader = load_models_and_data(config, device)
    
    # 选择样本
    idx_initial = config.data.sample_idx_initial
    idx_target = config.data.sample_idx_target
    
    sigma_initial, g_lift_initial, _ = prepare_sample(reader, idx_initial, device)
    sigma_target, _, u_target = prepare_sample(reader, idx_target, device)
    g_true_initial = reader.read_field('g')[idx_initial]
    
    print(f"初始电导率样本: {idx_initial}, 目标电导率样本: {idx_target}")
    
    # --- 1. FEM伴随解 ---
    grad_fem, u_fem, p_fem, t_fem = compute_gradient_fem_adjoint(
        sigma=sigma_initial.clone(), 
        u_target=u_target, 
        g=g_true_initial, 
        mesh_size=config.data.resolution
    )
    
    # --- 2. 原始FNO Autodiff ---
    grad_fno_ad, u_fno_ad, t_fno_ad = compute_gradient_fno_autodiff(
        model=fno_model,
        normalizers=fno_normalizers,
        sigma_initial=sigma_initial.clone(),
        g_lift_initial=g_lift_initial,
        u_target=u_target,
        device=device,
        apply_zero_mean=False
    )
    
    # --- 3. 伴随FNO ---
    # 加载伴随问题的归一化参数
    adjoint_norm_params = np.load(config.paths.adjoint_norm_params, allow_pickle=True).item()
    
    # 使用已经计算好的u_fno_ad，避免重复计算
    grad_adjoint_fno_ad, lambda_adjoint_fno, u_adjoint_fno, t_adjoint_fno_ad = compute_gradient_adjoint_fno(
        adjoint_model=adjoint_model,
        adjoint_normalizers=adjoint_normalizers,
        adjoint_params=adjoint_norm_params,
        sigma_initial=sigma_initial.clone(),
        u_current=torch.tensor(u_fno_ad, device=device),  # 使用已计算的正向解
        u_target=u_target,
        device=device,
        apply_zero_mean=False
    )
    
    # 计算误差统计
    results = {
        'sigma_initial': sigma_initial.detach().cpu().numpy(),
        'sigma_target': sigma_target.detach().cpu().numpy(),
        'u_target': u_target.detach().cpu().numpy(),
        'g_true': g_true_initial.detach().cpu().numpy(),
        'g_lift': g_lift_initial.detach().cpu().numpy(),
        'grad_fem': grad_fem, 'u_fem': u_fem, 'p_fem': p_fem, 't_fem': t_fem,
        'grad_fno_ad': grad_fno_ad, 'u_fno_ad': u_fno_ad, 't_fno_ad': t_fno_ad,
        'grad_adjoint_fno_ad': grad_adjoint_fno_ad, 'lambda_adjoint_fno': lambda_adjoint_fno, 'u_adjoint_fno': u_adjoint_fno, 't_adjoint_fno_ad': t_adjoint_fno_ad,
    }
    
    # 如果是第一个样本，显示详细结果和图像
    if show_plots:
        plot_results(results, config, idx_initial, idx_target)
        
    return 


if __name__ == '__main__':
    config = DotMap()
    
    # --- Paths ---
    config.paths = DotMap()
    config.paths.fno_model = '/amax/haibo/05-Diffusion4IP/diffusion_ip/work_dir/fno/20250903_145938/best_model.pth'
    config.paths.adjoint_model = '/amax/haibo/05-Diffusion4IP/diffusion_ip/work_dir/adjoint_fno/20250903_151032/checkpoints/best_adjoint_fno_model.pth'
    config.paths.data = '/amax/haibo/05-Diffusion4IP/diffusion_ip/data/test_data.npy'
    config.paths.fno_norm_params = '/amax/haibo/05-Diffusion4IP/diffusion_ip/work_dir/fno/20250903_145938/normalization_params.npy'
    config.paths.adjoint_norm_params = '/amax/haibo/05-Diffusion4IP/diffusion_ip/work_dir/adjoint_fno/20250903_151032/adjoint_normalization_params.npy'
    config.paths.output_dir = '/amax/haibo/05-Diffusion4IP/diffusion_ip/logs/gradient_comparison'

    # --- Model Config ---
    config.model = DotMap()
    config.model.modes1 = 12
    config.model.modes2 = 12
    config.model.fno_width = 32  # 原始FNO模型的width
    config.model.adjoint_width = 64  # 伴随FNO模型的width
    config.model.in_channels = 2

    # --- Data Config ---
    config.data = DotMap()
    config.data.resolution = 129
    config.data.sample_pairs = [
        (6, 7),   # 原始测试样本
        (0, 1),   # 新增样本对
        (2, 3),   # 新增样本对
        (4, 5),   # 新增样本对
        (8, 9),   # 新增样本对
    ]

    # 运行多个样本对的比较
    run_multiple_samples_comparison(config) 