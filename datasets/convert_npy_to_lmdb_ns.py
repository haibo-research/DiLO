"""
将Navier-Stokes .npy数据转换为LMDB格式

根据InverseBench的数据格式要求：
- 数据存储在LMDB数据库中
- 每个样本是一个128x128的float32数组
- 数据范围约为 [-10, 10]，均值接近0
- 使用涡度场（vorticity）作为训练目标

转换过程：
1. 加载 .npy 格式的训练数据
2. 提取初始涡度场（对应于 t=5.0 时演化的结果）
3. 将每个样本序列化为 float32 格式
4. 存储到LMDB数据库中
"""

import os
import lmdb
import numpy as np
import argparse
from tqdm import tqdm


def convert_npy_to_lmdb(npy_path, lmdb_path, data_key='initial_vorticity', 
                        resolution=128, verbose=True):
    """
    将.npy格式的数据转换为LMDB格式
    
    参数:
        npy_path: .npy文件路径
        lmdb_path: LMDB输出路径
        data_key: 要提取的数据键（默认为初始涡度）
        resolution: 图像分辨率
        verbose: 是否打印详细信息
    """
    # 加载.npy数据
    if verbose:
        print(f"加载数据: {npy_path}")
    
    data_dict = np.load(npy_path, allow_pickle=True).item()
    
    if verbose:
        print(f"数据字典包含的键: {list(data_dict.keys())}")
    
    # 提取目标数据
    if data_key not in data_dict:
        raise ValueError(f"数据字典中不包含键 '{data_key}'")
    
    data = data_dict[data_key]
    
    if verbose:
        print(f"\n数据信息:")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数值范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  均值: {data.mean():.4f}")
        print(f"  标准差: {data.std():.4f}")
    
    # 检查数据形状
    num_samples = data.shape[0]
    if len(data.shape) == 3 and data.shape[1] == resolution and data.shape[2] == resolution:
        # 数据已经是 (N, H, W) 格式
        pass
    elif len(data.shape) == 4 and data.shape[1] == 1:
        # 数据是 (N, 1, H, W) 格式，squeeze掉通道维度
        data = data.squeeze(1)
    else:
        raise ValueError(f"不支持的数据形状: {data.shape}")
    
    # 创建LMDB数据库
    os.makedirs(lmdb_path, exist_ok=True)
    
    # 计算map_size（数据大小的2倍以确保足够空间）
    map_size = num_samples * resolution * resolution * 4 * 2  # float32占4字节
    
    if verbose:
        print(f"\n创建LMDB数据库:")
        print(f"  输出路径: {lmdb_path}")
        print(f"  样本数量: {num_samples}")
        print(f"  Map size: {map_size / 1024 / 1024:.2f} MB")
    
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    # 写入数据
    with env.begin(write=True) as txn:
        for idx in tqdm(range(num_samples), desc="写入LMDB", disable=not verbose):
            # 获取单个样本
            sample = data[idx].astype(np.float32)
            
            # 添加通道维度 (1, H, W)
            sample = sample.reshape(1, resolution, resolution)
            
            # 序列化
            key = f'{idx}'.encode('utf-8')
            value = sample.tobytes()
            
            # 写入
            txn.put(key, value)
    
    env.close()
    
    if verbose:
        print(f"✓ 数据已成功转换到: {lmdb_path}")
        print(f"  总样本数: {num_samples}")


def convert_evolved_vorticity_to_lmdb(npy_path, lmdb_path, Re_key='Re_200',
                                     resolution=128, verbose=True):
    """
    将演化后的涡度场转换为LMDB格式
    
    这个函数用于转换经过时间演化后的涡度场，
    作为扩散模型的训练目标。
    
    参数:
        npy_path: .npy文件路径
        lmdb_path: LMDB输出路径
        Re_key: Reynolds数对应的键（例如 'Re_200'）
        resolution: 图像分辨率
        verbose: 是否打印详细信息
    """
    # 加载.npy数据
    if verbose:
        print(f"加载数据: {npy_path}")
    
    data_dict = np.load(npy_path, allow_pickle=True).item()
    
    if verbose:
        print(f"数据字典包含的键: {list(data_dict.keys())}")
    
    # 构建涡度场的键名
    vorticity_key = f'vorticity_{Re_key}'
    
    if vorticity_key not in data_dict:
        raise ValueError(f"数据字典中不包含键 '{vorticity_key}'")
    
    data = data_dict[vorticity_key]
    
    if verbose:
        print(f"\n涡度场数据信息:")
        print(f"  形状: {data.shape}")
        print(f"  数据类型: {data.dtype}")
        print(f"  数值范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"  均值: {data.mean():.4f}")
        print(f"  标准差: {data.std():.4f}")
    
    # 检查数据形状
    num_samples = data.shape[0]
    if len(data.shape) == 3 and data.shape[1] == resolution and data.shape[2] == resolution:
        pass
    elif len(data.shape) == 4 and data.shape[1] == 1:
        data = data.squeeze(1)
    else:
        raise ValueError(f"不支持的数据形状: {data.shape}")
    
    # 创建LMDB数据库
    os.makedirs(lmdb_path, exist_ok=True)
    map_size = num_samples * resolution * resolution * 4 * 2
    
    if verbose:
        print(f"\n创建LMDB数据库:")
        print(f"  输出路径: {lmdb_path}")
        print(f"  样本数量: {num_samples}")
        print(f"  Map size: {map_size / 1024 / 1024:.2f} MB")
    
    env = lmdb.open(lmdb_path, map_size=map_size)
    
    # 写入数据
    with env.begin(write=True) as txn:
        for idx in tqdm(range(num_samples), desc="写入LMDB", disable=not verbose):
            sample = data[idx].astype(np.float32)
            sample = sample.reshape(1, resolution, resolution)
            key = f'{idx}'.encode('utf-8')
            value = sample.tobytes()
            txn.put(key, value)
    
    env.close()
    
    if verbose:
        print(f"✓ 数据已成功转换到: {lmdb_path}")


def main():
    parser = argparse.ArgumentParser(description="将Navier-Stokes .npy数据转换为LMDB格式")
    parser.add_argument("--input", type=str, required=True, help=".npy输入文件路径")
    parser.add_argument("--output", type=str, required=True, help="LMDB输出目录路径")
    parser.add_argument("--data_key", type=str, default="initial_vorticity", 
                       help="要提取的数据键 (default: initial_vorticity)")
    parser.add_argument("--resolution", type=int, default=128, help="图像分辨率")
    parser.add_argument("--mode", type=str, default="initial", choices=["initial", "evolved"],
                       help="转换模式: initial (初始涡度) 或 evolved (演化后涡度)")
    parser.add_argument("--Re_key", type=str, default="Re_200",
                       help="Reynolds数键（仅用于evolved模式）")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Navier-Stokes数据格式转换: .npy -> LMDB")
    print("="*80)
    print(f"输入文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"转换模式: {args.mode}")
    print(f"分辨率: {args.resolution}x{args.resolution}")
    print("="*80)
    
    if args.mode == "initial":
        convert_npy_to_lmdb(
            args.input, 
            args.output, 
            data_key=args.data_key,
            resolution=args.resolution,
            verbose=True
        )
    elif args.mode == "evolved":
        convert_evolved_vorticity_to_lmdb(
            args.input,
            args.output,
            Re_key=args.Re_key,
            resolution=args.resolution,
            verbose=True
        )
    
    print("\n" + "="*80)
    print("转换完成！")
    print("="*80)
    
    # 验证LMDB数据
    print("\n验证LMDB数据...")
    env = lmdb.open(args.output, readonly=True, lock=False, create=False)
    txn = env.begin(write=False)
    stats = txn.stat()
    print(f"  总条目数: {stats['entries']}")
    
    # 读取第一个样本
    key = '0'.encode('utf-8')
    img_bytes = txn.get(key)
    if img_bytes:
        img = np.frombuffer(img_bytes, dtype=np.float32).reshape(1, args.resolution, args.resolution)
        print(f"  第一个样本:")
        print(f"    形状: {img.shape}")
        print(f"    数值范围: [{img.min():.4f}, {img.max():.4f}]")
        print(f"    均值: {img.mean():.4f}")
        print(f"    标准差: {img.std():.4f}")
    
    env.close()
    print("✓ 验证通过")


if __name__ == "__main__":
    main()


