import torch
import time
def timeit(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.time()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.time()
        print(f"{func.__name__} execution took {(end - start)*1000:.4f}ms")
        return ret
    return wrapper

from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Dict, Optional, Union

def visualize_head_seq(
    data_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
    batch_idx: int = 0,
    max_heads_per_row: int = 4,
    figsize_scale: float = 3.0,
    cmap: str = 'viridis',
    value_range: Union[str, tuple] = 'auto',
    colorbar: bool = True,
    symmetric_range: bool = True
):
    """
    专业级多头序列数据可视化函数
    
    参数:
    - data_dict: 数据字典 {标题: 张量 (batch, heads, seq, seq) 或 (heads, seq, seq)}
    - batch_idx: 批次索引 (仅当输入为4维时生效)
    - max_heads_per_row: 每行最大显示头数
    - figsize_scale: 图像尺寸缩放系数 (基础尺寸为每个子图的大小)
    - cmap: 颜色映射方案
    - value_range: 值范围 ('auto' 或 (vmin, vmax))
    - colorbar: 是否显示颜色条
    - symmetric_range: 是否强制对称颜色范围 (适用于相关性矩阵)
    """
    
    # 输入校验和预处理
    processed_data = {}
    for name, data in data_dict.items():
        # 维度处理
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
            
        if data.ndim == 4:
            data = data[batch_idx]  # (heads, seq, seq)
        elif data.ndim == 3:
            pass  # 直接使用 (heads, seq, seq)
        else:
            raise ValueError(f"输入数据 {name} 维度错误，应为3D或4D")
            
        processed_data[name] = data

    # 获取关键参数
    num_heads = min(d.shape[0] for d in processed_data.values())
    num_datasets = len(processed_data)
    seq_len = next(iter(processed_data.values())).shape[-1]
    
    # 智能布局计算
    rows = int(np.ceil(num_heads / max_heads_per_row))
    cols = max_heads_per_row * num_datasets
    
    # 创建画布
    fig, axes = plt.subplots(
        rows, 
        cols, 
        figsize=(cols * figsize_scale , 
                rows * figsize_scale),
        gridspec_kw={'wspace':0.3, 'hspace':0.1}
    )
    
    # 统一颜色范围
    all_values = np.concatenate([d.ravel() for d in processed_data.values()])
    if value_range == 'auto':
        vmin, vmax = (all_values.min(), all_values.max()) 
        if symmetric_range:
            bound = max(abs(vmin), abs(vmax))
            vmin, vmax = -bound, bound
    else:
        vmin, vmax = value_range

    # 可视化主循环
    for head_idx in range(num_heads):
        row = head_idx // max_heads_per_row
        col_start = (head_idx % max_heads_per_row) * num_datasets
        
        for data_idx, (name, data) in enumerate(processed_data.items()):
            ax = axes[row, col_start + data_idx] if rows > 1 else axes[col_start + data_idx]
            
            # 提取当前头数据
            current_data = data[head_idx] if data.shape[0] > 1 else data[0]
            
            # 绘制热力图
            im = ax.imshow(current_data, cmap=cmap, vmin=vmin, vmax=vmax)
            
            # 标注设置
            ax.set_xticks([])
            ax.set_yticks([])
            if head_idx == 0:
                ax.set_title(f"{name}\nSeqLen={seq_len}", 
                           fontsize=9, pad=12, color='#2F4F4F')
                
            if data_idx == 0:
                ax.text(-0.1, 0.5, f'Head {head_idx}', 
                       rotation=90, va='center', ha='right',
                       transform=ax.transAxes, fontsize=8)
            
            # 添加颜色条
            if colorbar and (head_idx == num_heads-1) and (data_idx == num_datasets-1):
                cax = fig.add_axes([ax.get_position().x1+0.02, 
                                  ax.get_position().y0, 
                                  0.02, 
                                  ax.get_position().height])
                fig.colorbar(im, cax=cax)

    # 隐藏空白子图
    for r in range(rows):
        for c in range(cols):
            if (r * max_heads_per_row + c//num_datasets) >= num_heads:
                if rows > 1:
                    axes[r,c].axis('off')
                else:
                    axes[c].axis('off')

    plt.suptitle(f"Multi-Head Attention Pattern Visualization (Batch {batch_idx})", 
                y=1.02, fontsize=11, color='#2F4F4F')
    plt.tight_layout()
    plt.show()

# 使用示例 ---------------------------------------------------
if __name__ == "__main__":
    # 生成模拟数据
    batch_size, num_heads, seq_len = 2, 6, 64
    
    # 原始注意力矩阵
    attn = torch.rand(batch_size, num_heads, seq_len, seq_len)
    # 稀疏掩码
    mask = (torch.rand(batch_size, 1, seq_len, seq_len) > 0.7).expand(-1, num_heads, -1, -1)
    # 相关性矩阵
    corr = torch.randn(batch_size, num_heads, seq_len, seq_len)

    # 调用可视化函数
    visualize_head_seq(
        data_dict={
            "Raw Attention": attn
        },
        max_heads_per_row=3,
        figsize_scale=2.5,
        cmap='coolwarm',
        symmetric_range=True,
        value_range=(-3, 3)
    )

import torch
from functools import wraps

def preserve_rng_state(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 保存当前的随机状态
        cpu_state = torch.get_rng_state()
        cuda_states = []
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                cuda_states.append(torch.cuda.get_rng_state(device))
        try:
            # 执行被装饰的函数
            result = func(*args, **kwargs)
            return result
        finally:
            # 恢复随机状态
            torch.set_rng_state(cpu_state)
            if torch.cuda.is_available():
                for device, state in enumerate(cuda_states):
                    torch.cuda.set_rng_state(state, device)
    return wrapper

import json
import matplotlib.pyplot as plt
def analyze_and_visualize(filename='sparsity_records.json', mask_type='all_mask'):
    """Analyze and visualize sparsity data from a file for the specified mask type.
    mask_type should be one of 'inner_frame_mask', 'outer_frame_mask', or 'all_mask'. 
    """
    # 尝试加载 JSON 文件
    try:
        with open(filename, 'r') as f:
            sparsity_records = json.load(f)
    except FileNotFoundError:
        print(f"文件 {filename} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"解析文件 {filename} 的 JSON 数据时出错。")
        return

    # 检查文件中是否有指定 mask_type 的数据
    if mask_type not in sparsity_records:
        print(f"文件中没有 '{mask_type}' 类型的数据。")
        return

    records = sparsity_records[mask_type]
    if not records:
        print("没有可用的 sparsity 数据进行分析。")
        return

    # 从记录中提取数据
    timesteps = [r[0] for r in records]
    layeridxs = [r[1] for r in records]
    sparsities = [r[2] for r in records]

    # 确定层数（假设层索引从 0 开始）
    layernum = max(layeridxs) + 1

    # 为每一层绘制 sparsity 随 timestep 变化的折线图
    plt.figure(figsize=(10, 6))
    for layer in range(layernum):
        layer_sparsities = [sparsities[i] for i in range(len(records)) if layeridxs[i] == layer]
        layer_timesteps = [timesteps[i] for i in range(len(records)) if layeridxs[i] == layer]
        if layer_sparsities:  # 仅在有数据时绘制
            plt.plot(layer_timesteps, layer_sparsities, label=f'Layer {layer}')

    plt.xlabel('Timestep')
    plt.ylabel('Sparsity')
    plt.title(f'Sparsity of {mask_type} over Timesteps for Each Layer')
    plt.legend()
    plt.grid(True)
    plt.show()