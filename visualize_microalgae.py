import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
from utils import get_view_matrix


def create_microalgae_colormap():
    """创建适合微藻可视化的颜色映射"""
    # 从透明绿色到不透明深绿色的渐变
    colors = [(0.0, 0.0, 0.0, 0.0),  # 透明
              (0.1, 0.5, 0.1, 0.3),  # 浅绿色，半透明
              (0.2, 0.7, 0.2, 0.7),  # 中绿色
              (0.0, 0.6, 0.3, 1.0)]  # 深绿色，不透明
    
    return LinearSegmentedColormap.from_list('microalgae', colors)


def visualize_microalgae_density(model, renderer, config, save_dir, view_name="top"):
    """可视化微藻密度分布"""
    model.eval()
    
    # 设置渲染参数
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size
    
    # 获取相机位置
    c2w = get_view_matrix(view_name)
    
    with torch.no_grad():
        # 渲染一个视图
        results = renderer.render_image(model, H, W, focal, c2w)
        
        # 获取密度信息
        if 'fine' in results:
            weights = results['fine']['weights']
            depth = results['fine']['depth']
        else:
            weights = results['coarse']['weights']
            depth = results['coarse']['depth']
    
    # 将权重重塑为图像
    weights_map = weights.sum(dim=-1).reshape(H, W).cpu().numpy()
    depth_map = depth.reshape(H, W).cpu().numpy()
    
    # 归一化以便可视化
    weights_map = (weights_map - weights_map.min()) / (weights_map.max() - weights_map.min() + 1e-8)
    
    # 创建Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # 绘制密度图
    cmap = create_microalgae_colormap()
    im1 = ax1.imshow(weights_map, cmap=cmap)
    ax1.set_title('Microalgae Density')
    plt.colorbar(im1, ax=ax1)
    
    # 绘制深度图
    im2 = ax2.imshow(depth_map, cmap='turbo')
    ax2.set_title('Depth Map')
    plt.colorbar(im2, ax=ax2)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"microalgae_density_{view_name}.png"), dpi=300)
    plt.close()
    
    # 另外创建一个彩色叠加图
    if 'fine' in results:
        rgb_map = results['fine']['rgb']
    else:
        rgb_map = results['coarse']['rgb']
    
    rgb_map = rgb_map.cpu().numpy()
    
    # 创建彩色叠加
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb_map)
    
    # 叠加密度图，使用自定义colormap和透明度
    density_overlay = ax.imshow(weights_map, cmap=cmap, alpha=0.5)
    ax.set_title('RGB with Density Overlay')
    plt.colorbar(density_overlay, ax=ax)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"microalgae_overlay_{view_name}.png"), dpi=300)
    plt.close()
    
    model.train()
    return rgb_map, weights_map, depth_map


def create_3d_visualization(model, renderer, config, save_dir, num_views=36):
    """创建3D可视化，展示微藻从不同角度的样子"""
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = config.device
    
    # 设置渲染参数
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size
    
    # 生成360度旋转的视图
    frames = []
    for i in range(num_views):
        angle = i * (360 / num_views) * np.pi / 180
        
        # 创建俯视旋转矩阵
        elevation = 30 * np.pi / 180  # 30度俯视角
        
        # 组合旋转矩阵
        c2w = torch.tensor([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, np.cos(elevation), -np.sin(elevation), 0],
            [np.sin(angle), np.sin(elevation), np.cos(angle) * np.cos(elevation), -2],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        
        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, c2w)
        
        # 使用细采样结果(如果有的话)，否则使用粗采样结果
        if 'fine' in results:
            rgb = results['fine']['rgb']
            weights = results['fine']['weights']
        else:
            rgb = results['coarse']['rgb']
            weights = results['coarse']['weights']
        
        # 将RGB和密度转换为numpy数组
        rgb_np = rgb.cpu().numpy()
        weights_np = weights.sum(dim=-1).reshape(H, W).cpu().numpy()
        weights_np = (weights_np - weights_np.min()) / (weights_np.max() - weights_np.min() + 1e-8)
        
        # 创建密度彩色图
        cmap = create_microalgae_colormap()
        weights_colored = plt.cm.get_cmap(cmap)(weights_np)
        weights_colored = (weights_colored[:, :, :3] * 255).astype(np.uint8)
        
        # 创建叠加效果 - 混合RGB和密度图
        rgb_uint8 = (rgb_np * 255).astype(np.uint8)
        
        # 保存图像
        rgb_path = os.path.join(save_dir, f"view_{i:03d}_rgb.png")
        weights_path = os.path.join(save_dir, f"view_{i:03d}_density.png")
        
        Image.fromarray(rgb_uint8).save(rgb_path)
        Image.fromarray(weights_colored).save(weights_path)
        
        frames.append(rgb_uint8)
    
    # 创建GIF动画
    try:
        import imageio
        gif_path = os.path.join(save_dir, "microalgae_3d.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        print(f"Created 3D visualization GIF at {gif_path}")
    except ImportError:
        print("imageio not found. Install with 'pip install imageio' to create GIFs.")
    
    model.train()
    
    return frames