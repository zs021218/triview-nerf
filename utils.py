import torch
import numpy as np
import os


def pos_encoding(x, num_dims):
    """位置编码: 将坐标映射到高维空间"""
    encodings = [x]
    for i in range(num_dims):
        for fn in [torch.sin, torch.cos]:
            encodings.append(fn(2.0 ** i * x))
    return torch.cat(encodings, dim=-1)


def generate_rays(H, W, focal, c2w):
    """生成一批光线"""
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H))
    i = i.t()  # 转置以匹配图像坐标
    j = j.t()

    # 归一化像素坐标
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)

    # 旋转光线方向
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)

    # 光线原点 (相机位置)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_view_matrix(view_type, distance=4.0):
    """获取特定视图的相机到世界变换矩阵"""
    # 设置三个基本视角
    if view_type == "front":
        # 正面视图 (z轴方向)
        c2w = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    elif view_type == "side":
        # 侧面视图 (x轴方向)
        c2w = torch.tensor([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    elif view_type == "top":
        # 顶视图 (y轴方向)
        c2w = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown view type: {view_type}")

    return c2w


def create_dirs(dirs):
    """创建目录"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)