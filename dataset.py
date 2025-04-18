import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2
from utils import get_view_matrix, generate_rays, preprocess_microalgae_image


class MicroalgaeDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.views = config.views

        # 设置数据目录
        self.data_dir = config.data_dir

        # 加载和预处理图像
        self.images = {}
        for view in self.views:
            img_path = os.path.join(self.data_dir, f"{view}.png")
            if os.path.exists(img_path):
                # 读取图像
                img = np.array(Image.open(img_path)) / 255.0

                # 应用微藻特定的预处理
                if hasattr(config, 'is_microalgae') and config.is_microalgae:
                    img = preprocess_microalgae_image(img)
                    if isinstance(img, np.ndarray) and img.max() > 1.0:
                        img = img / 255.0

                self.images[view] = torch.from_numpy(img).float()

                # 确保图像尺寸一致
                if self.images[view].shape[0] != config.image_size or self.images[view].shape[1] != config.image_size:
                    self.images[view] = torch.nn.functional.interpolate(
                        self.images[view].permute(2, 0, 1).unsqueeze(0),
                        size=(config.image_size, config.image_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)
            else:
                raise FileNotFoundError(f"图像未找到: {img_path}")

        # 生成光线数据
        self.rays_o = []
        self.rays_d = []
        self.target_rgb = []

        H = W = config.image_size
        focal = 0.5 * config.image_size  # 更适合圆形微藻的焦距

        print("生成训练光线...")
        for view in self.views:
            c2w = get_view_matrix(view, distance=config.far * 0.8)  # 调整距离
            rays_o, rays_d = generate_rays(H, W, focal, c2w)

            self.rays_o.append(rays_o.reshape(-1, 3))
            self.rays_d.append(rays_d.reshape(-1, 3))
            self.target_rgb.append(self.images[view].reshape(-1, 3))

        # 连接所有视图的光线
        self.rays_o = torch.cat(self.rays_o, 0)
        self.rays_d = torch.cat(self.rays_d, 0)
        self.target_rgb = torch.cat(self.target_rgb, 0)

        # 为训练随机打乱数据
        if split == 'train':
            indices = torch.randperm(self.rays_o.shape[0])
            self.rays_o = self.rays_o[indices]
            self.rays_d = self.rays_d[indices]
            self.target_rgb = self.target_rgb[indices]

        print(f"数据集创建完成: {self.rays_o.shape[0]} 个光线")

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        return {
            'rays_o': self.rays_o[idx],
            'rays_d': self.rays_d[idx],
            'rgb': self.target_rgb[idx]
        }


def generate_intermediate_views(config):
    """生成原始三视图之间的中间视角，便于更平滑的过渡"""
    views = []

    # 从前视图到侧视图的过渡
    num_steps = 8  # 中间步数
    for i in range(1, num_steps):
        angle = i * (90.0 / num_steps) * np.pi / 180
        name = f"front_to_side_{i}"

        c2w = torch.tensor([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), -config.far * 0.8],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        views.append({"name": name, "c2w": c2w})

    # 从侧视图到顶视图的过渡
    for i in range(1, num_steps):
        t = i / num_steps
        name = f"side_to_top_{i}"

        # 使用球面线性插值(SLERP)在两个视图之间平滑过渡
        side_c2w = get_view_matrix("side", distance=config.far * 0.8)
        top_c2w = get_view_matrix("top", distance=config.far * 0.8)

        # 提取旋转部分
        side_rot = side_c2w[:3, :3]
        top_rot = top_c2w[:3, :3]

        # 使用矩阵指数和对数进行插值
        import scipy.linalg
        log_side = scipy.linalg.logm(side_rot.numpy())
        log_top = scipy.linalg.logm(top_rot.numpy())
        interp_log = log_side + t * (log_top - log_side)
        interp_rot = scipy.linalg.expm(interp_log)

        # 组合回完整变换矩阵
        c2w = torch.zeros((4, 4), dtype=torch.float32)
        c2w[:3, :3] = torch.tensor(interp_rot, dtype=torch.float32)
        c2w[:3, 3] = side_c2w[:3, 3] + t * (top_c2w[:3, 3] - side_c2w[:3, 3])
        c2w[3, 3] = 1.0

        views.append({"name": name, "c2w": c2w})

    # 从顶视图到前视图的过渡
    for i in range(1, num_steps):
        t = i / num_steps
        name = f"top_to_front_{i}"

        top_c2w = get_view_matrix("top", distance=config.far * 0.8)
        front_c2w = get_view_matrix("front", distance=config.far * 0.8)

        # 使用类似的插值方法
        top_rot = top_c2w[:3, :3]
        front_rot = front_c2w[:3, :3]

        import scipy.linalg
        log_top = scipy.linalg.logm(top_rot.numpy())
        log_front = scipy.linalg.logm(front_rot.numpy())
        interp_log = log_top + t * (log_front - log_top)
        interp_rot = scipy.linalg.expm(interp_log)

        c2w = torch.zeros((4, 4), dtype=torch.float32)
        c2w[:3, :3] = torch.tensor(interp_rot, dtype=torch.float32)
        c2w[:3, 3] = top_c2w[:3, 3] + t * (front_c2w[:3, 3] - top_c2w[:3, 3])
        c2w[3, 3] = 1.0

        views.append({"name": name, "c2w": c2w})

    # 添加360度旋转视图
    num_rot = 36  # 360度等分36份，每10度一个
    for i in range(num_rot):
        angle = i * (360.0 / num_rot) * np.pi / 180
        name = f"rot_{i:03d}"

        # 创建围绕Y轴旋转的视图
        c2w = torch.tensor([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), -config.far * 0.8],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        views.append({"name": name, "c2w": c2w})

    print(f"生成了 {len(views)} 个中间视图")
    return views