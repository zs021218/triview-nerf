import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import get_view_matrix, generate_rays


class TriViewDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        self.image_size = config.image_size
        self.focal = 0.5 * self.image_size
        self.device = config.device
        self.training = split == 'train'

        # 加载三视图图像
        self.images = {}
        self.masks = {}

        for view in config.views:
            img_path = os.path.join(config.data_dir, f"{view}.png")
            mask_path = os.path.join(config.data_dir, f"{view}_mask.png")

            if os.path.exists(img_path):
                img = self.load_image(img_path)
                self.images[view] = img
            else:
                raise FileNotFoundError(f"Image {img_path} not found")

            # 如果有遮罩图像，加载它
            if os.path.exists(mask_path):
                mask = self.load_mask(mask_path)
                self.masks[view] = mask
            else:
                # 如果没有遮罩，则假设整个图像都是前景
                self.masks[view] = torch.ones((self.image_size, self.image_size), dtype=torch.float32)

        # 将图像保存在CPU内存中，减少GPU内存使用
        self.rays = self.generate_all_rays()

        # 根据分割决定使用的光线
        if split == 'train':
            # 训练中只使用前景区域的光线
            self.valid_indices = self.get_foreground_indices()

            # 对于大型数据集，可以随机抽样来减少内存使用
            if len(self.valid_indices) > 10000:
                # 随机抽样 10000 个光线
                rand_idx = torch.randperm(len(self.valid_indices))[:10000]
                self.valid_indices = self.valid_indices[rand_idx]
        else:
            # 测试中使用所有光线
            self.valid_indices = torch.arange(len(self.rays['rays_o']))

        print(f"Dataset loaded: {len(self.valid_indices)} valid rays")

    def load_image(self, path):
        """加载并预处理图像，针对微藻图像进行特别处理"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = np.array(img) / 255.0
        
        # 增强微藻图像的对比度
        img = np.power(img, 0.9)  # 轻微提高对比度
        
        # 如果有白色背景，可以减弱它的影响
        if hasattr(self.config, 'white_background') and self.config.white_background:
            # 假设背景接近白色，轻微压暗以突出微藻
            white_mask = (img.mean(axis=-1) > 0.9)
            img[white_mask] = img[white_mask] * 0.95
        
        return torch.tensor(img, dtype=torch.float32)

    def load_mask(self, path):
        """加载并预处理遮罩"""
        mask = Image.open(path).convert('L')
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = np.array(mask) > 0
        return torch.tensor(mask, dtype=torch.float32)

    def generate_all_rays(self):
        """为所有视角生成光线"""
        all_rays_o = []
        all_rays_d = []
        all_rgb = []
        all_view_ids = []

        for i, view in enumerate(self.config.views):
            c2w = get_view_matrix(view)
            rays_o, rays_d = generate_rays(self.image_size, self.image_size, self.focal, c2w)

            all_rays_o.append(rays_o.reshape(-1, 3))
            all_rays_d.append(rays_d.reshape(-1, 3))
            all_rgb.append(self.images[view].reshape(-1, 3))
            all_view_ids.append(torch.ones(rays_o.reshape(-1, 3).shape[0], dtype=torch.long) * i)

        return {
            'rays_o': torch.cat(all_rays_o, 0),  # 保存在CPU内存中
            'rays_d': torch.cat(all_rays_d, 0),  # 保存在CPU内存中
            'rgb': torch.cat(all_rgb, 0),  # 保存在CPU内存中
            'view_ids': torch.cat(all_view_ids, 0)  # 保存在CPU内存中
        }

    def get_foreground_indices(self):
        """获取前景区域的光线索引"""
        fg_indices = []

        start_idx = 0
        for i, view in enumerate(self.config.views):
            mask = self.masks[view].reshape(-1)
            num_rays = mask.shape[0]

            # 找到当前视图中的前景像素
            view_fg_indices = torch.where(mask > 0.5)[0] + start_idx
            fg_indices.append(view_fg_indices)

            start_idx += num_rays

        return torch.cat(fg_indices, 0)

    def data_augmentation(self, ray_batch):
        """对光线批次进行数据增强"""
        # 只在训练阶段进行增强
        if self.split != 'train' or not self.training:
            return ray_batch
        
        # 随机颜色抖动
        if np.random.random() < 0.5:
            # 随机颜色扰动因子
            color_factor = torch.ones(3) + torch.randn(3) * 0.05
            ray_batch['rgb'] = ray_batch['rgb'] * color_factor.to(ray_batch['rgb'].device)
            ray_batch['rgb'] = torch.clamp(ray_batch['rgb'], 0.0, 1.0)
        
        return ray_batch

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ray_idx = self.valid_indices[idx]
        
        ray_batch = {
            'rays_o': self.rays['rays_o'][ray_idx],
            'rays_d': self.rays['rays_d'][ray_idx],
            'rgb': self.rays['rgb'][ray_idx],
            'view_id': self.rays['view_ids'][ray_idx]
        }
        
        # 应用数据增强
        ray_batch = self.data_augmentation(ray_batch)
        
        return ray_batch