import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroalgaeLoss(nn.Module):
    """针对微藻重建的损失函数"""
    
    def __init__(self, config):
        super(MicroalgaeLoss, self).__init__()
        self.config = config
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def compute_rgb_loss(self, pred, target):
        """RGB重建损失"""
        return self.mse_loss(pred, target)
    
    def compute_transmittance_loss(self, transmittance):
        """透明度正则化，鼓励微藻特有的半透明特性"""
        # 鼓励合理的透明度分布，既不完全透明也不完全不透明
        return ((transmittance - 0.5).abs() - 0.4).clamp(min=0).mean()
    
    def compute_depth_smoothness_loss(self, depth, weights):
        """深度平滑损失，使重建的深度更加平滑"""
        # 只在权重较大的区域计算深度梯度
        depth_dx = torch.abs(depth[:, :-1] - depth[:, 1:])
        depth_dy = torch.abs(depth[:-1, :] - depth[1:, :])
        
        # 加权深度梯度
        weights_dx = (weights[:, :-1] + weights[:, 1:]) / 2
        weights_dy = (weights[:-1, :] + weights[1:, :]) / 2
        
        smoothness_x = (depth_dx * weights_dx).mean()
        smoothness_y = (depth_dy * weights_dy).mean()
        
        return smoothness_x + smoothness_y
    
    def compute_sparsity_loss(self, sigma):
        """稀疏性损失，鼓励空间中的稀疏表示"""
        # L1正则化鼓励稀疏性
        return torch.mean(torch.abs(sigma))
    
    def __call__(self, results, target_rgb, phase='coarse'):
        """计算总损失"""
        # 基本RGB损失
        rgb_loss = self.compute_rgb_loss(results[phase]['rgb'], target_rgb)
        loss = rgb_loss
        
        # 各种正则化损失
        if self.config.transmittance_weight > 0 and 'transmittance' in results[phase]:
            transmittance_loss = self.compute_transmittance_loss(results[phase]['transmittance'])
            loss = loss + self.config.transmittance_weight * transmittance_loss
        
        if self.config.depth_smooth_weight > 0 and 'depth' in results[phase] and 'weights' in results[phase]:
            depth_loss = self.compute_depth_smoothness_loss(
                results[phase]['depth'], results[phase]['weights'])
            loss = loss + self.config.depth_smooth_weight * depth_loss
        
        if self.config.sparsity_weight > 0 and 'sigma' in results[phase]:
            sparsity_loss = self.compute_sparsity_loss(results[phase]['sigma'])
            loss = loss + self.config.sparsity_weight * sparsity_loss
        
        return loss, rgb_loss