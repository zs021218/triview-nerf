import torch
import torch.nn.functional as F
import numpy as np
from utils import generate_rays


class MicroalgaeVolumeRenderer:
    """针对微藻优化的体积渲染器"""

    def __init__(self, config):
        self.config = config

    def sample_pdf(self, bins, weights, N_samples, det=False):
        """
        基于权重对bins进行重要性采样，加强对微藻边缘的采样
        """
        device = weights.device

        # 获取PDF
        weights = weights + 1e-5  # 防止除零
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

        # 微藻样本的边缘采样增强
        if hasattr(self.config, 'is_microalgae') and self.config.is_microalgae:
            # 扰动CDF使边缘更可能被选中
            mid_point = cdf.shape[-1] // 2
            edge_width = cdf.shape[-1] // 4

            # 在边缘区域增强CDF
            edge_mask = torch.zeros_like(cdf)
            edge_mask[..., mid_point - edge_width:mid_point + edge_width] = 1.0
            edge_factor = self.config.edge_sampling_boost * edge_mask

            # 应用边缘增强，同时保持CDF的单调性
            cdf_slope = cdf[..., 1:] - cdf[..., :-1]
            enhanced_slope = cdf_slope * (1.0 + edge_factor[..., :-1])
            enhanced_slope = F.softmax(enhanced_slope, dim=-1) * (cdf[..., -1:] - cdf[..., :1])

            # 重建CDF
            enhanced_cdf = torch.cat([torch.zeros_like(cdf[..., :1]),
                                      torch.cumsum(enhanced_slope, -1)], -1)

            # 平滑过渡
            alpha = 0.7  # 混合因子
            cdf = alpha * enhanced_cdf + (1.0 - alpha) * cdf

        # 采样
        if det:
            u = torch.linspace(0., 1., steps=N_samples, device=device)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

        # 反演CDF
        u = u.contiguous()
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds - 1), inds - 1)
        above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples

    def render_rays(self, model, rays_o, rays_d):
        """体渲染函数: 针对微藻特性优化"""
        batch_size = rays_o.shape[0]
        device = rays_o.device

        # 生成粗采样点
        t_vals = torch.linspace(self.config.near, self.config.far, self.config.num_coarse_samples, device=device)
        z_vals = t_vals.expand(batch_size, self.config.num_coarse_samples)

        # 添加随机抖动以减轻分层伪影
        if self.config.perturb:
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * t_rand

        # 计算采样点的3D位置
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # 展平点以并行评估网络
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d.unsqueeze(1).expand(pts.shape).reshape(-1, 3)
        dirs_flat = F.normalize(dirs_flat, dim=-1)

        # 使用分块处理来减少内存使用
        raw = []
        for i in range(0, pts_flat.shape[0], self.config.model_chunk_size):
            end_i = min(i + self.config.model_chunk_size, pts_flat.shape[0])
            pts_chunk = pts_flat[i:end_i]
            dirs_chunk = dirs_flat[i:end_i]
            raw_chunk = model(pts_chunk, dirs_chunk)
            raw.append(raw_chunk)

        raw = torch.cat(raw, 0)
        coarse_raw = raw.reshape(batch_size, self.config.num_coarse_samples, 4)

        # 解析网络输出
        rgb = coarse_raw[..., :3]  # [N_rays, N_samples, 3]
        sigma = coarse_raw[..., 3]  # [N_rays, N_samples]

        # 应用微藻特定的密度调整：增强边缘表示
        if hasattr(self.config, 'is_microalgae') and self.config.is_microalgae:
            # 计算距离光线原点的距离
            dists_from_origin = torch.norm(pts - rays_o[..., None, :], dim=-1)

            # 估计微藻边缘的位置 - 通常在中间区域
            normalized_dists = (dists_from_origin - self.config.near) / (self.config.far - self.config.near)
            edge_mask = torch.exp(-((normalized_dists - 0.5) ** 2) / 0.01)  # 高斯分布，在0.5处峰值

            # 增强边缘区域的密度
            sigma = sigma * (1.0 + edge_mask * 0.5)

        # 确保密度非负
        sigma = F.softplus(sigma)

        # 计算沿光线的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], -1)

        # Alpha合成
        alpha = 1.0 - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((batch_size, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

        # 计算粗模型的RGB和深度
        coarse_rgb_map = torch.sum(weights[..., None] * rgb, -2)
        coarse_depth_map = torch.sum(weights * z_vals, -1)
        coarse_acc_map = torch.sum(weights, -1)

        outputs = {
            'coarse': {
                'rgb': coarse_rgb_map,  # 粗采样的RGB
                'depth': coarse_depth_map,  # 粗采样的深度
                'acc': coarse_acc_map,  # 粗采样的不透明度累积
                'weights': weights,  # 权重
                'z_vals': z_vals,  # 采样点
                'sigma_avg': sigma.mean().item(),  # 平均密度
            }
        }

        # 如果使用分层采样，则进行第二次采样
        if self.config.use_hierarchical and self.config.num_fine_samples > 0:
            # 根据权重分布重新采样
            z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            fine_z_vals = self.sample_pdf(
                z_vals_mid, weights[..., 1:-1], self.config.num_fine_samples, det=(not self.config.perturb))
            fine_z_vals = fine_z_vals.detach()

            # 将细采样点与粗采样点合并并排序
            z_vals_combined, _ = torch.sort(torch.cat([z_vals, fine_z_vals], -1), -1)

            # 在合并后的点上重新计算
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
            pts_flat = pts.reshape(-1, 3)
            dirs_flat = rays_d.unsqueeze(1).expand(pts.shape).reshape(-1, 3)
            dirs_flat = F.normalize(dirs_flat, dim=-1)

            # 使用分块处理
            raw = []
            total_samples = self.config.num_coarse_samples + self.config.num_fine_samples
            for i in range(0, pts_flat.shape[0], self.config.model_chunk_size):
                end_i = min(i + self.config.model_chunk_size, pts_flat.shape[0])
                pts_chunk = pts_flat[i:end_i]
                dirs_chunk = dirs_flat[i:end_i]
                raw_chunk = model(pts_chunk, dirs_chunk)
                raw.append(raw_chunk)

            raw = torch.cat(raw, 0)
            fine_raw = raw.reshape(batch_size, total_samples, 4)

            # 解析网络输出
            rgb = fine_raw[..., :3]
            sigma = fine_raw[..., 3]

            # 针对微藻的密度调整 - 类似粗采样调整
            if hasattr(self.config, 'is_microalgae') and self.config.is_microalgae:
                dists_from_origin = torch.norm(pts - rays_o[..., None, :], dim=-1)
                normalized_dists = (dists_from_origin - self.config.near) / (self.config.far - self.config.near)
                edge_mask = torch.exp(-((normalized_dists - 0.5) ** 2) / 0.01)
                sigma = sigma * (1.0 + edge_mask * 0.5)

            sigma = F.softplus(sigma)

            # 重新计算距离和权重
            dists = z_vals_combined[..., 1:] - z_vals_combined[..., :-1]
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], -1)

            alpha = 1.0 - torch.exp(-sigma * dists)
            weights = alpha * torch.cumprod(
                torch.cat([torch.ones((batch_size, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

            # 计算细模型的最终RGB和深度
            rgb_map = torch.sum(weights[..., None] * rgb, -2)
            depth_map = torch.sum(weights * z_vals_combined, -1)
            acc_map = torch.sum(weights, -1)

            # 微藻颜色增强 - 增强饱和度
            if hasattr(self.config, 'is_microalgae') and self.config.is_microalgae:
                # 简单的颜色增强
                sat_factor = 1.2  # 饱和度增强因子

                # 计算灰度值
                gray = 0.299 * rgb_map[..., 0] + 0.587 * rgb_map[..., 1] + 0.114 * rgb_map[..., 2]
                gray = gray.unsqueeze(-1).repeat(1, 3)

                # 调整饱和度
                rgb_map = rgb_map + sat_factor * (rgb_map - gray)
                rgb_map = torch.clamp(rgb_map, 0.0, 1.0)

            outputs['fine'] = {
                'rgb': rgb_map,  # 细采样的RGB
                'depth': depth_map,  # 细采样的深度
                'acc': acc_map,  # 不透明度累积
                'weights': weights,  # 权重
                'z_vals': z_vals_combined,  # 采样点
                'sigma_avg': sigma.mean().item(),  # 平均密度
            }

        return outputs

    def render_image(self, model, H, W, focal, c2w):
        """渲染完整图像，使用分块处理"""
        rays_o, rays_d = generate_rays(H, W, focal, c2w)
        rays_o = rays_o.to(self.config.device)
        rays_d = rays_d.to(self.config.device)

        # 将光线展平
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # 使用分块处理来渲染图像
        all_rgb_coarse = []
        all_depth_coarse = []
        all_rgb_fine = []
        all_depth_fine = []

        for i in range(0, rays_o.shape[0], self.config.chunk_size):
            # 清除先前的缓存
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()

            end_i = min(i + self.config.chunk_size, rays_o.shape[0])
            results = self.render_rays(
                model,
                rays_o[i:end_i],
                rays_d[i:end_i]
            )

            # 收集结果
            all_rgb_coarse.append(results['coarse']['rgb'].cpu())
            all_depth_coarse.append(results['coarse']['depth'].cpu())

            if 'fine' in results:
                all_rgb_fine.append(results['fine']['rgb'].cpu())
                all_depth_fine.append(results['fine']['depth'].cpu())

            # 显式清理不再需要的张量
            del results
            if self.config.device.type == 'cuda':
                torch.cuda.empty_cache()

        # 合并所有结果
        rgb_coarse = torch.cat(all_rgb_coarse, 0).reshape(H, W, 3)
        depth_coarse = torch.cat(all_depth_coarse, 0).reshape(H, W)

        result = {
            'coarse': {
                'rgb': rgb_coarse,
                'depth': depth_coarse
            }
        }

        if len(all_rgb_fine) > 0:
            rgb_fine = torch.cat(all_rgb_fine, 0).reshape(H, W, 3)
            depth_fine = torch.cat(all_depth_fine, 0).reshape(H, W)

            result['fine'] = {
                'rgb': rgb_fine,
                'depth': depth_fine
            }

        return result