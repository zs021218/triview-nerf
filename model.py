import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
import gc

from config import Config
from model import TriViewNeRF
from renderer import VolumeRenderer
from dataset import TriViewDataset
from utils import create_dirs
from visualize import visualize_results, create_video
from losses import MicroalgaeLoss
from visualize_microalgae import visualize_microalgae_density, create_3d_visualization


def train():
    # 加载配置
    config = Config()

    # 创建目录
    create_dirs([config.save_dir])

    # 设置设备
    device = config.device
    print(f"Using device: {device}")

    # 启用cuda同步，以便更好地管理内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 创建模型
    model = TriViewNeRF(config).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # 创建渲染器
    renderer = VolumeRenderer(config)

    # 准备数据集
    dataset = TriViewDataset(config, split='train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2  # 减少工作线程数以节省内存
    )

    # 设置优化器 - 为微藻优化添加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 学习率调度器 - 使用更平滑的学习率衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # 使用微藻专用损失函数
    criterion = MicroalgaeLoss(config)

    # 启用混合精度训练（节省内存并加速）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # 训练循环
    print("Starting training with PDF sampling strategy...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        coarse_rgb_loss_total = 0
        fine_rgb_loss_total = 0

        # 在每个epoch开始时清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            # 定期清理内存
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # 将数据移动到设备
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            target_rgb = batch['rgb'].to(device)

            # 初始化损失变量
            loss = torch.tensor(0.0, device=device)

            try:
                # 使用混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 渲染
                        results = renderer.render_rays(model, rays_o, rays_d)
                        
                        # 计算损失 - 使用微藻专用损失函数
                        coarse_loss, coarse_rgb_loss = criterion(results, target_rgb, phase='coarse')
                        
                        if 'fine' in results:
                            fine_loss, fine_rgb_loss = criterion(results, target_rgb, phase='fine')
                            loss = coarse_loss + fine_loss
                        else:
                            loss = coarse_loss
                            fine_rgb_loss = torch.tensor(0.0, device=device)

                    # 更新梯度
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 渲染
                    results = renderer.render_rays(model, rays_o, rays_d)
                    
                    # 计算损失 - 使用微藻专用损失函数
                    coarse_loss, coarse_rgb_loss = criterion(results, target_rgb, phase='coarse')
                    
                    if 'fine' in results:
                        fine_loss, fine_rgb_loss = criterion(results, target_rgb, phase='fine')
                        loss = coarse_loss + fine_loss
                    else:
                        loss = coarse_loss
                        fine_rgb_loss = torch.tensor(0.0, device=device)

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                coarse_rgb_loss_total += coarse_rgb_loss.item()
                fine_rgb_loss_total += fine_rgb_loss.item()

                # 获取一些调试信息
                if 'fine' in results:
                    sigma_info = results['fine'].get('sigma_avg', 0)
                else:
                    sigma_info = results['coarse'].get('sigma_avg', 0)

            except Exception as e:
                print(f"错误发生在批次 {batch_idx}: {e}")
                # 如果发生错误，继续下一批次
                continue

            # 清理不再需要的张量
            del results, rays_o, rays_d, target_rgb
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # 打印进度
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.6f}, "
                      f"Coarse RGB: {coarse_rgb_loss.item():.6f}, Fine RGB: {fine_rgb_loss.item():.6f}")
                print(f"平均密度值: {sigma_info:.6f}")

                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_coarse_rgb_loss = coarse_rgb_loss_total / len(dataloader)
        avg_fine_rgb_loss = fine_rgb_loss_total / len(dataloader)

        print(f"Epoch {epoch + 1} completed. Total Loss: {avg_loss:.6f}, "
              f"Coarse RGB: {avg_coarse_rgb_loss:.6f}, Fine RGB: {avg_fine_rgb_loss:.6f}, "
              f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        # 每50个epoch保存一次模型和可视化
        if (epoch + 1) % 50 == 0 or epoch == 0:
            # 清理内存以便可视化
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(config.save_dir, f"model_epoch_{epoch + 1}.pth"))

            # 可视化
            with torch.no_grad():
                visualize_results(model, renderer, config, epoch + 1)

            # 清理内存
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # 最终保存
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(config.save_dir, "model_final.pth"))

    # 最终可视化
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    with torch.no_grad():
        visualize_results(model, renderer, config, "final", novel_views=True)

    # 创建进度视频
    create_video(config)
    
    # 在训练结束后添加微藻特定的可视化
    print("Generating microalgae-specific visualizations...")
    
    # 清理内存以便可视化
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        
    # 创建微藻可视化目录
    microalgae_viz_dir = os.path.join(config.save_dir, "microalgae_viz")
    os.makedirs(microalgae_viz_dir, exist_ok=True)
    
    # 可视化微藻密度
    for view in config.views:
        visualize_microalgae_density(model, renderer, config, microalgae_viz_dir, view_name=view)
    
    # 创建3D可视化
    create_3d_visualization(model, renderer, config, microalgae_viz_dir, num_views=18)
    
    print(f"Microalgae visualizations saved to {microalgae_viz_dir}")
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")


def evaluate_microalgae(model_path):
    """评估微藻重建质量"""
    # 加载配置
    config = Config()
    device = config.device
    
    # 创建模型并加载权重
    model = TriViewNeRF(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 创建渲染器
    renderer = VolumeRenderer(config)
    
    # 设置评估目录
    eval_dir = os.path.join(config.save_dir, "microalgae_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # 生成微藻密度分析
    print("Generating microalgae density analysis...")
    for view in config.views:
        visualize_microalgae_density(model, renderer, config, eval_dir, view_name=view)
    
    # 创建更多分析角度的3D可视化
    print("Creating detailed 3D visualization for microalgae analysis...")
    frames = create_3d_visualization(model, renderer, config, eval_dir, num_views=36)
    
    print(f"Microalgae evaluation results saved to {eval_dir}")
    return eval_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TriView NeRF with PDF Sampling for Microalgae Reconstruction")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'render_novel', 'microalgae_eval'],
                        help='运行模式: train(训练), test(测试), render_novel(渲染新视角), microalgae_eval(微藻评估)')
    parser.add_argument('--model', type=str, default=None,
                        help='用于测试的模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置中的批量大小')
    parser.add_argument('--num_views', type=int, default=8,
                        help='渲染新视角的数量')
    parser.add_argument('--transmittance_weight', type=float, default=None,
                        help='透明度损失权重(用于微藻重建)')
    parser.add_argument('--white_background', action='store_true',
                        help='使用白色背景(适用于微藻图像)')

    args = parser.parse_args()

    if args.batch_size is not None:
        # 覆盖配置中的批量大小
        Config.batch_size = args.batch_size
        print(f"覆盖批量大小: {args.batch_size}")
        
    if args.transmittance_weight is not None:
        # 覆盖透明度损失权重
        Config.transmittance_weight = args.transmittance_weight
        print(f"覆盖透明度损失权重: {args.transmittance_weight}")
        
    if args.white_background:
        # 设置白色背景
        Config.white_background = True
        print("启用白色背景")

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        if args.model is None:
            print("请使用 --model 参数提供测试用的模型路径")
        else:
            test(args.model)
    elif args.mode == 'render_novel':
        if args.model is None:
            print("请使用 --model 参数提供渲染用的模型路径")
        else:
            render_novel_views(args.model, args.num_views)
    elif args.mode == 'microalgae_eval':
        if args.model is None:
            print("请使用 --model 参数提供微藻评估用的模型路径")
        else:
            evaluate_microalgae(args.model)