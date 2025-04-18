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

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 损失函数
    mse_loss = nn.MSELoss()

    # 启用混合精度训练（节省内存并加速）
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # 训练循环
    print("Starting training with PDF sampling strategy...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0
        coarse_loss_total = 0
        fine_loss_total = 0

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

            # 初始化损失变量，确保它们总是被定义
            loss = torch.tensor(0.0, device=device)
            coarse_loss = torch.tensor(0.0, device=device)
            fine_loss = torch.tensor(0.0, device=device)

            try:
                # 使用混合精度训练
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 渲染
                        results = renderer.render_rays(model, rays_o, rays_d)

                        # 计算损失
                        coarse_loss = mse_loss(results['coarse']['rgb'], target_rgb)

                        if 'fine' in results:
                            fine_loss = mse_loss(results['fine']['rgb'], target_rgb)
                            loss = coarse_loss + fine_loss
                        else:
                            loss = coarse_loss

                    # 更新梯度
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # 渲染
                    results = renderer.render_rays(model, rays_o, rays_d)

                    # 计算损失
                    coarse_loss = mse_loss(results['coarse']['rgb'], target_rgb)

                    if 'fine' in results:
                        fine_loss = mse_loss(results['fine']['rgb'], target_rgb)
                        loss = coarse_loss + fine_loss
                    else:
                        loss = coarse_loss

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # 记录损失
                epoch_loss += loss.item()
                coarse_loss_total += coarse_loss.item()
                fine_loss_total += fine_loss.item()

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
                      f"Coarse: {coarse_loss.item():.6f}, Fine: {fine_loss.item():.6f}")
                print(f"平均密度值: {sigma_info:.6f}")

                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_coarse_loss = coarse_loss_total / len(dataloader)
        avg_fine_loss = fine_loss_total / len(dataloader)

        print(f"Epoch {epoch + 1} completed. Total Loss: {avg_loss:.6f}, "
              f"Coarse: {avg_coarse_loss:.6f}, Fine: {avg_fine_loss:.6f}, "
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

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")


def test(model_path):
    # 加载配置
    config = Config()

    # 设置设备
    device = config.device

    # 清理GPU内存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # 创建模型
    model = TriViewNeRF(config).to(device)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建渲染器
    renderer = VolumeRenderer(config)

    # 设置测试目录
    test_dir = os.path.join(config.save_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    # 生成360度旋转视频 - 减少帧数以节省内存
    print("Generating 360° rotation video...")
    frames = []
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size

    num_frames = 18  # 减少帧数，每20度一帧

    for i in tqdm(range(num_frames)):
        # 清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        angle = i * (360 / num_frames) * np.pi / 180

        # 创建旋转矩阵
        c2w = torch.tensor([
            [np.cos(angle), 0, -np.sin(angle), 0],
            [0, 1, 0, 0],
            [np.sin(angle), 0, np.cos(angle), -2],  # 更短的距离，使对象更清晰
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, c2w)

        # 使用细采样结果(如果有的话)，否则使用粗采样结果
        if 'fine' in results:
            rgb = results['fine']['rgb']
        else:
            rgb = results['coarse']['rgb']

        # 保存当前帧
        frame_path = os.path.join(test_dir, f"rotation_{i:03d}.png")
        rgb_np = rgb.detach().cpu().numpy()
        rgb_np = np.clip(rgb_np, 0, 1)
        rgb_np = (rgb_np * 255).astype(np.uint8)

        try:
            import imageio
            imageio.imwrite(frame_path, rgb_np)
            frames.append(rgb_np)
        except ImportError:
            from PIL import Image
            Image.fromarray(rgb_np).save(frame_path)
            frames.append(rgb_np)

        # 清理内存
        del results, rgb, rgb_np
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 创建视频
    try:
        import imageio
        video_path = os.path.join(test_dir, "360_rotation.mp4")

        try:
            # 尝试使用 imageio-ffmpeg
            writer = imageio.get_writer(video_path, fps=10, codec='libx264',
                                        pixelformat='yuv420p', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            print(f"MP4创建失败: {e}")
            print("尝试创建GIF...")
            # 备选方案：创建GIF
            gif_path = os.path.join(test_dir, "360_rotation.gif")
            imageio.mimsave(gif_path, frames, fps=10)
            print(f"创建GIF成功: {gif_path}")

        print(f"保存旋转视频到 {video_path}")
    except ImportError:
        print("imageio未找到。安装 'pip install imageio imageio-ffmpeg' 以创建视频。")


def render_novel_views(model_path, num_views=4):
    """渲染额外的新视角"""
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

    # 设置输出目录
    novel_dir = os.path.join(config.save_dir, "novel_views")
    os.makedirs(novel_dir, exist_ok=True)

    # 设置参数
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size

    # 定义不同的新视角
    novel_poses = []

    # 水平旋转视角
    for i in range(num_views):
        angle = i * (360 / num_views) * np.pi / 180
        novel_poses.append({
            "name": f"horizontal_{i:02d}",
            "c2w": torch.tensor([
                [np.cos(angle), 0, -np.sin(angle), 0],
                [0, 1, 0, 0],
                [np.sin(angle), 0, np.cos(angle), -2],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        })

    # 垂直旋转视角
    for i in range(num_views):
        angle = (i * (180 / (num_views - 1)) - 90) * np.pi / 180
        novel_poses.append({
            "name": f"vertical_{i:02d}",
            "c2w": torch.tensor([
                [1, 0, 0, 0],
                [0, np.cos(angle), -np.sin(angle), 0],
                [0, np.sin(angle), np.cos(angle), -2],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        })

    # 渲染每个视角
    for pose in tqdm(novel_poses, desc="Rendering novel views"):
        # 清理GPU内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, pose["c2w"])

        # 使用细采样结果(如果有的话)，否则使用粗采样结果
        if 'fine' in results:
            rgb = results['fine']['rgb']
            depth = results['fine']['depth']
        else:
            rgb = results['coarse']['rgb']
            depth = results['coarse']['depth']

        # 保存RGB图像
        rgb_path = os.path.join(novel_dir, f"{pose['name']}_rgb.png")
        rgb_np = rgb.detach().cpu().numpy()
        rgb_np = np.clip(rgb_np, 0, 1)
        rgb_np = (rgb_np * 255).astype(np.uint8)

        try:
            import imageio
            imageio.imwrite(rgb_path, rgb_np)
        except ImportError:
            from PIL import Image
            Image.fromarray(rgb_np).save(rgb_path)

        # 保存深度图
        depth_path = os.path.join(novel_dir, f"{pose['name']}_depth.png")
        depth_np = depth.detach().cpu().numpy()
        depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
        depth_np = (depth_np * 255).astype(np.uint8)

        try:
            import imageio
            imageio.imwrite(depth_path, depth_np)
        except ImportError:
            from PIL import Image
            Image.fromarray(depth_np).save(depth_path)

        # 清理内存
        del results, rgb, depth, rgb_np, depth_np
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"已保存 {len(novel_poses)} 个新视角渲染结果到 {novel_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TriView NeRF with PDF Sampling")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'render_novel'],
                        help='运行模式: train(训练), test(测试), render_novel(渲染新视角)')
    parser.add_argument('--model', type=str, default=None,
                        help='用于测试的模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置中的批量大小')
    parser.add_argument('--num_views', type=int, default=8,
                        help='渲染新视角的数量')

    args = parser.parse_args()

    if args.batch_size is not None:
        # 覆盖配置中的批量大小
        Config.batch_size = args.batch_size
        print(f"覆盖批量大小: {args.batch_size}")

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