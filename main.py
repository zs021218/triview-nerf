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
    print("Starting training...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0

        # 在每个epoch开始时清理内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            # 手动清理前一批次的缓存
            if device.type == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()

            # 将数据移动到适当的设备
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            target_rgb = batch['rgb'].to(device)

            # 使用混合精度训练
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    # 渲染
                    results = renderer.render_rays(model, rays_o, rays_d)
                    rgb = results['rgb']

                    # 计算损失
                    loss = mse_loss(rgb, target_rgb)

                # 更新梯度
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # 渲染
                results = renderer.render_rays(model, rays_o, rays_d)
                rgb = results['rgb']

                # 计算损失
                loss = mse_loss(rgb, target_rgb)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            # 清理不再需要的张量
            del results, rgb, rays_o, rays_d, target_rgb

            # 每50个批次打印一次进度
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")
                if device.type == 'cuda':
                    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.6f}, "
              f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        # 每10个epoch保存一次模型和可视化
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
            [np.sin(angle), 0, np.cos(angle), -4],
            [0, 0, 0, 1]
        ], dtype=torch.float32)

        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, c2w)

        rgb = results['rgb']

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
        imageio.mimsave(video_path, frames, fps=10)
        print(f"Saved 360° rotation video to {video_path}")
    except ImportError:
        print("imageio not found. Install with 'pip install imageio imageio-ffmpeg' to create videos.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TriView NeRF")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Run mode: train or test')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint for testing')

    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        if args.model is None:
            print("Please provide a model path for testing with --model")
        else:
            test(args.model)