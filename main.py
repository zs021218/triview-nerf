import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm

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

    # 创建模型
    model = TriViewNeRF(config).to(device)

    # 创建渲染器
    renderer = VolumeRenderer(config)

    # 准备数据集
    dataset = TriViewDataset(config, split='train')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # 损失函数
    mse_loss = nn.MSELoss()

    # 训练循环
    print("Starting training...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")):
            optimizer.zero_grad()

            # 将数据移动到适当的设备
            rays_o = batch['rays_o'].to(device)
            rays_d = batch['rays_d'].to(device)
            target_rgb = batch['rgb'].to(device)

            # 渲染
            results = renderer.render_rays(model, rays_o, rays_d)
            rgb = results['rgb']

            # 计算损失
            loss = mse_loss(rgb, target_rgb)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # 每100个批次打印一次进度
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.6f}")

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.6f}, "
              f"Time: {(time.time() - start_time) / 60:.2f} minutes")

        # 每10个epoch保存一次模型和可视化
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(config.save_dir, f"model_epoch_{epoch + 1}.pth"))

            # 可视化
            visualize_results(model, renderer, config, epoch + 1)

    # 最终保存
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, os.path.join(config.save_dir, "model_final.pth"))

    # 最终可视化
    visualize_results(model, renderer, config, "final", novel_views=True)

    # 创建进度视频
    create_video(config)

    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes")


def test(model_path):
    # 加载配置
    config = Config()

    # 设置设备
    device = config.device

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

    # 生成360度旋转视频
    print("Generating 360° rotation video...")
    frames = []
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size

    num_frames = 36  # 10度一帧

    for i in tqdm(range(num_frames)):
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

        import imageio
        imageio.imwrite(frame_path, rgb_np)
        frames.append(rgb_np)

    # 创建视频
    try:
        import imageio
        video_path = os.path.join(test_dir, "360_rotation.mp4")
        imageio.mimsave(video_path, frames, fps=20)
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