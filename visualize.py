import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from utils import get_view_matrix, create_circular_mask
from dataset import generate_intermediate_views


def save_image(img_tensor, path):
    """保存图像张量为PNG文件"""
    # 从GPU移动到CPU并转换为numpy数组
    img = img_tensor.detach().cpu().numpy()

    # 确保值在[0, 1]范围内
    img = np.clip(img, 0, 1)

    # 转换为uint8格式
    img = (img * 255).astype(np.uint8)

    # 微藻图像特殊处理 - 应用圆形掩码
    if "microalgae" in path:
        h, w = img.shape[:2]
        mask = create_circular_mask(h, w)
        if len(img.shape) == 3:
            mask = mask[..., np.newaxis]
        img = img * mask

    # 保存图像
    Image.fromarray(img).save(path)
    print(f"保存图像: {path}")


def enhance_microalgae_rendering(img_tensor):
    """增强微藻渲染效果"""
    img = img_tensor.clone()

    # 创建圆形掩码
    h, w = img.shape[:2]
    mask = create_circular_mask(h, w)
    mask_tensor = torch.from_numpy(mask).to(img.device).float()

    # 扩展掩码到3通道
    if len(mask_tensor.shape) == 2 and len(img.shape) == 3:
        mask_tensor = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)

    # 增强对比度和饱和度
    brightness = 1.1
    contrast = 1.2
    saturation = 1.3

    # 饱和度调整
    rgb_mean = img.mean(dim=-1, keepdim=True)
    img = (img - rgb_mean) * saturation + rgb_mean

    # 对比度调整
    img = (img - 0.5) * contrast + 0.5

    # 亮度调整
    img = img * brightness

    # 仅应用于圆形区域内
    img = img * mask_tensor

    # 裁剪到有效范围
    img = torch.clamp(img, 0.0, 1.0)

    return img


def visualize_results(model, renderer, config, epoch, novel_views=True):
    """可视化粗细两个模型的结果，针对微藻做特殊处理"""
    model.eval()
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size

    # 确保结果目录存在
    epoch_dir = os.path.join(config.save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # 渲染源视图
    for view in config.views:
        c2w = get_view_matrix(view, distance=config.far * 0.8)
        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, c2w)

        # 保存粗采样结果
        rgb_coarse = results['coarse']['rgb']
        depth_coarse = results['coarse']['depth']

        # 对微藻图像进行增强
        if hasattr(config, 'is_microalgae') and config.is_microalgae:
            rgb_coarse = enhance_microalgae_rendering(rgb_coarse)

        save_image(rgb_coarse, os.path.join(epoch_dir, f"microalgae_{view}_coarse_rgb.png"))

        # 归一化深度进行可视化
        depth_vis = (depth_coarse - depth_coarse.min()) / (depth_coarse.max() - depth_coarse.min() + 1e-8)
        # 应用圆形掩码到深度图
        h, w = depth_vis.shape
        mask = create_circular_mask(h, w)
        depth_vis = depth_vis * torch.from_numpy(mask).float()
        save_image(depth_vis.unsqueeze(-1).repeat(1, 1, 3),
                   os.path.join(epoch_dir, f"microalgae_{view}_coarse_depth.png"))

        # 如果有细采样结果，也保存它们
        if 'fine' in results:
            rgb_fine = results['fine']['rgb']
            depth_fine = results['fine']['depth']

            # 对微藻图像进行增强
            if hasattr(config, 'is_microalgae') and config.is_microalgae:
                rgb_fine = enhance_microalgae_rendering(rgb_fine)

            save_image(rgb_fine, os.path.join(epoch_dir, f"microalgae_{view}_fine_rgb.png"))

            depth_vis = (depth_fine - depth_fine.min()) / (depth_fine.max() - depth_fine.min() + 1e-8)
            # 应用圆形掩码到深度图
            depth_vis = depth_vis * torch.from_numpy(mask).float()
            save_image(depth_vis.unsqueeze(-1).repeat(1, 1, 3),
                       os.path.join(epoch_dir, f"microalgae_{view}_fine_depth.png"))

    # 渲染新视角
    if novel_views:
        if hasattr(config, 'is_microalgae') and config.is_microalgae:
            # 使用针对微藻优化的视角生成
            novel_views = generate_intermediate_views(config)
        else:
            # 标准新视角
            novel_views = [
                {"name": "novel_45", "rot_y": 45},
                {"name": "novel_90", "rot_y": 90},
                {"name": "novel_135", "rot_y": 135},
                {"name": "novel_180", "rot_y": 180}
            ]

        for view in novel_views[:8]:  # 限制为前8个视角以加快处理
            if isinstance(view, dict) and "rot_y" in view:
                angle = view["rot_y"] * np.pi / 180
                name = view["name"]

                # 创建旋转矩阵
                c2w = torch.tensor([
                    [np.cos(angle), 0, -np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [np.sin(angle), 0, np.cos(angle), -config.far * 0.8],
                    [0, 0, 0, 1]
                ], dtype=torch.float32)
            else:
                name = view["name"]
                c2w = view["c2w"]

            with torch.no_grad():
                results = renderer.render_image(model, H, W, focal, c2w)

            # 保存结果
            if 'fine' in results:
                rgb = results['fine']['rgb']
                # 微藻特殊处理
                if hasattr(config, 'is_microalgae') and config.is_microalgae:
                    rgb = enhance_microalgae_rendering(rgb)
                save_image(rgb, os.path.join(epoch_dir, f"microalgae_{name}_fine_rgb.png"))
            else:
                rgb = results['coarse']['rgb']
                # 微藻特殊处理
                if hasattr(config, 'is_microalgae') and config.is_microalgae:
                    rgb = enhance_microalgae_rendering(rgb)
                save_image(rgb, os.path.join(epoch_dir, f"microalgae_{name}_coarse_rgb.png"))

    # 创建比较图 - 原始视图与重建视图
    for view in config.views:
        if 'fine' in results:
            rendered = os.path.join(epoch_dir, f"microalgae_{view}_fine_rgb.png")
        else:
            rendered = os.path.join(epoch_dir, f"microalgae_{view}_coarse_rgb.png")

        original_path = os.path.join(config.data_dir, f"{view}.png")

        if os.path.exists(original_path) and os.path.exists(rendered):
            original = np.array(Image.open(original_path))
            rendered_img = np.array(Image.open(rendered))

            # 调整大小以匹配
            from skimage.transform import resize
            if original.shape[:2] != rendered_img.shape[:2]:
                original = resize(original, rendered_img.shape[:2], preserve_range=True).astype(np.uint8)

            # 创建比较图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(original)
            ax1.set_title(f"原始 {view}")
            ax1.axis('off')

            ax2.imshow(rendered_img)
            ax2.set_title(f"重建 {view} (轮次 {epoch})")
            ax2.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f"microalgae_{view}_比较.png"), dpi=150)
            plt.close()

    model.train()


def create_video(config, view_pattern="rot_*", output_name="microalgae_rotation", fps=15):
    """从渲染的帧创建视频"""
    try:
        import imageio
        import glob

        # 找到最后一个epoch或指定epoch的旋转视图
        epoch_dirs = sorted(glob.glob(os.path.join(config.save_dir, "epoch_*")))
        if not epoch_dirs:
            print("未找到渲染结果")
            return

        # 使用最后一个epoch
        latest_epoch = epoch_dirs[-1]

        # 查找所有旋转视图
        frame_pattern = os.path.join(latest_epoch, f"microalgae_{view_pattern}_fine_rgb.png")
        frame_files = sorted(glob.glob(frame_pattern))

        if not frame_files:
            # 尝试寻找粗采样结果
            frame_pattern = os.path.join(latest_epoch, f"microalgae_{view_pattern}_coarse_rgb.png")
            frame_files = sorted(glob.glob(frame_pattern))

        if not frame_files:
            print(f"未找到匹配 {view_pattern} 的渲染帧")
            return

        # 读取所有帧
        frames = []
        for file in frame_files:
            # 使用PIL读取图像，保留圆形遮罩
            from PIL import Image
            img = np.array(Image.open(file))
            frames.append(img)

        print(f"找到 {len(frames)} 帧")

        # 创建视频
        output_path = os.path.join(config.save_dir, f"{output_name}.mp4")

        try:
            # 使用imageio-ffmpeg
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8,
                                        pixelformat='yuv420p')
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception as e:
            print(f"创建MP4失败: {e}")
            # 备选：创建GIF
            gif_path = os.path.join(config.save_dir, f"{output_name}.gif")
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"已创建GIF: {gif_path}")
            return gif_path

        print(f"已创建视频: {output_path}")
        return output_path

    except ImportError:
        print("缺少依赖库。安装 'pip install imageio imageio-ffmpeg' 以创建视频。")
        return None