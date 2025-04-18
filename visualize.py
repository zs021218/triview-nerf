import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from utils import get_view_matrix


def save_image(img_tensor, path):
    """保存图像张量为PNG文件"""
    # 从GPU移动到CPU并转换为numpy数组
    img = img_tensor.detach().cpu().numpy()

    # 确保值在[0, 1]范围内
    img = np.clip(img, 0, 1)

    # 转换为uint8格式
    img = (img * 255).astype(np.uint8)

    # 保存图像
    Image.fromarray(img).save(path)
    print(f"Saved image to {path}")


def visualize_results(model, renderer, config, epoch, novel_views=True):
    """可视化结果和保存"""
    model.eval()
    H, W = config.render_size, config.render_size
    focal = 0.5 * config.render_size

    # 确保结果目录存在
    os.makedirs(os.path.join(config.save_dir, f"epoch_{epoch}"), exist_ok=True)

    # 渲染源视图
    for view in config.views:
        c2w = get_view_matrix(view)
        with torch.no_grad():
            results = renderer.render_image(model, H, W, focal, c2w)

        rgb = results['rgb']
        depth = results['depth']

        # 保存RGB和深度图
        save_image(rgb, os.path.join(config.save_dir, f"epoch_{epoch}", f"{view}_rgb.png"))

        # 归一化深度进行可视化
        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        save_image(depth_vis.unsqueeze(-1).repeat(1, 1, 3),
                   os.path.join(config.save_dir, f"epoch_{epoch}", f"{view}_depth.png"))

    # 渲染新视角 (如果需要)
    if novel_views:
        # 生成一些新视角
        novel_views = [
            {"name": "novel_45", "rot_y": 45},
            {"name": "novel_90", "rot_y": 90},
            {"name": "novel_135", "rot_y": 135},
            {"name": "novel_180", "rot_y": 180}
        ]

        for view in novel_views:
            angle = view["rot_y"] * np.pi / 180

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
            depth = results['depth']

            # 保存RGB和深度图
            save_image(rgb, os.path.join(config.save_dir, f"epoch_{epoch}", f"{view['name']}_rgb.png"))

            # 归一化深度进行可视化
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            save_image(depth_vis.unsqueeze(-1).repeat(1, 1, 3),
                       os.path.join(config.save_dir, f"epoch_{epoch}", f"{view['name']}_depth.png"))

    model.train()


def create_video(config, view_name="novel", fps=30):
    """从渲染的帧创建视频"""
    try:
        import imageio
        import glob

        # 找到所有相关帧
        frames_pattern = os.path.join(config.save_dir, "epoch_*", f"{view_name}*_rgb.png")
        frame_files = sorted(glob.glob(frames_pattern))

        if len(frame_files) == 0:
            print(f"No frames found matching pattern: {frames_pattern}")
            return

        # 读取所有帧
        frames = [imageio.imread(file) for file in frame_files]

        # 创建视频 - 确保使用正确的扩展名和视频写入器
        video_path = os.path.join(config.save_dir, f"{view_name}_progress.mp4")

        try:
            # 尝试使用 imageio-ffmpeg (首选)
            import imageio_ffmpeg
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264',
                                        pixelformat='yuv420p', quality=8)
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        except (ImportError, ValueError) as e:
            print(f"Could not use ffmpeg: {e}")
            print("Trying alternative method...")

            # 备选方法: 使用 pillow
            try:
                import numpy as np
                from PIL import Image

                # 创建临时GIF (大多数系统都支持)
                gif_path = os.path.join(config.save_dir, f"{view_name}_progress.gif")
                with imageio.get_writer(gif_path, mode='I', duration=1000 / fps) as writer:
                    for frame in frames:
                        writer.append_data(frame)
                print(f"Created GIF at {gif_path} (MP4 creation failed)")
                return

            except Exception as e2:
                print(f"GIF creation also failed: {e2}")
                print("Saving individual frames only.")
                return

        print(f"Created video at {video_path}")

    except ImportError as e:
        print(f"Required libraries not found: {e}")
        print("Install with 'pip install imageio imageio-ffmpeg' to create videos.")