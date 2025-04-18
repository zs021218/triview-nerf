import torch
import numpy as np
import os


def create_dirs(dirs):
    """创建目录列表"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def pos_encoding(x, L):
    """对输入进行位置编码"""
    encode = []
    for i in range(L):
        encode.append(torch.sin(2 ** i * np.pi * x))
        encode.append(torch.cos(2 ** i * np.pi * x))
    return torch.cat([x, *encode], dim=-1)


def get_view_matrix(view_type, distance=1.5):
    """为微藻生成视图矩阵 - 调整距离使其更合适"""
    if view_type == "front":
        # 正面视图
        c2w = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    elif view_type == "side":
        # 侧面视图
        c2w = torch.tensor([
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    elif view_type == "top":
        # 顶视图
        c2w = torch.tensor([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, -distance],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError(f"未知视图类型: {view_type}")

    return c2w


def generate_rays(H, W, focal, c2w):
    """生成采样光线"""
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W),
        torch.linspace(0, H - 1, H)
    )
    i = i.t()
    j = j.t()

    # 计算方向向量
    directions = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    # 转换到世界坐标系
    rot_matrix = c2w[:3, :3]
    ray_directions = torch.sum(directions[..., None, :] * rot_matrix[:, None, None], dim=-1)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True)

    # 获取相机位置
    ray_origins = c2w[:3, -1].expand(ray_directions.shape)

    return ray_origins, ray_directions


def create_circular_mask(h, w, center=None, radius=None):
    """为微藻创建圆形掩码"""
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def preprocess_microalgae_image(image, enhance_contrast=True):
    """预处理微藻图像，增强对比度并去噪"""
    import cv2

    # 转为灰度图计算阈值
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # 寻找圆形藻类区域
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    # 创建掩码
    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # 绘制圆形掩码
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
    else:
        # 如果没有检测到圆，使用默认圆
        h, w = image.shape[:2]
        mask = create_circular_mask(h, w)
        mask = mask.astype(np.uint8) * 255

    # 应用掩码
    if len(image.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # 仅保留藻类区域
    result = cv2.bitwise_and(image, mask)

    if enhance_contrast:
        # 增强对比度
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        updated_lab = cv2.merge((cl, a, b))
        result = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)

    return result


def rgb_to_hue(rgb):
    """从RGB计算色调，用于颜色保持损失"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    max_c = torch.max(rgb, dim=-1)[0]
    min_c = torch.min(rgb, dim=-1)[0]

    delta = max_c - min_c

    # 避免除零
    delta = torch.where(delta < 1e-6, torch.ones_like(delta), delta)

    hue = torch.zeros_like(r)

    # R是最大值
    mask_r = torch.eq(max_c, r)
    hue = torch.where(mask_r, (g - b) / delta % 6, hue)

    # G是最大值
    mask_g = torch.eq(max_c, g)
    hue = torch.where(mask_g, (b - r) / delta + 2, hue)

    # B是最大值
    mask_b = torch.eq(max_c, b)
    hue = torch.where(mask_b, (r - g) / delta + 4, hue)

    return hue / 6  # 归一化到[0,1]


def animate_training(image_dir, output_gif, pattern="*_rgb.png", fps=5):
    """创建训练过程动画"""
    import glob
    import imageio
    from PIL import Image

    # 找到所有匹配的图像
    files = sorted(glob.glob(f"{image_dir}/{pattern}"))

    if not files:
        print(f"在 {image_dir} 中未找到匹配 {pattern} 的文件")
        return

    # 读取图像
    frames = []
    for file in files:
        im = Image.open(file)
        frames.append(np.array(im))

    # 创建GIF
    print(f"创建GIF，包含 {len(frames)} 帧")
    imageio.mimsave(output_gif, frames, fps=fps)
    print(f"GIF已保存到 {output_gif}")