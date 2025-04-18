import torch
import os


class MicroalgaeConfig:
    # 数据配置
    data_dir = "./data"
    image_size = 512
    views = ["front", "side", "top"]

    # 模型配置
    pos_encoding_dims = 10  # 位置编码维度
    dir_encoding_dims = 4 # 方向编码维度
    hidden_dims = 256 # 隐藏层维度
    num_layers = 6  # 隐藏层数量

    # 体渲染配置
    num_coarse_samples = 64  # 粗采样点数
    num_fine_samples = 96  # 细采样点数，微藻需要更多细采样点
    use_hierarchical = True  # 使用分层采样
    perturb = True  # 添加随机扰动

    # 微藻特殊参数
    is_microalgae = True  # 标记为微藻样本
    transparency_weight = 0.7  # 透明度权重
    edge_sampling_boost = 2.0  # 边缘采样增强因子
    color_enhancement = True  # 开启颜色增强

    # 体素边界，微藻结构较小，范围要小一些
    near = 0.5  # 近平面
    far = 1.5  # 远平面

    # 训练配置
    batch_size = 512
    lr = 5e-4
    num_epochs = 250  # 微藻需要更多轮次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 分块处理配置
    chunk_size = 1024  # 光线分块大小
    model_chunk_size = 4096  # 模型分块大小

    # 渲染配置
    render_size = 512
    save_dir = "./microalgae_results"

    # 学习率调度配置
    lr_decay_steps = 50
    lr_decay_gamma = 0.5

    # 初始化函数
    def __init__(self, **kwargs):
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"更新配置 {key}: {value}")
            else:
                print(f"警告: 未知配置项 {key}")

        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)