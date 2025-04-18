import torch


class Config:
    # 数据配置
    data_dir = "./data"
    image_size = 256

    # 模型配置
    pos_encoding_dims = 10  # 位置编码维度
    dir_encoding_dims = 4  # 方向编码维度
    hidden_dims = 256  # 隐藏层维度
    num_layers = 8  # MLP层数

    # 体渲染配置
    num_samples = 64  # 每条光线上的采样点数
    near = 0.1  # 近平面距离
    far = 6.0  # 远平面距离

    # 训练配置
    batch_size = 1024  # 光线批量大小
    lr = 5e-4  # 学习率
    num_epochs = 200  # 训练轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 三视图特定参数
    views = ["front", "side", "top"]  # 三个视图名称

    # 可视化配置
    render_size = 256  # 渲染图像大小
    save_dir = "./results"  # 结果保存目录