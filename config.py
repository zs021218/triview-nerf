import torch


class Config:
    # 数据配置
    data_dir = "./data"
    image_size = 512

    # 模型配置
    pos_encoding_dims = 10
    dir_encoding_dims = 4
    hidden_dims = 256
    num_layers = 8

    # 添加分层采样配置
    num_coarse_samples = 64
    num_fine_samples = 128  # 增加细采样点数，更好捕捉微藻细节
    use_hierarchical = True
    perturb = True

    # 体渲染配置
    near = 0.01
    far = 3.0
    density_noise_std = 0.0  # 添加密度噪声参数

    # 内存优化配置
    chunk_size = 1024
    model_chunk_size = 4096

    # 训练配置
    batch_size = 1024
    lr = 5e-4
    num_epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 三视图特定参数
    views = ["front", "side", "top"]

    # 微藻特定参数
    transmittance_weight = 0.1  # 透明度损失权重
    depth_smooth_weight = 0.05  # 深度平滑权重
    sparsity_weight = 0.01  # 稀疏性损失权重
    white_background = True  # 微藻通常在白色背景下拍摄
    
    # 正则化参数
    weight_decay = 1e-5  # 添加权重衰减防止过拟合
    
    # 可视化配置
    render_size = 512
    save_dir = "./results"