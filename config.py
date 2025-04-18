import torch


class Config:
    # 数据配置
    data_dir = "./data"
    image_size = 512  # 降低图像大小，从256降到128

    # 模型配置
    pos_encoding_dims = 10  # 减少位置编码维度(从10降到6)
    dir_encoding_dims = 4  # 减少方向编码维度(从4降到2)
    hidden_dims = 256  # 减少隐藏层维度(从256降到128)
    num_layers = 8  # 减少MLP层数(从8降到6)

    # 添加分层采样配置
    num_coarse_samples = 64  # 粗采样点数
    num_fine_samples = 64 # 细采样点数
    use_hierarchical = True  # 是否使用分层采样
    perturb = True  # 是否添加随机扰动到采样点

    # 体渲染配置
    near = 0.01  # 近平面距离
    far = 3.0  # 远平面距离

    # 内存优化配置
    chunk_size = 1024  # 分块处理光线的大小
    model_chunk_size = 4096  # 模型处理点的分块大小

    # 训练配置
    batch_size = 1024  # 减少光线批量大小(从1024降到512)
    lr = 5e-4  # 学习率
    num_epochs = 200  # 训练轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 三视图特定参数
    views = ["front", "side", "top"]  # 三个视图名称

    # 可视化配置
    render_size = 512  # 减少渲染图像大小(从256降到128)
    save_dir = "./results"  # 结果保存目录