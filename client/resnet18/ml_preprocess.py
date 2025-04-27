from PIL import Image
import os
import pandas as pd
from torchvision import transforms
import torch
from tqdm import tqdm


def preprocess_image(image_dir, csv_path, output_dir, df_train_path):

    df_train = pd.read_csv(df_train_path)

    sample_img_path = os.path.join(image_dir, df_train['Image Index'].iloc[0].strip())
    img = Image.open(sample_img_path)

    # 输出图像属性
    print(f"图像路径: {sample_img_path}")
    print(f"图像格式: {img.format}")
    print(f"图像尺寸: {img.size} (宽 x 高)")
    print(f"图像模式: {img.mode}")  # L=灰度, RGB=彩色

    # output directory
    os.makedirs(output_dir, exist_ok=True)

    # ========== 加载图像列表 ==========
    df = pd.read_csv(csv_path)
    image_names = df['Image Index'].tolist()

    # ========== 计算你自己图像的 mean/std ==========
    # 用前1000张图像估算
    compute_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    n_images = 0
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)

    for img_name in tqdm(image_names[:1000], desc="统计 mean/std"):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('L')
            tensor = compute_transform(img)
            channel_sum += tensor.sum(dim=[1, 2])
            channel_squared_sum += (tensor ** 2).sum(dim=[1, 2])
            n_images += 1
        except:
            continue

    mean = channel_sum / (n_images * 224 * 224)
    std = (channel_squared_sum / (n_images * 224 * 224) - mean ** 2).sqrt()

    print("✅ 图像数量:", n_images)
    print("🎯 自定义 mean:", mean.tolist())
    print("🎯 自定义 std :", std.tolist())

    final_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # ========== 开始转换并保存 ==========
    for img_name in tqdm(image_names, desc="预处理并保存图像"):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('L')
            tensor = final_transform(img)  # [3, 224, 224]
            torch.save(tensor, os.path.join(
                output_dir, img_name.replace('.png', '.pt')))
        except Exception as e:
            print(f"❌ 处理失败: {img_name} → {e}")

    print("✅ 所有图像预处理完成，保存在：", output_dir)

    # 找到任意一张图像
    pt_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
    sample_path = os.path.join(output_dir, pt_files[0])

    # 加载张量
    img_tensor = torch.load(sample_path)

    # 打印格式信息
    print(f"✅ 文件名：{pt_files[0]}")
    print(f"📐 张量形状：{img_tensor.shape}")        # 应该是 [3, 224, 224]
    print(f"📦 数据类型：{img_tensor.dtype}")       # 应该是 float32
    print(f"📈 最大值：{img_tensor.max().item():.4f}")
    print(f"📉 最小值：{img_tensor.min().item():.4f}")
    print(f"📊 均值：{img_tensor.mean().item():.4f}")
    print(f"📊 标准差：{img_tensor.std().item():.4f}")