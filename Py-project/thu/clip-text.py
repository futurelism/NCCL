import torch
import clip
from PIL import Image
import os

# 设置图像文件夹路径
image_folder = "/home/future/Py-project/thu/photo/"

# 获取图像文件列表
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 加载 CLIP 模型和预处理函数
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义文本描述
text_descriptions = ["这是一只老虎", "这是一只熊猫", "这是一只黄猫"]  # 替换为你的文本描述
text = clip.tokenize(text_descriptions).to(device)

# 对每个图像进行处理
for image_file in image_files:
    # 构造图像路径
    image_path = os.path.join(image_folder, image_file)

    # 加载并预处理图像
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue

    # 计算图像和文本的特征
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    # 对特征进行归一化
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()

    # 打印结果
    print(f"Image: {image_file}")
    print("Label probs:", probs)
    print("-" * 50)

