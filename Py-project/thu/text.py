import os
import torch
import clip
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


image_folder = "/home/future/Py-project/thu/photo/"

# 定义文本描述
text_descriptions = ["这是一只大老虎", "这是一只大熊猫", "这是一只小黄猫"] 
# 分布式训练函数
def train_process(rank, world_size):
    # 设置分布式环境的主节点地址和端口
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')  # 定义设备为当前进程的GPU


    model, preprocess = clip.load("ViT-B/32", device=device)

    # 包装模型为DistributedDataParallel，实现多GPU训练
    ddp_model = DDP(model, device_ids=[rank])

    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # 加载并预处理文本描述
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
            # 通过访问 DDP 包装器的底层模型来调用 encode_image 和 encode_text
            image_features = ddp_model.module.encode_image(image)
            text_features = ddp_model.module.encode_text(text)

        # 对特征进行归一化
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 计算相似度
        logits_per_image, logits_per_text = ddp_model.module(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().detach().numpy()


        if rank == 0:
            print(f"Image: {image_file}")
            print("Label probs:", probs)
            print("-" * 50)

    # 清理分布式环境
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()  

    mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()  
