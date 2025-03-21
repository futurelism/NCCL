import os
import torch
import clip
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import time
import torch.backends.cudnn as cudnn

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置数据路径
image_folder = "/home/future/Py-project/thu/photo/"  # 图片文件夹路径
checkpoint_dir = "checkpoints"  # 模型保存路径

# 定义文本描述，用于训练
text_descriptions = ["这是一片草", "这是只熊猫", "这是一只小黄猫"]


def print_gpu_info():
    """打印GPU信息，用于调试和监控"""
    logger.info("=== GPU Information ===")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"当前GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            if i > 0:  # 只显示GPU 1及以上的信息
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("没有检测到CUDA设备")
    logger.info("=====================")


def setup_cuda():
    """配置CUDA优化参数"""
    cudnn.benchmark = True  # 启用cuDNN自动调优
    cudnn.deterministic = False  
    torch.backends.cuda.matmul.allow_tf32 = True  
    torch.backends.cudnn.allow_tf32 = True  # 启用cuDNN的TF32支持


class ImageDataset(Dataset):
    """自定义数据集类，用于加载和处理图片"""
    def __init__(self, image_folder, preprocess):
        # 获取所有图片文件
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_folder = image_folder
        self.preprocess = preprocess  # CLIP的预处理函数
        logger.info(f"加载了 {len(self.image_files)} 张图片")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """加载并预处理单张图片"""
        image_path = os.path.join(self.image_folder, self.image_files[idx])
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)  # 应用CLIP的预处理
            return image, self.image_files[idx]
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None


def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'  
    os.environ['MASTER_PORT'] = '12355' 
    # 初始化进程组，使用NCCL后端
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 设置当前进程使用的GPU
    setup_cuda()  # 配置CUDA优化
    logger.info(f"进程 {rank} 使用 GPU {rank}")


def save_checkpoint(model, optimizer, epoch, rank, checkpoint_dir):
    """保存模型检查点"""
    if rank == 0:  # 只在主进程保存
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt'))
        logger.info(f"保存检查点到 {checkpoint_dir}/checkpoint_epoch_{epoch}.pt")


def train_process(rank, world_size):
    """训练进程的主要函数"""
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')

        
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        ddp_model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)

        # 创建数据加载器
        dataset = ImageDataset(image_folder, preprocess)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        
        dataloader = DataLoader(
            dataset,
            batch_size=min(32, len(dataset)),
            sampler=sampler,
            num_workers=4,  # 多进程加载数据
            pin_memory=True,  # 使用固定内存
            persistent_workers=True,  
            prefetch_factor=2  
        )

        # 设置优化器和损失函数
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-6)
        loss_fn = torch.nn.CrossEntropyLoss()

        # 预处理文本描述
        text = clip.tokenize(text_descriptions).to(device)

        # 训练循环
        for epoch in range(50):
            sampler.set_epoch(epoch)  # 设置采样器epoch
            ddp_model.train()
            total_loss = 0
            start_time = time.time()

            # 设置进度条
            if rank == 0:
                pbar = tqdm(dataloader, desc=f'Epoch {epoch}', ncols=100)
            else:
                pbar = dataloader

            # 批次训练
            for batch_idx, (images, image_names) in enumerate(pbar):
                if images is None:
                    continue

                # 将数据移到GPU
                images = images.to(device, non_blocking=True)
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

                # 前向传播
                logits_per_image, _ = ddp_model(images, text)
                loss = loss_fn(logits_per_image, ground_truth).mean()

                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)  
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)  
                optimizer.step()  

                total_loss += loss.item()

                # 更新进度条
                if rank == 0:
                    probs = torch.softmax(logits_per_image, dim=1)
                    confidence = probs.max(dim=1)[0].mean().item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.2f}',
                        'conf': f'{confidence:.2f}'
                    })

            # 计算平均损失
            avg_loss = total_loss / len(dataloader)

            
            save_checkpoint(ddp_model, optimizer, epoch, rank, checkpoint_dir)

            # 打印训练信息
            if rank == 0:
                epoch_time = time.time() - start_time
                logger.info(f'Epoch {epoch} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.4f}')

    except Exception as e:
        logger.error(f"Error in training process {rank}: {e}")
        raise e
    finally:
        dist.destroy_process_group()  # 清理分布式环境


def main():
    """主函数"""
    try:
        print_gpu_info()  
        
        world_size = torch.cuda.device_count()  
        if world_size == 0:
            raise RuntimeError("No GPU devices found!")

        logger.info(f"Starting distributed training with {world_size} GPUs")
       
        mp.spawn(train_process, args=(world_size,), nprocs=world_size, join=True)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        raise e


if __name__ == "__main__":
    main()
