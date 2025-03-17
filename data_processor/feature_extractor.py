import json
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import (
    BertTokenizer, 
    BertModel, 
    BertForMaskedLM,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os
from torch import nn

def check_cuda_status():
    """检查CUDA状态并打印详细信息"""
    print("\n=== CUDA Status ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Running on CPU.")
        return False
        
    print("✓ CUDA is available!")
    
    # 获取所有可用的GPU
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} GPU(s):")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        gpu_free_memory = torch.cuda.memory_reserved(i) / 1024**3
        gpu_used_memory = torch.cuda.memory_allocated(i) / 1024**3
        
        print(f"\nGPU {i}: {gpu_name}")
        print(f"- Total Memory: {gpu_total_memory:.2f} GB")
        print(f"- Reserved Memory: {gpu_free_memory:.2f} GB")
        print(f"- Used Memory: {gpu_used_memory:.2f} GB")
    
    # 设置CUDA性能优化
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    if hasattr(torch.backends.cudnn, 'benchmark'):
        torch.backends.cudnn.benchmark = True
        
    return True

# Set up paths
VINE_DATA_FILE = 'sampled_post-comments_vine.json'
BERT_MODEL_PATH = Path('models/bert_model')
PROCESSED_DATA_PATH = Path('processed')

# Create directories if they don't exist
BERT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, anchor_features, positive_features):
        # 计算相似度
        similarity = F.cosine_similarity(anchor_features, positive_features)
        # 计算对比损失
        loss = -torch.log(torch.exp(similarity / self.temperature))
        return loss.mean()

class TextBatchDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

class BertFeatureExtractor:
    def __init__(self):
        # 检查CUDA状态
        self.has_cuda = check_cuda_status()
        
        print("\nInitializing BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        self.device = torch.device("cuda" if self.has_cuda else "cpu")
        print(f"Using device: {self.device}")
        
        if self.has_cuda:
            # 使用 DistributedDataParallel 来加速
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                self.model = torch.nn.DataParallel(self.model)
            
            # 设置为半精度训练来提高速度和减少内存使用
            self.model = self.model.half()
        
        self.model.to(self.device)
        self.model.eval()
        
        # 根据GPU显存大小调整batch_size
        if self.has_cuda:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.batch_size = int(min(256, max(32, gpu_mem * 16)))  # 根据显存动态调整
        else:
            self.batch_size = 8
            
        print(f"Batch size set to: {self.batch_size}")
        
        self.max_length = 512
        self.contrastive_loss = ContrastiveLoss()

    def extract_features_batch(self, texts: List[str]) -> np.ndarray:
        """批量提取特征"""
        dataset = TextBatchDataset(texts, self.tokenizer, self.max_length)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=4 if self.has_cuda else 0,
            pin_memory=self.has_cuda
        )
        
        all_features = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                if self.has_cuda:
                    input_ids = input_ids.half()
                    attention_mask = attention_mask.half()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # 获取最后四层的隐藏状态
                last_four_layers = outputs.hidden_states[-4:]
                
                # 使用[CLS]标记的输出并取最后四层的平均
                features = torch.stack([layer[:, 0, :] for layer in last_four_layers])
                features = torch.mean(features, dim=0)
                
                all_features.append(features.cpu().float().numpy())
        
        return np.concatenate(all_features, axis=0)

    def process_and_save(self) -> None:
        """处理评论数据并保存特征"""
        try:
            # 加载数据
            df = self.load_comments()
            
            # 将数据分成批次
            texts = df['text'].tolist()
            batch_size = 1000  # 每次处理1000条评论
            num_batches = (len(texts) + batch_size - 1) // batch_size
            
            all_features = []
            
            for i in tqdm(range(num_batches), desc="Processing batches"):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # 批量提取特征
                batch_features = self.extract_features_batch(batch_texts)
                all_features.append(batch_features)
                
                # 显示GPU使用情况
                if self.has_cuda:
                    gpu_memory = torch.cuda.memory_allocated() / 1024**2
                    print(f"\nBatch {i+1}/{num_batches}, GPU memory: {gpu_memory:.2f} MB")
                    # 主动清理GPU内存
                    torch.cuda.empty_cache()
            
            all_features = np.concatenate(all_features, axis=0)
            
            # 保存特征和对应的评论ID
            feature_file = PROCESSED_DATA_PATH / "vine_features.npy"
            id_file = PROCESSED_DATA_PATH / "comment_ids.txt"
            
            np.save(feature_file, all_features)
            with open(id_file, 'w', encoding='utf-8') as f:
                for comment_id in df['id']:
                    f.write(f"{comment_id}\n")
                
            print(f"\n特征已保存到: {feature_file}")
            print(f"评论ID已保存到: {id_file}")
            print(f"特征矩阵形状: {all_features.shape}")
            
        except Exception as e:
            print(f"Error in processing and saving: {str(e)}")
            if self.has_cuda:
                torch.cuda.empty_cache()
            raise

    def load_comments(self) -> pd.DataFrame:
        """加载评论数据"""
        comments = []
        
        print("正在加载评论数据...")
        with open(VINE_DATA_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    comment = json.loads(line)
                    comments.append({
                        'id': comment['_id'],
                        'text': comment['commentText'],
                        'username': comment['username'],
                        'created': comment['created']
                    })
                except json.JSONDecodeError:
                    continue
                    
        df = pd.DataFrame(comments)
        print(f"成功加载 {len(df)} 条评论")
        return df

def main():
    try:
        # 初始化特征提取器
        extractor = BertFeatureExtractor()
        
        # 处理数据并保存特征
        extractor.process_and_save()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

if __name__ == "__main__":
    main() 