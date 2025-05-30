import torch
import numpy as np
import json
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class BertFeatureExtractor:
    """BERT特征提取器，用于从文本中提取语义特征"""
    
    def __init__(self, model_name='bert-base-uncased', device=None, output_dim=64):
        """
        初始化BERT特征提取器
        
        Args:
            model_name: BERT模型名称
            device: 计算设备 (cuda或cpu)
            output_dim: 降维后的特征维度
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"BERT特征提取器使用设备: {self.device}")
        
        # 加载BERT分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 设置为评估模式
        
        # BERT输出维度
        self.bert_dim = 768
        self.output_dim = output_dim
        
        # 创建降维层
        self.dim_reducer = nn.Sequential(
            nn.Linear(self.bert_dim, self.bert_dim // 2),
            nn.LayerNorm(self.bert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.bert_dim // 2, self.output_dim),
            nn.LayerNorm(self.output_dim)
        ).to(self.device)
    
    def extract_features(self, texts, batch_size=16, max_length=128):
        """
        从文本列表中提取BERT特征
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            max_length: 最大序列长度
            
        Returns:
            features: 降维后的特征张量 [n_texts, output_dim]
        """
        # 创建数据集和数据加载器
        dataset = TextDataset(texts, self.tokenizer, max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # 提取特征
        all_features = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="提取BERT特征"):
                # 将输入移动到设备
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 获取BERT输出
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # 使用[CLS]标记的输出作为文本表示
                cls_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
                
                # 降维
                reduced_features = self.dim_reducer(cls_features)  # [batch_size, output_dim]
                
                all_features.append(reduced_features.cpu())
        
        # 合并所有批次的特征
        if all_features:
            return torch.cat(all_features, dim=0)
        else:
            return torch.zeros((0, self.output_dim))
    
    def process_comments(self, comments_file, output_file, batch_size=16):
        """
        处理评论文件，提取BERT特征并保存
        
        Args:
            comments_file: 评论JSON文件路径
            output_file: 输出文件路径
            batch_size: 批处理大小
        """
        print(f"处理评论文件: {comments_file}")
        
        # 加载评论数据
        with open(comments_file, 'r', encoding='utf-8') as f:
            comments = json.load(f)
        
        # 提取评论ID和文本
        comment_ids = [comment['commentId'] for comment in comments]
        comment_texts = [comment['text'] for comment in comments]
        
        print(f"共加载 {len(comment_texts)} 条评论")
        
        # 提取BERT特征
        features = self.extract_features(comment_texts, batch_size=batch_size)
        
        # 将特征转换为NumPy数组
        features_np = features.numpy()
        
        # 创建结果字典
        result = {
            'comment_ids': comment_ids,
            'bert_features': features_np.tolist()
        }
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        
        print(f"评论BERT特征已保存至: {output_file}")
        print(f"特征维度: {features.shape}")
    
    def process_media_sessions(self, media_file, output_file, batch_size=16):
        """
        处理媒体会话文件，提取标题的BERT特征并保存
        
        Args:
            media_file: 媒体会话JSON文件路径
            output_file: 输出文件路径
            batch_size: 批处理大小
        """
        print(f"处理媒体会话文件: {media_file}")
        
        # 加载媒体会话数据
        with open(media_file, 'r', encoding='utf-8') as f:
            media_sessions = json.load(f)
        
        # 提取媒体会话ID和描述
        media_ids = [session['postId'] for session in media_sessions]
        media_descriptions = [session.get('description', '') for session in media_sessions]
        
        print(f"共加载 {len(media_descriptions)} 个媒体会话")
        
        # 提取BERT特征
        features = self.extract_features(media_descriptions, batch_size=batch_size)
        
        # 将特征转换为NumPy数组
        features_np = features.numpy()
        
        # 创建结果字典
        result = {
            'media_ids': media_ids,
            'bert_features': features_np.tolist()
        }
        
        # 保存结果
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f)
        
        print(f"媒体会话BERT特征已保存至: {output_file}")
        print(f"特征维度: {features.shape}")


class TextDataset(Dataset):
    """文本数据集，用于批处理"""
    
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 处理空文本
        if not text or text.isspace():
            text = "[PAD]"
        
        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 去除批次维度
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def main():
    """主函数，用于处理评论和媒体会话数据"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    output_dir = "data/processed/bert_features"
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化BERT特征提取器
    extractor = BertFeatureExtractor(output_dim=64)
    
    # 处理评论数据
    comments_file = "data/processed/integration/comments.json"
    comments_output = os.path.join(output_dir, "comment_bert_features.json")
    extractor.process_comments(comments_file, comments_output)
    
    # 处理媒体会话数据
    media_file = "data/processed/integration/media_sessions.json"
    media_output = os.path.join(output_dir, "media_bert_features.json")
    extractor.process_media_sessions(media_file, media_output)


if __name__ == "__main__":
    main()
