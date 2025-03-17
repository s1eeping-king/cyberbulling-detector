import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from transformers import BertTokenizer, BertModel
import numpy as np

class BullyingDetectionWithKG(nn.Module):
    def __init__(self, num_classes, bert_hidden_size=768, gcn_hidden_size=256):
        super(BullyingDetectionWithKG, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        # 知识图谱的GCN层
        self.gcn1 = GCNConv(bert_hidden_size, gcn_hidden_size)
        self.gcn2 = GCNConv(gcn_hidden_size, gcn_hidden_size)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(bert_hidden_size + gcn_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, text, edge_index, edge_attr=None):
        # 文本特征提取
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        bert_output = self.bert(**inputs).last_hidden_state[:, 0, :]  # 使用[CLS]标记的输出
        
        # 知识图谱特征提取
        gcn_output = self.gcn1(bert_output, edge_index, edge_attr)
        gcn_output = torch.relu(gcn_output)
        gcn_output = self.gcn2(gcn_output, edge_index, edge_attr)
        
        # 特征融合
        combined_features = torch.cat([bert_output, gcn_output], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        return output

def create_sample_knowledge_graph():
    # 示例：创建一个简单的知识图谱
    # 节点表示文本中的实体，边表示实体间的关系
    edge_index = torch.tensor([[0, 1, 1, 2],
                             [1, 0, 2, 1]], dtype=torch.long)
    return edge_index

def main():
    # 示例使用
    model = BullyingDetectionWithKG(num_classes=2)  # 二分类：霸凌/非霸凌
    
    # 示例文本
    sample_texts = ["你是个笨蛋", "今天天气真好"]
    
    # 创建知识图谱
    edge_index = create_sample_knowledge_graph()
    
    # 模型推理
    outputs = model(sample_texts, edge_index)
    
    # 获取预测结果
    predictions = torch.softmax(outputs, dim=1)
    print("预测结果:", predictions)

if __name__ == "__main__":
    main()
