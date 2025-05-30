import torch
from detoxify import Detoxify
from typing import Dict, Any, List, Tuple
import logging
import json
import os
import time
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AggressionAnalyzer:
    """使用Detoxify模型分析文本的攻击性内容"""

    def __init__(self):
        """初始化攻击性分析器"""
        # 检查是否有GPU可用
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 加载Detoxify模型
        logger.info("加载Detoxify模型...")
        self.model = Detoxify('original', device=self.device)

        # 定义攻击类型及其对应的阈值
        self.attack_types = {
            "toxicity": 0.5,        # 有毒/有害内容
            "severe_toxicity": 0.5,  # 严重有毒内容
            "obscene": 0.5,         # 淫秽内容
            "threat": 0.4,          # 威胁内容
            "insult": 0.5,          # 侮辱内容
            "identity_attack": 0.4   # 身份攻击
        }

        logger.info("攻击性分析器初始化完成")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """分析文本的攻击性内容

        Args:
            text: 要分析的文本

        Returns:
            Dict with keys:
            - aggression_score: 综合攻击性得分 (0-1)
            - attack_types: 检测到的攻击类型列表
            - attack_scores: 各种攻击类型的得分
            - is_aggressive: 是否被判定为攻击性内容
        """
        if not text or len(text.strip()) == 0:
            return {
                "aggression_score": 0.0,
                "attack_types": [],
                "attack_scores": {},
                "is_aggressive": False
            }

        try:
            # 使用Detoxify预测文本的毒性
            results = self.model.predict(text)

            # 确定检测到的攻击类型
            detected_attacks = []
            for attack_type, threshold in self.attack_types.items():
                if results[attack_type] >= threshold:
                    detected_attacks.append(attack_type)

            # 计算综合攻击性得分 (加权平均)
            weights = {
                "toxicity": 1.0,
                "severe_toxicity": 1.5,
                "obscene": 0.8,
                "threat": 1.2,
                "insult": 1.0,
                "identity_attack": 1.3
            }

            total_weight = sum(weights.values())
            aggression_score = sum(results[attack_type] * weights[attack_type]
                                  for attack_type in self.attack_types.keys()) / total_weight

            # 判断是否为攻击性内容
            is_aggressive = len(detected_attacks) > 0 or aggression_score >= 0.5

            return {
                "aggression_score": aggression_score,
                "attack_types": detected_attacks,
                "attack_scores": {k: float(v) for k, v in results.items()},
                "is_aggressive": is_aggressive
            }

        except Exception as e:
            logger.error(f"分析文本时出错: {e}")
            return {
                "aggression_score": 0.0,
                "attack_types": [],
                "attack_scores": {},
                "is_aggressive": False,
                "error": str(e)
            }

    def analyze_text_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """批量分析多个文本的攻击性内容

        Args:
            texts: 要分析的文本列表
            batch_size: 批处理大小

        Returns:
            每个文本的分析结果列表
        """
        if not texts:
            return []

        # 过滤掉空文本
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [{
                "aggression_score": 0.0,
                "attack_types": [],
                "attack_scores": {},
                "is_aggressive": False
            } for _ in texts]

        # 初始化结果列表
        results = [{
            "aggression_score": 0.0,
            "attack_types": [],
            "attack_scores": {},
            "is_aggressive": False
        } for _ in texts]

        # 分批处理
        for i in tqdm(range(0, len(valid_texts), batch_size), desc="批量分析文本"):
            batch_texts = valid_texts[i:i+batch_size]
            try:
                # 使用Detoxify批量预测
                batch_results = self.model.predict(batch_texts)

                # 处理每个文本的结果
                for j, text_idx in enumerate(valid_indices[i:i+batch_size]):
                    # 确定检测到的攻击类型
                    detected_attacks = []
                    for attack_type, threshold in self.attack_types.items():
                        if batch_results[attack_type][j] >= threshold:
                            detected_attacks.append(attack_type)

                    # 计算综合攻击性得分 (加权平均)
                    weights = {
                        "toxicity": 1.0,
                        "severe_toxicity": 1.5,
                        "obscene": 0.8,
                        "threat": 1.2,
                        "insult": 1.0,
                        "identity_attack": 1.3
                    }

                    total_weight = sum(weights.values())
                    aggression_score = sum(batch_results[attack_type][j] * weights[attack_type]
                                          for attack_type in self.attack_types.keys()) / total_weight

                    # 判断是否为攻击性内容
                    is_aggressive = len(detected_attacks) > 0 or aggression_score >= 0.5

                    results[text_idx] = {
                        "aggression_score": float(aggression_score),
                        "attack_types": detected_attacks,
                        "attack_scores": {k: float(v[j]) for k, v in batch_results.items()},
                        "is_aggressive": is_aggressive
                    }
            except Exception as e:
                logger.error(f"批量分析文本时出错 (批次 {i}): {e}")

        return results

# 单例模式，避免重复加载模型
_analyzer_instance = None

def get_analyzer():
    """获取AggressionAnalyzer的单例实例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AggressionAnalyzer()
    return _analyzer_instance

def analyze_comments_file(input_file: str, output_file: str, batch_size: int = 32):
    """分析评论文件并保存结果

    Args:
        input_file: 输入文件路径 (JSON格式)
        output_file: 输出文件路径
        batch_size: 批处理大小
    """
    start_time = time.time()
    logger.info(f"开始分析评论文件: {input_file}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 加载评论数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            comments = json.load(f)
        logger.info(f"加载了 {len(comments)} 条评论")
    except Exception as e:
        logger.error(f"加载评论文件时出错: {e}")
        return

    # 提取评论文本
    comment_texts = []
    for comment in comments:
        text = comment.get("text", "")
        comment_texts.append(text)

    # 初始化分析器
    analyzer = get_analyzer()

    # 分析评论
    logger.info("开始分析评论...")
    results = analyzer.analyze_text_batch(comment_texts, batch_size=batch_size)

    # 将结果添加到评论数据中
    for i, (comment, result) in enumerate(zip(comments, results)):
        comment["aggression_score"] = result["aggression_score"]
        comment["attack_types"] = result["attack_types"]
        comment["attack_scores"] = result["attack_scores"]
        comment["is_aggressive"] = result["is_aggressive"]

    # 保存结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comments, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存结果时出错: {e}")

    # 统计信息
    aggressive_count = sum(1 for result in results if result["is_aggressive"])
    elapsed_time = time.time() - start_time

    logger.info(f"分析完成。总计 {len(comments)} 条评论，其中 {aggressive_count} 条具有攻击性 ({aggressive_count/len(comments)*100:.2f}%)")
    logger.info(f"处理时间: {elapsed_time:.2f} 秒，平均每条评论 {elapsed_time/len(comments):.4f} 秒")

    # 创建摘要文件
    summary = {
        "total_comments": len(comments),
        "aggressive_comments": aggressive_count,
        "non_aggressive_comments": len(comments) - aggressive_count,
        "aggressive_percentage": aggressive_count / len(comments) * 100 if comments else 0,
        "processing_time": elapsed_time,
        "average_time_per_comment": elapsed_time / len(comments) if comments else 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    summary_file = os.path.join(os.path.dirname(output_file), "aggression_analysis_summary.json")
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(f"摘要已保存到: {summary_file}")
    except Exception as e:
        logger.error(f"保存摘要时出错: {e}")

def main():
    """主函数"""
    # 定义输入和输出文件路径
    input_file = "data/processed/integration/comments.json"
    output_file = "data/processed/integration/comments_with_aggression.json"

    # 分析评论
    analyze_comments_file(input_file, output_file)

if __name__ == "__main__":
    main()
