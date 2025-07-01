#!/usr/bin/env python3
"""
测试GDS特征提取功能
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from user_features_extended import UserFeaturesExtractor

def test_gds_features():
    """测试GDS特征提取功能"""
    
    # Neo4j连接信息
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "12345678"  # 请替换为实际密码
    
    # 初始化特征提取器
    extractor = UserFeaturesExtractor(uri, username, password)
    
    try:
        print("开始测试GDS特征提取...")
        
        # 获取少量媒体会话进行测试
        media_sessions = extractor.get_media_sessions_by_time()
        test_sessions = media_sessions[:10]  # 只取前10个会话进行测试
        
        print(f"测试会话数量: {len(test_sessions)}")
        
        # 获取这些会话中的用户
        test_users = extractor.get_users_in_media_sessions([session[0] for session in test_sessions])
        test_users = test_users[:5]  # 只取前5个用户进行测试
        
        print(f"测试用户数量: {len(test_users)}")
        
        if not test_users:
            print("没有找到测试用户，退出测试")
            return
        
        # 测试GDS特征提取
        print("\n测试GDS特征提取...")
        gds_features = extractor.extract_user_features(
            user_ids=test_users,
            statistic_media_sessions=[session[0] for session in test_sessions],
            use_gds=True,
            split_name="train"
        )
        
        print(f"GDS特征提取完成，提取了 {len(gds_features)} 个用户的特征")
        
        # 测试传统Cypher特征提取进行对比
        print("\n测试传统Cypher特征提取...")
        cypher_features = extractor.extract_user_features(
            user_ids=test_users,
            statistic_media_sessions=[session[0] for session in test_sessions],
            use_gds=False,
            split_name=None
        )
        
        print(f"Cypher特征提取完成，提取了 {len(cypher_features)} 个用户的特征")
        
        # 比较结果
        print("\n比较GDS和Cypher特征提取结果...")
        if len(gds_features) == len(cypher_features):
            print("✓ 特征数量一致")
            
            # 比较第一个用户的特征
            if gds_features and cypher_features:
                gds_user = gds_features[0]
                cypher_user = cypher_features[0]
                
                print(f"\n用户 {gds_user['userId']} 的特征对比:")
                print("特征名称\t\tGDS值\t\tCypher值")
                print("-" * 50)
                
                for key in gds_user.keys():
                    if key in cypher_user:
                        gds_val = gds_user[key]
                        cypher_val = cypher_user[key]
                        print(f"{key:<20}\t{gds_val}\t\t{cypher_val}")
        else:
            print("✗ 特征数量不一致")
        
        print("\n测试完成！")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭连接
        extractor.close()

if __name__ == "__main__":
    test_gds_features()
