示例执行顺序：
1.使用data_processor下的各个classifier进行侵略性和情感分类.
2.使用data_processor下的integration_dataset.py进行元数据的提取，得到三个节点类别的文件，其中都包含了可能有用的元数据特征
3.使用knowledge_graph下的kg_builder构建知识图谱，每个版本都代表不同的图结构和边/节点特征
4.使用experiments下的各个版本进行霸凌检测任务

knowledge_graph:
    kg_builder_v4: 添加了participates_in、offensive_comment和non_offensive_comment关系
    kg_builder_v5: 添加了评论之间的offensive关系，offensive_mentions和non_offensive_mentions
data/processed:
    bert_features: 从评论和媒体会话描述提取出的bert特征，包括64以及768维度
    integration: 
        users|comments|media_sessions.json: data_processor/intrgration_dataset.py - vine数据集中清洗出的用户、评论、媒体会话元数据
        comments_with_aggression.json: data_processor/aggression_analyzer.py - 使用Detoxify进行有毒评论分类
        aggression_analysis_summary.json: data_processor/aggression_analyzer.py - 每个用户的评论有毒程度统计
        text_classification: 
            comment_classifications.json: data_processor/comment_classifier.py - 使用twitter-roberta-base-offensive对评论进行侵略性分类
            video_classifications.json: data_processor/video_classifier.py - 使用twitter-roberta-base-offensive对视频描述进行侵略性分类
            user_classifications.json: data_processor/user_classifier.py - 使用twitter-roberta-base-offensive对用户个人简介进行侵略性分类
    roberta_classification:
        comment_sentiment.json: data_processor/comment_sentiment_classifier.py - 使用cardiffnlp/twitter-roberta-base-sentiment-latest对评论进行情感分类
        video_descriptions_sentiment.json: data_processor/video_sentiment_classifier.py - 使用cardiffnlp/twitter-roberta-base-sentiment-latest对视频描述进行情感分类
    features_extended: 
        user_features_extended.csv: data_processor/user_features_extended_v1.py - 使用cypher从图中提取出额外的用户特征，其中包含40列，与定义里的34列相比添加了userId以及原特征中的5个比例类型的数值的分子, 如19. **被攻击集中度（收到）** (`从单一用户收到的最大OFFENSIVE次数 / 总被OFFENSIVE次数 (3)`)中的分子: 从单一用户收到的最大OFFENSIVE次数
        user_features_extended.csv: data_processor/user_features_extended_v2.py - 使用cypher从图中提取出额外的用户特征，在v1的基础上，删除了与霸凌相关的统计特征，以方便后续统计工作，且将注释和表头修改为了正确的"mention"描述
        user_features_extended.csv: data_processor/user_features_extended_v3.py - 使用cypher从图中提取出额外的用户特征，在v2的基础上，添加了除mention外的其他全局统计特征。
experiments_heterograph:
    old_v1-v7: 旧版，可能使用了错误的cypher，效果不佳，选用了不同的模型以及图结构
    v8: 选用v4还是v5图，忘了，使用HGT进行建模
    v9: 选用v4图，使用GAT进行建模。模型准确率76%，F1分数上霸凌0.44, 非霸凌0.84，在数据及模型表现上存在严重不平衡
    v10: 选用v4图，使用GAT进行建模，在v9的基础上，使用了过采样技术，强制每个批次中霸凌及非霸凌样本均衡，以及添加了交叉熵损失的类别权重，霸凌:非霸凌约为0.18:0.82的权重（当前版本为1：1的权重）。模型准确率74%，F1分数上霸凌0.78，非霸凌0.68
    v11: 选用v4图，在v10的基础上，替换GAT为RGCN网络，运行速度大幅降低，准确率几乎一致，但RGCN明显存在过拟合
    v12: 选用v4图，在v10的基础上，扩展了29个用户特征，并且数据集按照时序进行划分，准确率和F1 Score上升到了77%左右，并且非霸凌和霸凌的预测结果很平衡，没有倾斜
    v13: 选用v4图，在v12的基础上...
experiments_others:
    v1: 使用随机森林进行分类任务，数据来源是v4图数据库，特征包括媒体会话、评论以及用户。现在发现用户特征少了一个likeCount维度
    v2: 使用随机森林进行分类任务，数据来源是由data_processor\user_features_extended_v1.py提取的用户特征user_features_extended.csv，特征只有用户，元特征+统计特征，总共40个特征。模拟霸凌子图结构识别，将媒体会话中有mention关系的用户提取出来作为该媒体会话下的特征，然后用户特征平均池化，送入模型分类，准确率高达93%，F1高达0.90（但是这里是存在泄题问题，且可用会话数从970->501，约缩减一半）
experiments_multigraphs:
    v1: 将图分为三个同构图，用户图，评论图，媒体会话图，分别聚合后得到图表征，将图表征进行拼接送入模型进行分类
models:
    设置的错误路径，experiments_heterograph的模型保存路径，后期有需要可以修改回去