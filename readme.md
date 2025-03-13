# 数据集结构

以下是数据集的结构说明：

- **根目录**
  - `SampledASONAMPosts.json`  — 视频数据
  - `sampled_post-comments_vine.json`  — 评论数据
  - `vine_users_data.json`  — 用户数据
  - `knowledge_graph.pkl`  — 知识图谱数据
  - `urls_to_postids.txt`  — URL 和帖子 ID 映射

- **label_data/**  
  - `vine_labeled_cyberbullying_data.csv`  — 网络欺凌标签数据
  - `aggregate_video_emotion_survey.csv`  — 视频情感标签数据

- **processed/**
  - **frame_features/**
    - `{postid}.npy`  — 视频帧特征
  - `vine_features.npy`  — BERT 文本特征
  - `comment_ids.txt`  — 评论 ID 映射
  - `video_ids.txt`  — 视频 ID 映射