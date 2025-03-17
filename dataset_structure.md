# 数据集结构

- **data/**  — 数据根目录
  - `SampledASONAMPosts.json`  — 视频数据
  - `sampled_post-comments_vine.json`  — 评论数据
  - `vine_users_data.json`  — 用户数据
  - `urls_to_postids.txt`  — URL 和帖子 ID 映射

  - **label_data/**  — 标注数据目录
    - `vine_labeled_cyberbullying_data.csv`  — 网络欺凌标签数据
    - `aggregate video emotion survey.csv`  — 视频情感标签数据
    - `individual video emotion survey.csv`  — 个人视频情感标签数据

  - **processed/**  — 预处理数据目录
    - **frame_features/**  — 视频帧特征目录
      - `{postid}.npy`  — 各视频的帧特征文件
    - `vine_features.npy`  — BERT 文本特征
    - `comment_ids.txt`  — 评论 ID 映射
