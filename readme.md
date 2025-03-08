数据集结构：
/
├── SampledASONAMPosts.json                # 视频数据
├── sampled_post-comments_vine.json        # 评论数据
├── vine_users_data.json                   # 用户数据
├── knowledge_graph.pkl                    # 知识图谱数据
└── urls_to_postids.txt                    # URL和帖子ID映射

label_data/
    ├── vine_labeled_cyberbullying_data.csv    # 网络欺凌标签数据
    └── aggregate video emotion survey.csv     # 视频情感标签数据

processed/
    ├── frame_features/
    │   ├── {postid}.npy                        # 视频帧特征
    ├── vine_features.npy                       # bert文本特征
    ├── comment_ids.txt                         # 评论ID映射
    └── video_ids.txt                           # 视频ID映射
