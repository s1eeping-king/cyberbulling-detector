# 霸凌检测知识图谱设计（更新版）

## 1. 核心实体

### A. Video（视频）
- **节点类型**: `Video`
- **主要属性**：
  - `id`: 视频唯一标识符 (postId)
  - `url`: 视频链接
  - `description`: 视频描述
  - `likes_count`: 点赞数
  - `timestamp`: 发布时间
  - `frame_features`: 视频帧特征向量 (从frame_features/目录获取)
  - `comment_count`: 评论数量

### B. User（用户）
- **节点类型**: `User`
- **主要属性**：
  - `id`: 用户唯一标识符
  - `profile`: 用户资料
  - `post_count`: 发布视频数量
  - `comment_count`: 评论数量
  - `activity_score`: 用户活跃度分数

### C. MediaSession（媒体会话）
- **节点类型**: `MediaSession`
- **主要属性**：
  - `video_id`: 关联视频ID
  - `comment_count`: 评论数量
  - `aggregated_comments`: 聚合的评论内容
  - `session_features`: 会话特征向量

### D. Label（标签）
- **节点类型**: `Label`
- **主要属性**：
  - `target_id`: 关联目标ID (视频ID或媒体会话ID)
  - `label_type`: 标签类型 (bullying/aggression/emotion/topic)
  - `value`: 标签值
  - `confidence`: 标签置信度 (人工标注的一致性程度)

## 2. 关系设计

### A. 创作关系
- **关系类型**: `USER_POSTED_VIDEO`
- **起点**: User
- **终点**: Video
- **属性**:
  - `timestamp`: 发布时间

### B. 互动关系
- **关系类型**: `USER_COMMENTED_VIDEO`
- **起点**: User
- **终点**: Video
- **属性**:
  - `content`: 评论内容
  - `timestamp`: 评论时间
  - `comment_id`: 评论ID

### C. 会话关系
- **关系类型**: `VIDEO_HAS_SESSION`
- **起点**: Video
- **终点**: MediaSession
- **属性**:
  - `creation_time`: 会话创建时间

### D. 标签关系
- **关系类型**: `HAS_LABEL`
- **起点**: Video 或 MediaSession
- **终点**: Label
- **属性**:
  - `source`: 标签来源 (人工标注)
  - `annotation_time`: 标注时间

## 3. 数据源映射

| 实体/关系 | 数据源 | 字段映射 |
|----------|--------|---------|
| Video | SampledASONAMPosts.json | _id → id, url → url, description → description, likes_count → likes_count |
| User | vine_users_data.json | _id → id, profile → profile |
| MediaSession | 由视频和评论组合生成 | postId → video_id, comments → aggregated_comments |
| Label (bullying) | vine_labeled_cyberbullying_data.csv | question2 → value, _golden → confidence |
| Label (aggression) | vine_labeled_cyberbullying_data.csv | question1 → value, _golden → confidence |
| Label (emotion) | aggregate video emotion survey.csv, individual video emotion survey.csv | emotion → value, _trusted → confidence |
| Label (topic) | aggregate video emotion survey.csv | topic → value, _trusted → confidence |
| USER_POSTED_VIDEO | SampledASONAMPosts.json | username → User.id, _id → Video.id |
| USER_COMMENTED_VIDEO | sampled_post-comments_vine.json | username → User.id, postId → Video.id, commentText → content |
| VIDEO_HAS_SESSION | 构建过程中生成 | Video.id → MediaSession.video_id |
| 视频特征 | frame_features/*.npy | 文件名(postId) → Video.id, 特征向量 → frame_features |
| 评论特征 | vine_features.npy, comment_ids.txt | 特征向量 → MediaSession.session_features |

## 4. 图谱构建流程

1. **数据加载与预处理**
   - 加载所有数据源
   - 创建postId和videolink的映射关系
   - 计算用户活跃度指标
   - 处理标签置信度信息

2. **节点构建**
   - 构建Video节点
   - 构建活跃User节点 (活跃度阈值: ≥3次互动)
   - 构建MediaSession节点 (聚合视频及其评论)
   - 构建Label节点 (包含置信度)

3. **关系构建**
   - 构建USER_POSTED_VIDEO关系
   - 构建USER_COMMENTED_VIDEO关系
   - 构建VIDEO_HAS_SESSION关系
   - 构建HAS_LABEL关系

4. **特征向量整合**
   - 添加视频帧特征
   - 处理评论文本特征
   - 构建媒体会话特征

## 5. 优化策略

1. **节点优化**
   - 只保留活跃用户 (减少约80%的用户节点)
   - 评论作为关系属性而非独立节点 (减少约80,000个节点)
   - 添加MediaSession节点作为霸凌检测的主要目标 (约970个节点)

2. **关系优化**
   - 直接连接用户和视频 (减少中间节点)
   - 标签直接关联到相应实体 (视频或媒体会话)

3. **特征优化**
   - 视频特征直接附加到节点
   - 评论特征聚合到媒体会话节点
   - 保留标签置信度信息

## 6. 霸凌检测应用

1. **节点分类任务**
   - 目标: 预测MediaSession节点的霸凌和攻击性标签
   - 特征: 视频内容特征 + 用户互动模式 + 评论文本特征

2. **消息传递机制**
   - User → Video: 用户行为和互动模式
   - Video → MediaSession: 视频内容信息
   - Label → MediaSession: 标签信息传递
   - MediaSession → MediaSession: 内容相似性传递

3. **模型架构**
   - 异构图神经网络 (HGNN)
   - 多任务学习 (霸凌检测 + 攻击性检测)
   - 注意力机制整合多模态特征
   - 置信度加权的损失函数

## 7. 图谱统计

预期统计:
- 节点数量: ~3,000 (约970个视频 + ~970个媒体会话 + ~1,000个活跃用户 + ~1,000个标签)
- 关系数量: ~12,000 (用户发布、评论、会话和标签关系)
- 平均度数: ~4 (每个节点平均连接数)

## 8. 重要注意点

1. **媒体会话的重要性**:
   - 霸凌和攻击性标签是针对整个媒体会话的 (视频+评论)，而不仅仅是视频内容
   - 需要将视频内容和评论内容整合到MediaSession节点中

2. **标签置信度**:
   - 所有标签都来自人工标注，具有置信度信息
   - 情感标签、主题标签、霸凌标签和攻击性标签都有各自的置信度
   - 置信度应该被整合到模型训练中

3. **多人标注**:
   - individual video emotion survey.csv 中每个视频有3个人进行标注
   - 可以利用多人标注的一致性作为置信度的指标
