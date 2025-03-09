总数据集：
allVineComments1-4.json: 所有视频下方的评论数据
vine_posts_data.json: 所有视频有关数据，包括视频的描述，视频的url，点赞数等等
vine_users_data.json: 所有用户资料

从总数据集进行的采样，总共983个视频（部分视频不知道什么原因变成了970个视频）: 
1.sampled_post-comments_vine.json: allVineComments1-4.json的样本(80945个评论，983个视频)
2.SampledASONAMPosts.json: vine_posts_data.json的样本(983个视频)
3.urls_to_postids.txt: postid和视频url的键值对(970个视频)

以下这些是基于人工标注的对于某个视频的情感定位，如neutral, joy, sad, love, surprise, fear and anger: 
4.aggregate video emotion survey.csv: 人工标注对视频情感及主题分类的数据。情感包括neutral, joy, sad, love, surprise, fear and anger；主题包括：people, person, indoor, outdoor, cartoon, text, activity, animal and other(970个视频)
5.emotion content labels vine.csv: 似乎和aggregate video emotion survey.csv一致(970个视频)
6.individual video emotion survey.csv: 每个视频都有随机三个人进行情感分类和贴标签(2175个视频)

下面这个分为noneAgg, aggression && noneBll, bullying
7.vine_labeled_cyberbullying_data.csv: 人工判断该媒体会话（包括视频内容和该视频下的评论内容）是否存在霸凌或攻击现象(970个媒体会话（每个媒体会话包括视频内容及其下方所有评论）)

feature_extractor.py: 特征提取器
processed/
frame_features: 视频帧特征(文件数量不全，不足970个视频，约800多个，但每个npy是完整的特征。文件名格式为postid)
comment_ids.txt: 内容格式为postid_commentid
vine_features.npy: 视频下的评论特征，通过feature_extractor.py输出
vine_bert_features_origin.npy: 之前用bert提取的特征，可作为对比文件
processed_vine.csv: vine_labeled_cyberbullying_data.csv的缩小版数据，将原数据的comments数量从660缩小到了1，其余不变 (因此，仅供理解！并不能被用作数据集！)

数据集间关联关系(简化为数字表述，每个数字的具体名称在上方有写)：
(注：3是txt文件，因此列名其实不存在，其文件内容为postid, videolink格式)
1:postId - 2:postId - 3:postId
4:videolink - 3:videolink - 5:videolink - 6:videolink - 7:videolink
4:_unit_id - 3:_unit_id - 5:_unit_id - 6:_unit_id

