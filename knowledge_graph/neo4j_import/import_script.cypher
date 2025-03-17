// 在执行导入之前，请确保：
// 1. 所有CSV文件已复制到Neo4j的import目录
// 2. 已设置允许导入本地文件，方法如下：
//    方法1：在neo4j.conf中设置：
//    dbms.security.allow_csv_import_from_file_urls=true
//    
//    方法2：使用APOC库（推荐）：
//    CALL apoc.import.csv(...)

// 清空数据库（可选，请谨慎使用）
// MATCH (n) DETACH DELETE n;

// 创建约束
CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:MediaSession) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.id IS UNIQUE;

// 导入Video节点
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/video_nodes.csv' AS row
CREATE (:Video {id: row.id, url: row.url, description: row.description, likes_count: row.likes_count, timestamp: row.timestamp, comment_count: row.comment_count});

// 导入User节点
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/user_nodes.csv' AS row
CREATE (:User {id: row.id, post_count: row.post_count, comment_count: row.comment_count, activity_score: row.activity_score});

// 导入MediaSession节点
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/mediasession_nodes.csv' AS row
CREATE (:MediaSession {id: row.id, video_id: row.video_id, comment_count: row.comment_count, aggregated_comments: row.aggregated_comments});

// 导入Label节点
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/label_nodes.csv' AS row
CREATE (:Label {id: row.id, target_id: row.target_id, label_type: row.label_type, value: row.value, confidence: row.confidence});

// 导入VIDEO_HAS_SESSION关系
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/video_has_session_rels.csv' AS row
MATCH (source {id: row.source})
MATCH (target {id: row.target})
CREATE (source)-[:VIDEO_HAS_SESSION {creation_time: row.creation_time}]->(target);

// 导入HAS_LABEL关系
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/has_label_rels.csv' AS row
MATCH (source {id: row.source})
MATCH (target {id: row.target})
CREATE (source)-[:HAS_LABEL]->(target);

// 导入USER_COMMENTED_VIDEO关系
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/user_commented_video_rels.csv' AS row
MATCH (source {id: row.source})
MATCH (target {id: row.target})
CREATE (source)-[:USER_COMMENTED_VIDEO {content: row.content, timestamp: row.timestamp, comment_id: row.comment_id}]->(target);

// 导入USER_POSTED_VIDEO关系
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/user_posted_video_rels.csv' AS row
MATCH (source {id: row.source})
MATCH (target {id: row.target})
CREATE (source)-[:USER_POSTED_VIDEO {timestamp: row.timestamp}]->(target);
