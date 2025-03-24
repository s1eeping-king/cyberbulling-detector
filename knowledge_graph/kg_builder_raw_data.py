from neo4j import GraphDatabase
import json
import os
from typing import Dict, List, Any
import logging

class KnowledgeGraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the knowledge graph builder with Neo4j connection details"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = self.setup_logger()

    @staticmethod
    def setup_logger():
        """Set up logging configuration"""
        logger = logging.getLogger('KnowledgeGraphBuilder')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            self.logger.info("Database cleared successfully")

    def create_constraints(self):
        """Create necessary constraints for the nodes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:MediaSession) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.id IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.error(f"Error creating constraint: {e}")
            self.logger.info("Constraints created successfully")

    def create_user_nodes(self, users: List[Dict]):
        """Create user nodes in the database"""
        query = """
        UNWIND $users AS user
        MERGE (u:User {id: user.id})
        SET u += user.properties
        """
        with self.driver.session() as session:
            session.run(query, users=users)
            self.logger.info(f"Created {len(users)} user nodes")

    def create_media_session_nodes(self, media_sessions: List[Dict]):
        """Create media session nodes in the database"""
        query = """
        UNWIND $sessions AS session
        MERGE (m:MediaSession {id: session.id})
        SET m += session.properties
        """
        with self.driver.session() as session:
            session.run(query, sessions=media_sessions)
            self.logger.info(f"Created {len(media_sessions)} media session nodes")

    def create_comment_nodes(self, comments: List[Dict]):
        """Create comment nodes in the database"""
        query = """
        UNWIND $comments AS comment
        MERGE (c:Comment {id: comment.id})
        SET c += comment.properties
        """
        with self.driver.session() as session:
            session.run(query, comments=comments)
            self.logger.info(f"Created {len(comments)} comment nodes")
            # 记录一些评论节点的示例，用于调试
            if comments:
                sample_size = min(5, len(comments))
                self.logger.info(f"Sample comment nodes: {json.dumps(comments[:sample_size], indent=2)}")

    def create_label_nodes(self, labels: List[Dict]):
        """Create label nodes in the database"""
        query = """
        UNWIND $labels AS label
        MERGE (l:Label {id: label.id})
        SET l += label.properties
        """
        with self.driver.session() as session:
            session.run(query, labels=labels)
            self.logger.info(f"Created {len(labels)} label nodes")

    def create_relationships(self, relationships: Dict[str, List[Dict]]):
        """Create relationships between nodes"""
        # 首先验证和记录关系数据
        for rel_type, rels in relationships.items():
            self.logger.info(f"Processing {rel_type} relationships. Total count: {len(rels)}")
            if rels:
                sample_size = min(5, len(rels))
                self.logger.info(f"Sample {rel_type} relationships: {json.dumps(rels[:sample_size], indent=2)}")

        relationship_queries = {
            "publishes": """
            // User -> MediaSession
            UNWIND $rels AS rel
            MATCH (u:User {id: rel.source})
            MATCH (m:MediaSession {id: rel.target})
            MERGE (u)-[r:PUBLISHES]->(m)
            SET r += rel.properties
            """,
            "creates": """
            // User -> Comment
            UNWIND $rels AS rel
            MATCH (u:User {id: rel.source})
            MATCH (c:Comment {id: rel.target})
            MERGE (u)-[r:CREATES]->(c)
            SET r += rel.properties
            RETURN count(*) as created_count
            """,
            "mentions": """
            // Comment -> User
            UNWIND $rels AS rel
            MATCH (c:Comment {id: rel.source})
            MATCH (u:User {id: rel.target})
            MERGE (c)-[r:MENTIONS]->(u)
            SET r += rel.properties
            """,
            "annotates": """
            // MediaSession -> Label
            UNWIND $rels AS rel
            MATCH (m:MediaSession {id: rel.source})
            MATCH (l:Label {id: rel.target})
            MERGE (m)-[r:HAS_LABEL]->(l)
            SET r += rel.properties
            """,
            "belongs_to": """
            // Comment -> MediaSession
            UNWIND $rels AS rel
            MATCH (c:Comment {id: rel.source})
            MATCH (m:MediaSession {id: rel.target})
            MERGE (c)-[r:BELONGS_TO]->(m)
            SET r += rel.properties
            """
        }

        verification_queries = {
            "creates": """
            MATCH p=(u:User)-[r:CREATES]->(c:Comment)
            RETURN count(p) as relationship_count
            """,
            "publishes": """
            MATCH p=(u:User)-[r:PUBLISHES]->(m:MediaSession)
            RETURN count(p) as relationship_count
            """,
            "mentions": """
            MATCH p=(c:Comment)-[r:MENTIONS]->(u:User)
            RETURN count(p) as relationship_count
            """,
            "annotates": """
            MATCH p=(m:MediaSession)-[r:HAS_LABEL]->(l:Label)
            RETURN count(p) as relationship_count
            """,
            "belongs_to": """
            MATCH p=(c:Comment)-[r:BELONGS_TO]->(m:MediaSession)
            RETURN count(p) as relationship_count
            """
        }

        with self.driver.session() as session:
            for rel_type, rels in relationships.items():
                if rel_type in relationship_queries:
                    try:
                        # 执行关系创建
                        result = session.run(relationship_queries[rel_type], rels=rels)
                        
                        # 验证创建的关系
                        if rel_type in verification_queries:
                            verify_result = session.run(verification_queries[rel_type])
                            actual_count = verify_result.single()["relationship_count"]
                            self.logger.info(f"Created and verified {actual_count} {rel_type} relationships")
                            
                            if actual_count < len(rels):
                                # 检查缺失的关系
                                self.logger.warning(f"Missing {len(rels) - actual_count} {rel_type} relationships")
                                # 添加诊断查询
                                missing_nodes_query = """
                                UNWIND $rels AS rel
                                OPTIONAL MATCH (source {id: rel.source})
                                OPTIONAL MATCH (target {id: rel.target})
                                WITH rel, source, target
                                WHERE source IS NULL OR target IS NULL
                                RETURN DISTINCT rel.source as source_id, 
                                       rel.target as target_id,
                                       source IS NOT NULL as source_exists,
                                       target IS NOT NULL as target_exists
                                LIMIT 5
                                """
                                missing_results = session.run(missing_nodes_query, rels=rels)
                                for record in missing_results:
                                    self.logger.warning(f"Missing nodes for {rel_type} relationship: {dict(record)}")
                    except Exception as e:
                        self.logger.error(f"Error creating {rel_type} relationships: {e}")
                        if rels:
                            self.logger.error(f"Sample relationship data: {json.dumps(rels[0], indent=2)}")

    def build_knowledge_graph(self, data_file: str):
        """Build the complete knowledge graph from the input data file"""
        try:
            # Load the data
            self.logger.info(f"Loading data from {data_file}")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 记录数据统计
            self.logger.info("Data statistics:")
            for node_type, nodes in data['nodes'].items():
                self.logger.info(f"  {node_type}: {len(nodes)} nodes")
            for rel_type, rels in data['relationships'].items():
                self.logger.info(f"  {rel_type}: {len(rels)} relationships")

            # Clear existing data
            self.clear_database()

            # Create constraints
            self.create_constraints()

            # Create nodes
            self.create_user_nodes(data['nodes']['users'])
            self.create_media_session_nodes(data['nodes']['media_sessions'])
            self.create_comment_nodes(data['nodes']['comments'])
            self.create_label_nodes(data['nodes']['labels'])

            # Create relationships
            self.create_relationships(data['relationships'])

            # 验证最终结果
            with self.driver.session() as session:
                # 验证节点数量
                for node_type in ['User', 'Comment', 'MediaSession', 'Label']:
                    result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                    count = result.single()["count"]
                    self.logger.info(f"Final count of {node_type} nodes: {count}")

                # 验证关系数量和方向
                relationship_checks = [
                    ("User-[CREATES]->Comment", "MATCH p=(u:User)-[r:CREATES]->(c:Comment) RETURN count(p) as count"),
                    ("User-[PUBLISHES]->MediaSession", "MATCH p=(u:User)-[r:PUBLISHES]->(m:MediaSession) RETURN count(p) as count"),
                    ("Comment-[MENTIONS]->User", "MATCH p=(c:Comment)-[r:MENTIONS]->(u:User) RETURN count(p) as count"),
                    ("MediaSession-[HAS_LABEL]->Label", "MATCH p=(m:MediaSession)-[r:HAS_LABEL]->(l:Label) RETURN count(p) as count"),
                    ("Comment-[BELONGS_TO]->MediaSession", "MATCH p=(c:Comment)-[r:BELONGS_TO]->(m:MediaSession) RETURN count(p) as count")
                ]

                for description, query in relationship_checks:
                    result = session.run(query)
                    count = result.single()["count"]
                    self.logger.info(f"Final count of {description}: {count}")

            self.logger.info("Knowledge graph built successfully")

        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {e}")
            raise

def main():
    # Load Neo4j connection information
    with open("knowledge_graph/neo4j_information.txt", 'r') as f:
        neo4j_info = {}
        for line in f:
            key, value = line.strip().split(' = ')
            # Remove quotes if present
            value = value.strip('"')
            neo4j_info[key] = value

    # Initialize and run the knowledge graph builder
    builder = KnowledgeGraphBuilder(
        uri=neo4j_info['uri'],
        username=neo4j_info['username'],
        password=neo4j_info['password']
    )

    try:
        # Build the knowledge graph using the combined features file
        builder.build_knowledge_graph("data/processed/full_features.json")
    finally:
        builder.close()

if __name__ == "__main__":
    main()
