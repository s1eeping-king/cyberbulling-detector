from neo4j import GraphDatabase
import json
import os
import time
from typing import Dict, List, Any, Tuple, Union
import logging
import numpy as np
from tqdm import tqdm

class KnowledgeGraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        """Initialize the knowledge graph builder with Neo4j connection details"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.logger = self.setup_logger()

        # Load pre-computed classification results
        self.logger.info("Loading pre-computed classification results...")

        # Load comment classifications
        comment_classifications_path = "data/processed/integration/text_classification/comment_classifications.json"
        self.comment_classifications = {}
        if os.path.exists(comment_classifications_path):
            with open(comment_classifications_path, 'r', encoding='utf-8') as f:
                comment_classifications = json.load(f)
                for item in comment_classifications:
                    self.comment_classifications[item.get("commentId", "")] = {
                        "classification": item.get("classification", "not_offensive"),
                        "confidence": item.get("confidence", 1.0)
                    }
            self.logger.info(f"Loaded {len(self.comment_classifications)} comment classifications")
        else:
            self.logger.warning(f"Comment classifications file not found: {comment_classifications_path}")

        # Load video classifications
        video_classifications_path = "data/processed/integration/text_classification/video_classifications.json"
        self.video_classifications = {}
        if os.path.exists(video_classifications_path):
            with open(video_classifications_path, 'r', encoding='utf-8') as f:
                video_classifications = json.load(f)
                for item in video_classifications:
                    self.video_classifications[item.get("postId", "")] = {
                        "classification": item.get("classification", "not_offensive"),
                        "confidence": item.get("confidence", 1.0)
                    }
            self.logger.info(f"Loaded {len(self.video_classifications)} video classifications")
        else:
            self.logger.warning(f"Video classifications file not found: {video_classifications_path}")

        # Load user classifications
        user_classifications_path = "data/processed/integration/text_classification/user_classifications.json"
        self.user_classifications = {}
        if os.path.exists(user_classifications_path):
            with open(user_classifications_path, 'r', encoding='utf-8') as f:
                user_classifications = json.load(f)
                for item in user_classifications:
                    self.user_classifications[item.get("userId", "")] = {
                        "classification": item.get("classification", "not_offensive"),
                        "confidence": item.get("confidence", 1.0)
                    }
            self.logger.info(f"Loaded {len(self.user_classifications)} user classifications")
        else:
            self.logger.warning(f"User classifications file not found: {user_classifications_path}")

        self.logger.info("Initialization complete")

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

    # These methods have been removed as we now use pre-computed classification results

    def normalize_values(self, values: List[float]) -> List[float]:
        """Normalize a list of numerical values to range [0, 1]"""
        if not values:
            return []

        try:
            values_array = np.array(values, dtype=float)
            min_val = np.min(values_array)
            max_val = np.max(values_array)

            # Avoid division by zero
            if max_val == min_val:
                return [0.0] * len(values)

            normalized = (values_array - min_val) / (max_val - min_val)
            return normalized.tolist()
        except Exception as e:
            self.logger.error(f"Error normalizing values: {e}")
            return [0.0] * len(values)

    def normalize_by_field(self, items: List[Dict], field_names: List[str], method="log"):
        """
        Normalize numerical values for each field separately across all items using non-linear methods.

        Args:
            items: List of dictionaries (users, media_sessions, etc.)
            field_names: List of field names to normalize
            method: Normalization method to use ("log", "sqrt", "sigmoid", "linear")

        This method adds normalized values as new fields with "_normalized" suffix
        """
        self.logger.info(f"Normalizing fields {field_names} across {len(items)} items using {method} normalization")

        try:
            # For each field, collect all values, normalize them, and update the items
            for field in field_names:
                # Extract values for this field from all items
                field_values = []
                valid_indices = []

                for i, item in enumerate(items):
                    properties = item.get("properties", {})
                    if field in properties:
                        try:
                            value = float(properties[field])
                            field_values.append(value)
                            valid_indices.append(i)
                        except (ValueError, TypeError):
                            # Skip non-numeric values
                            pass

                if not field_values:
                    self.logger.warning(f"No valid values found for field {field}")
                    continue

                # Convert to numpy array for efficient operations
                values_array = np.array(field_values, dtype=float)

                # Calculate min and max
                min_val = np.min(values_array)
                max_val = np.max(values_array)

                # Apply non-linear transformation based on the selected method
                if method == "log":
                    # Log normalization: log(1 + x)
                    # Add a small epsilon to avoid log(0)
                    epsilon = 1e-10
                    transformed_values = np.log1p(values_array)
                    transformed_min = np.log1p(min_val)
                    transformed_max = np.log1p(max_val)

                    # Normalize to [0, 1] range
                    if transformed_max == transformed_min:
                        normalized_values = np.full_like(transformed_values, 0.5)
                    else:
                        normalized_values = (transformed_values - transformed_min) / (transformed_max - transformed_min)

                elif method == "sqrt":
                    # Square root normalization
                    transformed_values = np.sqrt(values_array)
                    transformed_min = np.sqrt(min_val) if min_val >= 0 else 0
                    transformed_max = np.sqrt(max_val) if max_val >= 0 else 0

                    # Normalize to [0, 1] range
                    if transformed_max == transformed_min:
                        normalized_values = np.full_like(transformed_values, 0.5)
                    else:
                        normalized_values = (transformed_values - transformed_min) / (transformed_max - transformed_min)

                elif method == "sigmoid":
                    # Sigmoid normalization
                    # First, scale to [-6, 6] to utilize the most sensitive part of sigmoid
                    if max_val == min_val:
                        scaled_values = np.zeros_like(values_array)
                    else:
                        scaled_values = 12 * (values_array - min_val) / (max_val - min_val) - 6

                    # Apply sigmoid: 1 / (1 + exp(-x))
                    normalized_values = 1 / (1 + np.exp(-scaled_values))

                else:  # "linear" or any other value
                    # Standard linear normalization
                    if max_val == min_val:
                        normalized_values = np.full_like(values_array, 0.5)
                    else:
                        normalized_values = (values_array - min_val) / (max_val - min_val)

                # Update items with normalized values
                for idx, norm_value in zip(valid_indices, normalized_values):
                    items[idx]["properties"][f"{field}_normalized"] = float(norm_value)

                # Calculate some statistics for the normalized values
                if len(normalized_values) > 0:
                    # Find the normalized value for some reference points (if they exist in the data)
                    reference_points = [100, 1000, 10000, 100000, 1000000]
                    reference_info = []

                    for ref_point in reference_points:
                        if min_val <= ref_point <= max_val:
                            # Find the closest value to the reference point
                            closest_idx = np.argmin(np.abs(values_array - ref_point))
                            closest_val = values_array[closest_idx]
                            norm_val = normalized_values[closest_idx]
                            reference_info.append(f"{ref_point:.0f} → {norm_val:.4f}")

                    if reference_info:
                        self.logger.info(f"Reference points for {field}: {', '.join(reference_info)}")

                self.logger.info(f"Normalized field {field}: min={min_val}, max={max_val}, count={len(field_values)}")

            return items
        except Exception as e:
            self.logger.error(f"Error in normalize_by_field: {e}")
            return items

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
        """Create user nodes in the database with offensive content analysis and normalized values"""
        # Process each user to add offensive content analysis
        for user in users:
            properties = user.get("properties", {})
            user_id = user.get("id", "")

            # Use pre-computed offensive content analysis for user description
            if user_id in self.user_classifications:
                classification_result = self.user_classifications[user_id]
                properties["description_offensive"] = classification_result["classification"]
                properties["description_offensive_confidence"] = classification_result["confidence"]
                self.logger.debug(f"Applied pre-computed classification for user {user_id}: {classification_result['classification']}")
            else:
                # Default to not_offensive if no classification is available
                properties["description_offensive"] = "not_offensive"
                properties["description_offensive_confidence"] = 1.0
                self.logger.debug(f"No pre-computed classification found for user {user_id}, using default")

            # Add bullying-related counters (initially set to 0)
            properties["bullying_comments_count"] = 0
            properties["bullying_mentions_count"] = 0

            # Update the properties in the user dictionary
            user["properties"] = properties

        # Normalize numerical values across all users using log normalization
        numerical_fields = ["followerCount", "followingCount", "likeCount", "postCount"]
        users = self.normalize_by_field(users, numerical_fields, method="log")

        # Create user nodes in the database
        query = """
        UNWIND $users AS user
        MERGE (u:User {id: user.id})
        SET u += user.properties
        """
        with self.driver.session() as session:
            session.run(query, users=users)
            self.logger.info(f"Created {len(users)} user nodes")

    def create_media_session_nodes(self, media_sessions: List[Dict]):
        """Create media session nodes with normalized values"""
        # Process each media session
        for media in media_sessions:
            properties = media.get("properties", {})
            media_id = media.get("id", "")

            # Remove aggression_confidence, aggression, bullying, and bullying_confidence properties
            if "aggression_confidence" in properties:
                del properties["aggression_confidence"]
            if "aggression" in properties:
                del properties["aggression"]
            if "bullying" in properties:
                del properties["bullying"]
            if "bullying_confidence" in properties:
                del properties["bullying_confidence"]

            # Use pre-computed offensive content analysis for media description
            if media_id in self.video_classifications:
                classification_result = self.video_classifications[media_id]
                properties["description_offensive"] = classification_result["classification"]
                properties["description_offensive_confidence"] = classification_result["confidence"]
                self.logger.debug(f"Applied pre-computed classification for media {media_id}: {classification_result['classification']}")
            else:
                # Default to not_offensive if no classification is available
                properties["description_offensive"] = "not_offensive"
                properties["description_offensive_confidence"] = 1.0
                self.logger.debug(f"No pre-computed classification found for media {media_id}, using default")

            # Update the properties in the media session dictionary
            media["properties"] = properties

        # Normalize numerical values across all media sessions using log normalization
        numerical_fields = ["likeCount", "commentCount", "loopCount", "repostCount"]
        media_sessions = self.normalize_by_field(media_sessions, numerical_fields, method="log")

        # Create media session nodes in the database
        query = """
        UNWIND $sessions AS session
        MERGE (m:MediaSession {id: session.id})
        SET m += session.properties
        """
        with self.driver.session() as session:
            session.run(query, sessions=media_sessions)
            self.logger.info(f"Created {len(media_sessions)} media session nodes")

    def create_comment_nodes(self, comments: List[Dict]):
        """Create comment nodes with sentiment analysis and aggressive vocabulary analysis"""
        self.logger.info("Processing comments with pre-computed classifications and Detoxify analysis...")

        # 使用tqdm创建进度条
        total_comments = len(comments)
        processed_count = 0
        start_time = time.time()

        # Process each comment with progress bar
        for comment in tqdm(comments, desc="处理评论", unit="条"):
            properties = comment.get("properties", {})
            comment_id = comment.get("id", "")

            # Use pre-computed offensive content analysis for comment text
            if comment_id in self.comment_classifications:
                classification_result = self.comment_classifications[comment_id]
                properties["offensive"] = classification_result["classification"]
                properties["offensive_confidence"] = classification_result["confidence"]
                self.logger.debug(f"Applied pre-computed classification for comment {comment_id}: {classification_result['classification']}")
            else:
                # Default to not_offensive if no classification is available
                properties["offensive"] = "not_offensive"
                properties["offensive_confidence"] = 1.0
                self.logger.debug(f"No pre-computed classification found for comment {comment_id}, using default")

            # 移除了攻击性分析相关代码

            # Update the properties in the comment dictionary
            comment["properties"] = properties

            # 更新处理计数和显示速度
            processed_count += 1
            if processed_count % 100 == 0:
                elapsed_time = time.time() - start_time
                comments_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                self.logger.info(f"已处理 {processed_count}/{total_comments} 条评论，速度: {comments_per_second:.2f} 条/秒")

        # Normalize numerical values across all comments if needed
        # Currently, comments don't have numerical fields that need normalization
        # But if they do in the future, we can add them here
        # numerical_fields = ["replyCount", "likeCount"]
        # comments = self.normalize_by_field(comments, numerical_fields)

        self.logger.info(f"Completed processing {len(comments)} comments")

        # Create comment nodes in the database
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
        """Create relationships between nodes with enhanced properties"""
        # Process relationships to add enhanced properties
        if "mentions" in relationships:
            # Get all comment nodes to access their offensive content
            comment_data = {}
            with self.driver.session() as session:
                result = session.run("MATCH (c:Comment) RETURN c.id AS id, c.offensive AS offensive")
                for record in result:
                    comment_id = record["id"]
                    offensive = record.get("offensive", "not_offensive")
                    comment_data[comment_id] = {
                        "offensive": offensive
                    }

            # Process mentions relationships to classify them based on offensive content
            self.logger.info(f"处理 {len(relationships['mentions'])} 条mentions关系...")
            for rel in tqdm(relationships["mentions"], desc="处理mentions关系", unit="条"):
                comment_id = rel["source"]
                if comment_id in comment_data:
                    offensive = comment_data[comment_id]["offensive"]

                    # Classify based on offensive content
                    if offensive == "offensive":
                        # Offensive content - clear attack
                        rel["properties"]["attack_type"] = "clear_attack"
                    else:
                        # Non-offensive content - non-attack
                        rel["properties"]["attack_type"] = "non_attack"

                    rel["properties"]["offensive"] = offensive

        # Process creates relationships to record publishing user
        if "creates" in relationships:
            for rel in relationships["creates"]:
                rel["properties"]["is_publisher"] = True

        # Process belongs_to relationships to connect comments to media sessions
        if "belongs_to" in relationships:
            for rel in relationships["belongs_to"]:
                rel["properties"]["connected"] = True

        # Log relationship statistics
        for rel_type, rels in relationships.items():
            self.logger.info(f"Processing {rel_type} relationships. Total count: {len(rels)}")
            if rels:
                sample_size = min(5, len(rels))
                self.logger.info(f"Sample {rel_type} relationships: {json.dumps(rels[:sample_size], indent=2)}")

        # Define relationship queries
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
            """,
            "mentions": """
            // Comment -> User
            UNWIND $rels AS rel
            MATCH (c:Comment {id: rel.source})
            MATCH (u:User {id: rel.target})
            MERGE (c)-[r:MENTIONS {attack_type: rel.properties.attack_type}]->(u)
            SET r.created = rel.properties.created,
                r.offensive = rel.properties.offensive
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

        # Create relationships in the database
        with self.driver.session() as session:
            # 使用tqdm显示关系创建进度
            total_rels = sum(len(rels) for rels in relationships.values())
            self.logger.info(f"开始创建 {total_rels} 条关系...")

            # 记录开始时间
            start_time = time.time()
            processed_count = 0

            for rel_type, rels in relationships.items():
                if rel_type in relationship_queries:
                    try:
                        # 执行关系创建
                        self.logger.info(f"创建 {len(rels)} 条 {rel_type} 关系...")
                        session.run(relationship_queries[rel_type], rels=rels)

                        # 更新处理计数和显示速度
                        processed_count += len(rels)
                        elapsed_time = time.time() - start_time
                        rels_per_second = processed_count / elapsed_time if elapsed_time > 0 else 0
                        self.logger.info(f"已创建 {processed_count}/{total_rels} 条关系，速度: {rels_per_second:.2f} 条/秒")

                        self.logger.info(f"Created {len(rels)} {rel_type} relationships")
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

            self.logger.info("Knowledge graph built successfully")

        except Exception as e:
            self.logger.error(f"Error building knowledge graph: {e}")
            raise

def load_integration_data():
    """Load data from the integration directory"""
    data = {
        "nodes": {
            "users": [],
            "media_sessions": [],
            "comments": [],
            "labels": []  # Will be populated with emotion, theme, aggression, bullying labels
        },
        "relationships": {
            "publishes": [],  # User -> MediaSession
            "creates": [],    # User -> Comment
            "mentions": [],   # Comment -> User
            "annotates": [],  # MediaSession -> Label
            "belongs_to": []  # Comment -> MediaSession
        }
    }

    # Create label nodes for bullying only
    label_types = {
        "bullying": ["bullying", "noneBll"]
    }

    # Create label nodes
    label_id_counter = 1
    label_id_map = {}  # Maps label_type:label_value to label_id

    for label_type, values in label_types.items():
        for value in values:
            label_id = f"label_{label_id_counter}"
            label_node = {
                "id": label_id,
                "properties": {
                    "type": label_type,
                    "value": value
                }
            }
            data["nodes"]["labels"].append(label_node)
            label_id_map[f"{label_type}:{value}"] = label_id
            label_id_counter += 1

    # Load users
    print("Loading users...")
    with open("data/processed/integration/users.json", 'r', encoding='utf-8') as f:
        users_data = json.load(f)
        for user in users_data:
            user_node = {
                "id": user["userId"],
                "properties": user
            }
            data["nodes"]["users"].append(user_node)
    print(f"Loaded {len(data['nodes']['users'])} users")

    # Load media sessions
    print("Loading media sessions...")
    with open("data/processed/integration/media_sessions.json", 'r', encoding='utf-8') as f:
        media_sessions_data = json.load(f)
        for media in media_sessions_data:
            media_node = {
                "id": media["postId"],
                "properties": media
            }
            data["nodes"]["media_sessions"].append(media_node)

            # Create publishes relationship (User -> MediaSession)
            if "userId" in media:
                rel = {
                    "source": media["userId"],
                    "target": media["postId"],
                    "properties": {
                        "created": media.get("created", "")
                    }
                }
                data["relationships"]["publishes"].append(rel)

            # Create annotates relationships (MediaSession -> Label)
            # For bullying label - default to noneBll if not specified
            bullying_value = media.get("bullying", "noneBll")
            bullying_label_id = label_id_map.get(f"bullying:{bullying_value}")
            if bullying_label_id:
                rel = {
                    "source": media["postId"],
                    "target": bullying_label_id,
                    "properties": {}  # No confidence needed
                }
                data["relationships"]["annotates"].append(rel)

    print(f"Loaded {len(data['nodes']['media_sessions'])} media sessions")
    print(f"Created {len(data['relationships']['publishes'])} publishes relationships")
    print(f"Created {len(data['relationships']['annotates'])} annotates relationships")

    # Load comments
    print("Loading comments...")
    with open("data/processed/integration/comments.json", 'r', encoding='utf-8') as f:
        comments_data = json.load(f)
        for comment in comments_data:
            comment_node = {
                "id": comment["commentId"],
                "properties": comment
            }
            data["nodes"]["comments"].append(comment_node)

            # Create creates relationship (User -> Comment)
            if "userId" in comment:
                rel = {
                    "source": comment["userId"],
                    "target": comment["commentId"],
                    "properties": {
                        "created": comment.get("created", "")
                    }
                }
                data["relationships"]["creates"].append(rel)

            # Create belongs_to relationship (Comment -> MediaSession)
            if "postId" in comment:
                rel = {
                    "source": comment["commentId"],
                    "target": comment["postId"],
                    "properties": {
                        "created": comment.get("created", "")
                    }
                }
                data["relationships"]["belongs_to"].append(rel)

            # Create mentions relationship (Comment -> User)
            if "mentionId" in comment and comment["mentionId"]:
                rel = {
                    "source": comment["commentId"],
                    "target": comment["mentionId"],
                    "properties": {
                        "created": comment.get("created", ""),
                        "attack_type": "non_attack"  # 默认为非攻击性提及
                    }
                }
                data["relationships"]["mentions"].append(rel)
    print(f"Loaded {len(data['nodes']['comments'])} comments")
    print(f"Created {len(data['nodes']['labels'])} label nodes")
    print(f"Created {len(data['relationships']['creates'])} creates relationships")
    print(f"Created {len(data['relationships']['belongs_to'])} belongs_to relationships")
    print(f"Created {len(data['relationships']['mentions'])} mentions relationships")
    print(f"Created {len(data['relationships']['annotates'])} annotates relationships")

    return data

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
        # Load data from integration directory
        print("Loading data from integration directory...")
        data = load_integration_data()

        # Save the data to a temporary file
        temp_file = "data/processed/integration/temp_integration_data.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Build the knowledge graph using the temporary file
        builder.build_knowledge_graph(temp_file)
    finally:
        builder.close()

if __name__ == "__main__":
    main()
