import pickle
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import torch

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_tensor_to_list(value):
    """将tensor转换为Python原生类型"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, dict):
        return {k: convert_tensor_to_list(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [convert_tensor_to_list(item) for item in value]
    return value

def process_properties(properties):
    """处理节点或关系的属性"""
    processed = {}
    for k, v in properties.items():
        if k == 'node_type' or k == 'relation':
            continue
        
        # 转换tensor和numpy数组
        v = convert_tensor_to_list(v)
        
        # 处理特征向量，计算统计值
        if isinstance(v, (list, np.ndarray)) and len(v) > 0:
            if isinstance(v[0], (int, float)):
                processed[f"{k}_mean"] = float(np.mean(v))
                processed[f"{k}_std"] = float(np.std(v))
                processed[f"{k}_min"] = float(np.min(v))
                processed[f"{k}_max"] = float(np.max(v))
            continue
        
        # 处理基本类型
        if isinstance(v, (str, int, float, bool)):
            processed[k] = v
        elif isinstance(v, dict):
            # 展平字典
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, (str, int, float, bool)):
                    processed[f"{k}_{sub_k}"] = sub_v
    
    return processed

def create_cypher_node_statement(node_type, properties):
    """创建节点的Cypher语句"""
    props = []
    for col in properties:
        if col != 'id':
            props.append(f"{col}: row.{col}")
    
    base = f"CREATE (:{node_type} {{id: row.id"
    if props:
        return base + ", " + ", ".join(props) + "})"
    return base + "})"

def create_cypher_relationship_statement(edge_type, properties):
    """创建关系的Cypher语句"""
    props = []
    for col in properties:
        if col not in ['source', 'target']:
            props.append(f"{col}: row.{col}")
    
    if props:
        return f"CREATE (source)-[:{edge_type} {{{', '.join(props)}}}]->(target)"
    return f"CREATE (source)-[:{edge_type}]->(target)"

def export_to_csv(graph, output_dir='knowledge_graph/neo4j_import'):
    """
    将图导出为Neo4j可导入的CSV文件
    
    Args:
        graph: NetworkX图对象
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按类型收集节点
    nodes_by_type = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get('node_type', 'Unknown')
        if node_type not in nodes_by_type:
            nodes_by_type[node_type] = []
        
        # 处理节点属性
        properties = process_properties(data)
        properties['id'] = node  # 添加节点ID
        nodes_by_type[node_type].append(properties)
    
    # 导出节点CSV文件
    logger.info("Exporting nodes...")
    for node_type, nodes in nodes_by_type.items():
        if not nodes:
            continue
            
        df = pd.DataFrame(nodes)
        filename = output_path / f"{node_type.lower()}_nodes.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(nodes)} {node_type} nodes to {filename}")
    
    # 收集边
    edges_by_type = {}
    for source, target, data in graph.edges(data=True):
        edge_type = data.get('relation', 'UNKNOWN')
        if edge_type not in edges_by_type:
            edges_by_type[edge_type] = []
        
        # 处理边属性
        properties = process_properties(data)
        edge_data = {
            'source': source,
            'target': target,
            **properties
        }
        edges_by_type[edge_type].append(edge_data)
    
    # 导出边CSV文件
    logger.info("Exporting relationships...")
    for edge_type, edges in edges_by_type.items():
        if not edges:
            continue
            
        df = pd.DataFrame(edges)
        filename = output_path / f"{edge_type.lower()}_rels.csv"
        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(edges)} {edge_type} relationships to {filename}")
    
    # 生成Cypher导入脚本
    import_script = []
    
    # 添加配置说明
    import_script.append("""// 在执行导入之前，请确保：
// 1. 所有CSV文件已复制到Neo4j的import目录
// 2. 已设置允许导入本地文件，方法如下：
//    方法1：在neo4j.conf中设置：
//    dbms.security.allow_csv_import_from_file_urls=true
//    
//    方法2：使用APOC库（推荐）：
//    CALL apoc.import.csv(...)
""")
    
    # 添加清理数据库的命令（可选）
    import_script.append("""// 清空数据库（可选，请谨慎使用）
// MATCH (n) DETACH DELETE n;
""")
    
    # 添加约束创建（如果需要）
    import_script.append("""// 创建约束
CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (m:MediaSession) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.id IS UNIQUE;
""")
    
    # 节点导入命令
    for node_type, nodes in nodes_by_type.items():
        if not nodes:
            continue
        
        create_statement = create_cypher_node_statement(node_type, nodes[0].keys())
        import_script.append(f"""// 导入{node_type}节点
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/{node_type.lower()}_nodes.csv' AS row
{create_statement};
""")
    
    # 边导入命令
    for edge_type, edges in edges_by_type.items():
        if not edges:
            continue
        
        create_statement = create_cypher_relationship_statement(edge_type, edges[0].keys())
        import_script.append(f"""// 导入{edge_type}关系
LOAD CSV WITH HEADERS FROM 'file:///neo4j_import/{edge_type.lower()}_rels.csv' AS row
MATCH (source {{id: row.source}})
MATCH (target {{id: row.target}})
{create_statement};
""")
    
    # 保存导入脚本
    with open(output_path / 'import_script.cypher', 'w', encoding='utf-8') as f:
        f.write('\n'.join(import_script))
    
    # 生成详细的导入说明
    with open(output_path / 'README.md', 'w', encoding='utf-8') as f:
        f.write("""# Neo4j 导入说明

## 准备工作

1. 找到 Neo4j 的 import 目录
   - Windows: `C:/Users/<username>/AppData/Local/Neo4j/Relate/Data/dbmss/<dbname>/import/`
   - Linux: `/var/lib/neo4j/import/`
   - Docker: `/var/lib/neo4j/import/`

2. 创建目标目录
   ```bash
   mkdir -p <neo4j-import-dir>/neo4j_import
   ```

3. 复制所有CSV文件到Neo4j的import目录
   ```bash
   cp *.csv <neo4j-import-dir>/neo4j_import/
   ```

## 导入方法

### 方法1：使用Neo4j Browser

1. 打开 Neo4j Browser
2. 打开 `import_script.cypher` 文件
3. 逐条执行其中的命令

### 方法2：使用APOC库（推荐）

如果遇到文件访问权限问题，可以使用APOC库：

1. 确保已安装APOC插件
2. 使用以下命令格式导入：
   ```cypher
   CALL apoc.import.csv([
     {fileName: 'neo4j_import/video_nodes.csv', labels: ['Video']},
     {fileName: 'neo4j_import/user_nodes.csv', labels: ['User']},
     ...
   ], {})
   ```

## 常见问题

1. 文件访问错误
   - 检查文件权限
   - 确认文件路径正确
   - 在neo4j.conf中设置：`dbms.security.allow_csv_import_from_file_urls=true`

2. 内存不足
   - 使用 PERIODIC COMMIT
   - 增加Neo4j的堆内存设置

3. 约束冲突
   - 先删除现有数据
   - 删除现有约束
   - 重新创建约束
""")
    
    logger.info(f"Export completed. Files saved in {output_path}")
    logger.info("Please check README.md for detailed import instructions")

def main():
    try:
        # 读取知识图谱
        logger.info("Loading knowledge graph...")
        with open('knowledge_graph/knowledge_graph.pkl', 'rb') as f:
            graph = pickle.load(f)
        
        # 导出为CSV
        export_to_csv(graph)
        
    except Exception as e:
        logger.error(f"Error in export process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 