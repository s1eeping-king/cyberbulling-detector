# Neo4j 导入说明

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
