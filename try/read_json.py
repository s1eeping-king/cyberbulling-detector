import json

# 逐行读取 JSON 文件
data = []
with open('sampled_post-comments_vine.json', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            # 尝试解析每一行
            entry = json.loads(line)
            data.append(entry)
        except json.JSONDecodeError:
            print("无法解析这一行:", line)

# 输出第一条记录的内容
if data:
    first_entry = data[0]
    for key, value in first_entry.items():
        print(f"{key}: {value}")
else:
    print("没有有效的 JSON 数据。")