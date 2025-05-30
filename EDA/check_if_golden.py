import pandas as pd

# 读取 CSV 文件
df = pd.read_csv("data/label_data/vine_labeled_cyberbullying_data.csv")

# 筛选 _golden 列为 True 的行
filtered_df = df[df['_golden'] == True]

# 统计 question2 列为 'bullying' 的数量
bullying_count = (filtered_df['question2'] == 'bullying').sum()

print(f"数量为 True 的行中，question2 列为 'bullying' 的行数: {bullying_count}")