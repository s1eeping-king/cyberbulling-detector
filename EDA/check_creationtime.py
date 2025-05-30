import pandas as pd

# 加载数据
file_path = 'data/label_data/vine_labeled_cyberbullying_data.csv'
df = pd.read_csv(file_path)

# 打印creationtime列的前10个值
print("creationtime列的前10个值:")
for i, value in enumerate(df['creationtime'].head(10)):
    print(f"{i+1}. {value}")

# 检查是否有空值
null_count = df['creationtime'].isnull().sum()
print(f"\n空值数量: {null_count} (占总记录的 {null_count/len(df):.2%})")

# 检查唯一值的数量
unique_count = df['creationtime'].nunique()
print(f"唯一值数量: {unique_count} (占总记录的 {unique_count/len(df):.2%})")

# 检查数据类型
print(f"数据类型: {df['creationtime'].dtype}")

# 如果是字符串类型，检查长度分布
if df['creationtime'].dtype == 'object':
    df['length'] = df['creationtime'].astype(str).str.len()
    length_stats = df['length'].describe()
    print(f"\n字符串长度统计:\n{length_stats}")
    
    # 打印一些不同长度的示例
    print("\n不同长度的示例:")
    for length in sorted(df['length'].unique())[:5]:  # 只打印前5种不同长度
        sample = df[df['length'] == length]['creationtime'].iloc[0]
        print(f"长度 {length}: {sample}")
