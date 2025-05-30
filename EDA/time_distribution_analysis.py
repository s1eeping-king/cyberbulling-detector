import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
import seaborn as sns
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def extract_creation_time(text):
    """
    从creationtime列中提取时间信息
    示例输入1: "Media posted at:2014-12-16T02:39:48.000000"
    示例输入2: "<font color=""#0066CC"">// 'Heva' //</font>::Lycia Faith (created_at:2014-09-18T20:09:12.000000)"
    """
    try:
        # 转换为字符串以防万一
        text = str(text)

        # 尝试匹配 "Media posted at:" 格式
        media_match = re.search(r'Media posted at:(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text)
        if media_match:
            time_str = media_match.group(1)
            # 解析时间字符串为datetime对象
            return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # 尝试匹配 "created_at:" 格式
        created_match = re.search(r'created_at:(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text)
        if created_match:
            time_str = created_match.group(1)
            # 解析时间字符串为datetime对象
            return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        # 如果都不匹配，尝试直接匹配日期格式
        date_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', text)
        if date_match:
            time_str = date_match.group(1)
            # 解析时间字符串为datetime对象
            return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')

        return None
    except Exception as e:
        # 不打印错误信息，以减少输出量
        # print(f"Error processing: {text}")
        # print(f"Error message: {e}")
        return None

def main():
    # 加载数据
    file_path = 'data/label_data/vine_labeled_cyberbullying_data.csv'

    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"正在加载数据: {file_path}")
    df = pd.read_csv(file_path)

    # 查看数据基本信息
    print(f"数据集大小: {df.shape}")
    print("\n前5行数据:")
    print(df.head())

    # 检查creationtime列是否存在
    if 'creationtime' not in df.columns:
        print("\n'creationtime'列不存在，请检查列名")
        # 打印所有列名以便参考
        print("可用列名:", df.columns.tolist())
        return

    # 提取时间信息
    print("\n正在提取时间信息...")
    df['creation_datetime'] = df['creationtime'].apply(extract_creation_time)

    # 移除无法解析的时间
    valid_time_df = df.dropna(subset=['creation_datetime'])
    print(f"成功提取时间的记录数: {valid_time_df.shape[0]} (占总记录的 {valid_time_df.shape[0]/df.shape[0]:.2%})")

    # 检查question2列是否存在
    if 'question2' not in valid_time_df.columns:
        print("\n'question2'列不存在，请检查列名")
        # 打印所有列名以便参考
        print("可用列名:", valid_time_df.columns.tolist())

        # 如果没有question2列，使用原来的代码
        # 提取年月信息用于按月分组
        valid_time_df['year_month'] = valid_time_df['creation_datetime'].dt.to_period('M')

        # 按月统计记录数
        monthly_counts = valid_time_df['year_month'].value_counts().sort_index()

        # 转换Period索引为字符串，以便更好地显示
        monthly_counts.index = monthly_counts.index.astype(str)

        # 创建图表
        plt.figure(figsize=(15, 8))

        # 使用Seaborn的barplot绘制更美观的柱状图
        ax = sns.barplot(x=monthly_counts.index, y=monthly_counts.values)

        # 设置图表标题和标签
        plt.title('Vine数据集按月创建时间分布', fontsize=16)
        plt.xlabel('年月', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)

        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')

        # 在每个柱子上显示具体数值
        for i, count in enumerate(monthly_counts.values):
            ax.text(i, count + 5, str(count), ha='center')
    else:
        # 提取年月信息用于按月分组
        valid_time_df['year_month'] = valid_time_df['creation_datetime'].dt.to_period('M')

        # 检查question2列的唯一值
        unique_values = valid_time_df['question2'].unique()
        print(f"\nquestion2列的唯一值: {unique_values}")

        # 确保question2列的值是我们期望的
        valid_time_df['bullying_type'] = valid_time_df['question2'].apply(
            lambda x: 'bullying' if 'bullying' in str(x).lower() else 'nonBll'
        )

        # 按月和霸凌类型分组统计
        grouped = valid_time_df.groupby(['year_month', 'bullying_type']).size().unstack(fill_value=0)

        # 确保两个类别都存在
        if 'bullying' not in grouped.columns:
            grouped['bullying'] = 0
        if 'nonBll' not in grouped.columns:
            grouped['nonBll'] = 0

        # 按月份排序
        grouped = grouped.sort_index()

        # 转换Period索引为字符串，以便更好地显示
        grouped.index = grouped.index.astype(str)

        # 创建图表
        plt.figure(figsize=(15, 8))

        # 创建堆叠柱状图
        ax = grouped.plot(kind='bar', stacked=True, figsize=(15, 8),
                         color=['#FF9999', '#66B2FF'],  # 红色代表霸凌，蓝色代表非霸凌
                         width=0.8)

        # 设置图表标题和标签
        plt.title('Vine数据集按月创建时间和霸凌类型分布', fontsize=16)
        plt.xlabel('年月', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)
        plt.legend(['霸凌 (bullying)', '非霸凌 (nonBll)'])

        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')

        # 在每个柱子上方显示总数
        for i, (idx, row) in enumerate(grouped.iterrows()):
            total = row.sum()
            ax.text(i, total + 2, str(int(total)), ha='center')

            # 在柱子中间显示各部分的数量
            bullying_height = row['bullying']
            nonbll_height = row['nonBll']

            # 只有当数量足够大时才显示数字
            if bullying_height > 5:
                ax.text(i, bullying_height/2, str(int(bullying_height)),
                       ha='center', va='center', color='white', fontweight='bold')

            if nonbll_height > 5:
                ax.text(i, bullying_height + nonbll_height/2, str(int(nonbll_height)),
                       ha='center', va='center', color='white', fontweight='bold')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    output_dir = 'EDA/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'vine_creation_time_monthly_distribution.png')
    plt.savefig(output_path)
    print(f"\n图表已保存至: {output_path}")

    # 显示图表
    plt.show()

    # 额外分析：按年统计
    valid_time_df['year'] = valid_time_df['creation_datetime'].dt.year
    yearly_counts = valid_time_df['year'].value_counts().sort_index()

    print("\n按年统计:")
    for year, count in yearly_counts.items():
        print(f"{year}: {count} 记录 ({count/valid_time_df.shape[0]:.2%})")

    # 额外分析：按星期几统计
    valid_time_df['weekday'] = valid_time_df['creation_datetime'].dt.day_name()

    # 按星期几的顺序排序
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # 检查是否有霸凌类型信息
    if 'bullying_type' in valid_time_df.columns:
        # 按星期几和霸凌类型分组统计
        weekday_grouped = valid_time_df.groupby(['weekday', 'bullying_type']).size().unstack(fill_value=0)

        # 确保两个类别都存在
        if 'bullying' not in weekday_grouped.columns:
            weekday_grouped['bullying'] = 0
        if 'nonBll' not in weekday_grouped.columns:
            weekday_grouped['nonBll'] = 0

        # 按星期几的顺序排序
        weekday_grouped = weekday_grouped.reindex(weekday_order)

        print("\n按星期几和霸凌类型统计:")
        for day, row in weekday_grouped.iterrows():
            total = row.sum()
            bullying = row['bullying']
            nonbll = row['nonBll']
            print(f"{day}: 总计 {total} 记录 ({total/valid_time_df.shape[0]:.2%}), "
                  f"霸凌 {bullying} ({bullying/total:.2%}), "
                  f"非霸凌 {nonbll} ({nonbll/total:.2%})")

        # 创建星期几分布图 - 按霸凌类型分块
        plt.figure(figsize=(12, 6))
        ax = weekday_grouped.plot(kind='bar', stacked=True, figsize=(12, 6),
                                color=['#FF9999', '#66B2FF'],  # 红色代表霸凌，蓝色代表非霸凌
                                width=0.8)
        plt.title('Vine数据集按星期几和霸凌类型分布', fontsize=16)
        plt.xlabel('星期', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)
        plt.legend(['霸凌 (bullying)', '非霸凌 (nonBll)'])

        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=0)

        # 在每个柱子上方显示总数
        for i, (day, row) in enumerate(weekday_grouped.iterrows()):
            total = row.sum()
            ax.text(i, total + 2, str(int(total)), ha='center')

            # 在柱子中间显示各部分的数量
            bullying_height = row['bullying']
            nonbll_height = row['nonBll']

            # 只有当数量足够大时才显示数字
            if bullying_height > 5:
                ax.text(i, bullying_height/2, str(int(bullying_height)),
                       ha='center', va='center', color='white', fontweight='bold')

            if nonbll_height > 5:
                ax.text(i, bullying_height + nonbll_height/2, str(int(nonbll_height)),
                       ha='center', va='center', color='white', fontweight='bold')
    else:
        # 如果没有霸凌类型信息，使用原来的代码
        weekday_counts = valid_time_df['weekday'].value_counts()
        weekday_counts = weekday_counts.reindex(weekday_order)

        print("\n按星期几统计:")
        for day, count in weekday_counts.items():
            print(f"{day}: {count} 记录 ({count/valid_time_df.shape[0]:.2%})")

        # 创建星期几分布图
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
        plt.title('Vine数据集按星期几分布', fontsize=16)
        plt.xlabel('星期', fontsize=14)
        plt.ylabel('记录数量', fontsize=14)

        # 在每个柱子上显示具体数值
        for i, count in enumerate(weekday_counts.values):
            ax.text(i, count + 5, str(count), ha='center')

    # 调整布局
    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(output_dir, 'vine_creation_time_weekday_distribution.png')
    plt.savefig(output_path)
    print(f"\n星期几分布图已保存至: {output_path}")

    # 显示图表
    plt.show()

if __name__ == "__main__":
    main()
