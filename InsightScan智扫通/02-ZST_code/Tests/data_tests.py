import os
import sys
import pandas as pd

# 添加项目根目录到 sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 使用绝对路径读取 CSV，确保在任何目录下运行都有效
csv_path = os.path.join(root_dir, 'data', 'query_from_others', 'query_data.csv')
df = pd.read_csv(csv_path)

# 转换为嵌套列表
data_list = df.values.tolist()
print(data_list[:3])  # 打印前3行查看