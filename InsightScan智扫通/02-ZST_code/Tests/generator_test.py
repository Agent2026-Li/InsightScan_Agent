import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.ReportGenerationTool import generator

b = generator("生成2023年1月份的销售报告。")
print(b)