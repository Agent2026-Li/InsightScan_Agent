import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.FetchFromOtherSys import fetch_data_from_external_systems
from Tools.ExtractInfo import extract_info

a = extract_info("根据我的扫地机器人的使用情况，生成一份使用报告,要2025年6月份的")
b = fetch_data_from_external_systems(info_dict=a)
print(b)