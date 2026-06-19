import sys
# HACK: 动态添加路径以确保 Utils 模块在工具被单独调用或从 Agent 调用时都能正常导入
sys.path.append('../..')
sys.path.append('.')
sys.path.append('./')
from Utils.FileReader import FileReader
import pandas as pd
from datetime import datetime

# NOTE: 该函数模拟从第三方系统（如 ERP/CRM）提取业务数据
def fetch_data_from_external_systems(uuid=1001, info_dict={}):
    """
    根据传入的用户 ID 和时间信息，从外部 CSV 模拟数据库中检索数据
    Args:
        uuid (int): 用户 ID，标识唯一的业务主体
        info_dict (dict): 查询过滤器，如 {"时间": "2025-01"}
    Returns:
        pd.DataFrame or None: 包含查询结果的数据框
    """
    result = None
    try:
        # 使用项目统一的 FileReader 加载模拟数据
        reader = FileReader()
        # FIX: 使用原始字符串 (r'') 避免 Windows 路径下的转义字符冲突
        data = reader.read_csvs(r'data\query_from_others\query_data.csv')
        
        # 业务逻辑：如果指定了“时间”字段，则执行复合过滤；否则仅按 UUID 过滤
        if "时间" in info_dict.keys():
            time = info_dict['时间']
            result = data[(data['用户ID']==uuid) & (data['时间']==time)]
        else:
            result = data[data['用户ID']==uuid]
    except Exception as e:
        # 异常处理：捕获文件缺失、格式错误或 Pandas 逻辑异常，防止 Agent 流程中断
        print(f'从第三方系统获取信息出错:{e}')
    return result
