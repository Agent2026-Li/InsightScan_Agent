import os
import sys
# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tools.RagQATool import rag_qa

print(rag_qa("为什么有些机器人会'迷路'？"))