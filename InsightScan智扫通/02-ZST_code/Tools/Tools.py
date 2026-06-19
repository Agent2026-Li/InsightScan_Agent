import warnings
warnings.filterwarnings("ignore")

# 核心修复：在新版 LangChain (v0.3+) 中，StructuredTool 已经完全迁移到 core 包下
from langchain_core.tools import StructuredTool
from .RagQATool import rag_qa
from .ReportGenerationTool import generator

rag_qa_tool = StructuredTool.from_function(
    func=rag_qa,
    name="RagQA",
    description="使用RAG检索知识进行问答"
)

report_generation_tool = StructuredTool.from_function(
    func=generator,
    name="ReportGeneration",
    description="从第三方系统查询信息，完成报告"
)