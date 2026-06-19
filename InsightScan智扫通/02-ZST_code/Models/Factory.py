from langchain_ollama import ChatOllama, OllamaEmbeddings
import os
import sys
# NOTE: 动态将项目根目录添加到 sys.path，保证无论从哪个子目录运行，都能正确导入 Utils, Tools 等模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.FileReader import ConfigHandler

# NOTE: 全局配置初始化。ConfigHandler 负责解析 config.yml 映射。
config_reader = ConfigHandler()
model_conf = config_reader.read_yaml("config.yml")

class ChatModelFactory:
    """
    聊天模型工厂类，负责根据配置文件创建符合 LangChain 规范的 Ollama 聊天实例
    """
    def __init__(self) -> None:
        # 从配置中提取本地 LLM 服务的 API 地址
        self.base_url = model_conf['server_url']

    def get_model(self):
        """
        创建并返回 ChatOllama 实例
        Returns:
            ChatOllama: 预配置的模型对象，可直接用于处理链或 Agent
        """
        # 指定模型名称（如 deepseek-r1:1.5b）及服务地址
        llm = ChatOllama(model=model_conf['model_name'], base_url=self.base_url)
        return llm

class EmbeddingModelFactory:
    """
    嵌入模型工厂类，负责创建用于向量化文本的 OllamaEmbedding 实例
    """
    def __init__(self) -> None:
        self.base_url = model_conf['server_url']

    def get_model(self):
        """
        创建并返回 OllamaEmbeddings 实例
        Returns:
            OllamaEmbeddings: 预配置的嵌入模型对象，用于 Chroma 数据库的向量化
        """
        # 指定嵌入模型名称（如 bge-m3:latest）
        embeddings = OllamaEmbeddings(model=model_conf['embedding_name'], base_url=self.base_url)
        return embeddings

# 导出全局单例模型实例，方便在全项目直接导入使用 (from Models.Factory import llm)
llm = ChatModelFactory().get_model()
embed_model = EmbeddingModelFactory().get_model()
