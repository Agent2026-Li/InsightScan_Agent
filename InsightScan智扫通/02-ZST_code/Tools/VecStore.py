"""
向量数据库工具模块：负责 RAG (检索增强生成) 流程中的数据准备工作
包括：本地文件读取、文本分块 (Chunking)、向量化 (Embedding) 以及向量数据库 (Chroma) 的全生命周期管理
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
import os
import sys

from Utils.FileReader import FileReader, ConfigHandler
from Models.Factory import embed_model

# 初始化文件读取器和配置处理器
reader = FileReader()
config_reader = ConfigHandler()
# NOTE: rag_config.yml 定义了数据存放路径和持久化路径
config_yaml = config_reader.read_yaml("rag_config.yml")
print(f">>> 读取模型配置：{config_yaml}")

class VecStore:
    """
    向量存储中心：管理文档的入库、持久化与加载
    """
    def __init__(self, dir=config_yaml['DATA_PATH'], save_dir=config_yaml['CHROMA_PATH']) -> None:
        """
        Args:
            dir: 待处理的原始文档目录（如 data/）
            save_dir: Chroma 数据库的持久化存储目录（如 Chroma/）
        """
        # 构建绝对路径，确保在不同运行环境下路径引用的一致性
        self.dir = os.path.join(config_reader.get_project_root(), dir)
        print(f"加载向量数据所在目录：{self.dir}")
        self.save_dir = os.path.join(config_reader.get_project_root(), save_dir)
        print(f"保存向量数据库目录：{self.save_dir}")

    def get_db(self):
        """
        一键构建并返回向量数据库
        Returns:
            Chroma: 包含分块数据和嵌入信息的向量数据库对象
        """
        print(f"开始加载文档...")
        # 1. 从磁盘加载不同格式的文件
        documents = self.load_documents()
        # 2. 对长文本进行分块，以便 LLM 能够处理并减少上下文压力
        chunks = self.split_text(documents)
        # 3. 将分块后的 Document 列表转化为向量并存入 Chroma
        chroma_db = Chroma.from_documents(
            documents=chunks,
            embedding=embed_model, # 使用 Factory 中定义的嵌入模型
            persist_directory=self.save_dir
        )
        return chroma_db

    def load_documents(self):
        """
        多格式文档加载器：支持 docx、pdf 和 csv
        """
        files = os.listdir(self.dir)
        docs = []
        for file in files:
            file_path = os.path.join(self.dir, file)
            # 根据后缀名调用 Utils.FileReader 中的对应方法
            if file.endswith('.docx'):
                txts = reader.read_docx(file_path)
            elif file.endswith('.pdf'):
                txts = reader.read_pdf(file_path)
            elif file.endswith('.csv'):
                df = reader.read_csvs(file_path)
                txts = df.values.tolist() # CSV 通常需要特殊处理或映射为文本描述
            else:
                txts = []
            docs.extend(txts)
        return docs

    def split_text(self, documents):
        """
        文本切分逻辑
        Args:
            documents: 原始 Document 对象列表
        Returns:
            list[Document]: 切分后的短文本块列表
        """
        # 使用递归字符切分器，优先在段落、句子边界切分，保留语义完整性
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, # 每个分块的最大字符数
            chunk_overlap=100, # 相邻分块间的重叠字符数，防止切断上下文
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def save_to_chroma(self, db):
        """
        数据持久化到本地磁盘
        """
        db.persist()
        print(f"Saved chunks to {config_yaml['CHROMA_PATH']}.")

    def load_chroma(self):
        """
        直接从磁盘读取现有的向量数据库（不需要重新分块和转换）
        """
        db = Chroma(persist_directory=self.save_dir, embedding_function=embed_model)
        return db

# 脚本模式启动：直接运行此文件可进行数据库索引构建
if __name__ == "__main__":
    vec = VecStore()
    db = vec.get_db()
    vec.save_to_chroma(db)
