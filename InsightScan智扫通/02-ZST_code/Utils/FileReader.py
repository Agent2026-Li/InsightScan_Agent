import os

from langchain_community.document_loaders import UnstructuredWordDocumentLoader,PyPDFLoader
import yaml
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("项目根目录为：%s" % project_root)
# 文档读取工具类
class FileReader:
    def __init__(self) -> None:
        """
        初始化文件读取工具
        """
        pass

    def read_docx(self,file_path):
        """
        读取docx文件内容
        Args:
            file_path: 文件路径
        Returns:
        """
        # 加载docx文件，并返回一个列表，列表中的元素是文档对象
        loader = UnstructuredWordDocumentLoader(file_path)
        doc = loader.load_and_split()
        return doc

    def read_md(self,file_path):
        """
        读取md文件内容
        Args:
            file_path: 文件路径
        Returns:
        """
        try:
            # 读取文件内容
            with open(file_path,'r',encoding='utf-8') as f:
                content = f.read()
                return content
        except FileNotFoundError:
            print(f'错误：文件{file_path}未找到')
            return None
        except Exception as e:
            print(f'读取文件时出错：{e}')
            return None
    
    def read_pdf(self,file_path):
        """
        读取pdf文件内容
        Args:
            file_path: 文件路径
        Returns:
        """
        loader = PyPDFLoader(file_path)
        # 加载pdf文件，并返回一个列表，列表中的元素是文档对象
        pages = loader.load_and_split()
        return pages

    def read_csvs(self,file_path):
        """
        读取csv文件内容
        Args:
            file_path: 文件路径
        Returns:
            pandas.DataFrame: 包含csv文件内容的DataFrame
        """
        data = pd.read_csv(file_path)
        return data

# 配置文件读取工具类
class ConfigHandler:
    def __init__(self) -> None:
        pass   

    def read_yaml(self, file_name="rag_config.yaml", encoding='utf-8'):
        """
        读取rag_config.yaml文件内容
        Args:
            file_path: 文件路径
            encoding: 编码方式
        Returns:
            dict: 包含yaml文件内容的字典
        """
        file_path = os.path.join(project_root, "Configs", file_name)
        with open(file_path,'r',encoding=encoding) as f:
            return yaml.load(f.read(),Loader=yaml.FullLoader)

    def get_project_root(self):
        """
        获取项目根目录
        Returns:
            str: 项目根目录
        """
        return project_root
 
if __name__ == '__main__':
    reader = FileReader()
    config_reader = ConfigHandler()
    # config_yaml = config_reader.read_yaml(os.path.join(os.path.dirname(os.getcwd()),"Configs/rag_config.yml"))
    # print(os.path.dirname(os.getcwd()))
    model_conf = config_reader.read_yaml(os.path.join(os.path.dirname(os.getcwd()),"Configs/config.yml"))
    print(f">>> 读取模型配置：{model_conf}")
    # 查看模型名称
    print(f">>> model_name:{model_conf['model_name']}")
    # 查看Embedding模型名称
    print(f">>> embedding_name：{model_conf['embedding_name']}")
