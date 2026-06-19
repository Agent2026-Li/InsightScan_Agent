import os
from Models.Factory import ChatModelFactory  # 👈 修复了这里：补全了顶部的 from Models
from Tools.VecStore import VecStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

vec = VecStore()
db = vec.load_chroma()
def rag_qa(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    response = qa.run(query)
    return response