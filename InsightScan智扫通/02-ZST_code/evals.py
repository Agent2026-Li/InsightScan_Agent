"""
模型评估指标计算

在 LLM 开发中，尤其是使用像 deepseek-r1:1.5b 这种参数规模较小的模型时，出现解析异常是非常普遍的。以下是导致您在评估过程中看到“异常”的四个核心原因：

1. 模型指令遵循能力的局限 (Small Model Constraint)
    1.5b 规模的模型虽然有很强的推理能力，但对于严格 JSON 格式的遵循并不稳定。它有时会：
        误报 Schema：把参考文档中的 JSON Schema（描述字段的 JSON）当成结果输出了，而不是填入真实值。例如输出了 {"tool_name": {"title": "RagQA", ...}}。
        输出格式混淆：在 JSON 块之外夹杂了中文解释，或者忘记闭合反引号（```）。
2. 格式不一致 (Positional vs Keyword Args)
    代码预期：原始代码要求 args 必须是一个字典（如 {"query": "..."}）。
    模型输出：模型有时会按直觉输出一个列表（如 ["..."]），因为它觉得只有一个参数，没必要写键名。这在 Pydantic 校验中属于类型不匹配（List != Dict），从而引发 ValidationError。
3. 提示词冲突 (Prompt Instruction Conflict)
    在prompts/main.txt 中，有两条指令可能会产生冲突：
        指令 A：要求输出特定格式的 JSON 代码块。
        指令 B：如果无法回答，请直接返回“我不知道”。 当模型决定回答“我不知道”时，它可能跳过 JSON 格式直接输出纯文本。由于原本的代码逻辑是“必须解析为 Action 对象”，纯文本自然会导致解析器报错。
4. 思考链 (Chain of Thought) 的干扰
    DeepSeek-R1 系列模型会生成 <think>...</think> 思考过程。如果这部分内容太长，或者模型在思考结束后没有正确切换回 JSON 格式，传统的 LangChain 解析器就会因为噪音干扰而无法定位到合法的 JSON 字符串。
    我们是如何解决的？
    针对这些“异常”，我为您做了工程化加固（这也是 AI 应用开发中最关键的一步）：
        容错验证 (Action.py)：在 Pydantic 中增加了重置逻辑，无论模型给的是列表、字典还是 Schema 格式，代码都会自动将其“转换”为正确的字典。
        安全解析 (ReAct.py)：给解析器穿上了一层“防护服”（try-except）。如果模型输出实在太乱，代码不再崩溃退出，而是标记为“未知动作”并继续运行。
总结：这些异常不是因为代码写错了，而是因为 LLM 的不确定性。通过我们的加固，您的 Agent 现在已经具备了处理这种不确定性的能力。
"""
import json
import time
import numpy as np
import re
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer
import torch
from bert_score import BERTScorer
from bert_score.utils import lang2model, model2layers
from sklearn.metrics import precision_score, recall_score   # pip install scikit-learn
import pandas as pd
import jieba
import sys, os

sys.path.append('../..')
sys.path.append('.')
sys.path.append('./')
from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools.Tools import *
from Tools.RagQATool import rag_qa
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from tqdm import tqdm
import random
import ast

# 判断当前的环境是否支持cuda，若支持则使用cuda，否则使用cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
stopwords = []
# 加载停用词列表
with open(os.path.join(os.getcwd(), "stopwords.txt"), 'r', encoding='utf-8') as f:
    for line in f:
        stopwords.append(line.strip())

# 1. 准备评测数据
eval_data = pd.read_csv(os.path.join(os.getcwd(), 'Eval/eval_data.csv'))
print("eval_data:", eval_data)

# 2.获取agent
llm = ChatModelFactory().get_model()

# 定义工具列表
tools = [
    rag_qa_tool, # rag检索工具
    report_generation_tool, # 报告生成工具
]

# 创建代理实例
agent = ReActAgent(
    llm=llm, # 语言模型
    tools=tools, # 工具列表
    main_prompt_file_path="prompts/main.txt" # 主提示词文件路径
)

def launch_agent(query, agent, uuid=1001):
    """
    启动智能代理执行查询任务
    参数:
        query (str): 用户查询字符串
        agent: 智能代理实例
        uuid (int, optional): 用户唯一标识符，默认为1001
    返回值:
        str: 代理执行后的回复内容
    """
    chat_history = ChatMessageHistory()
    reply = agent.execute(query, uuid, chat_history=chat_history, verbose=True)
    return reply


# 3. 评测指标计算函数
def evaluate_retrieval(true_docs, retrieved_docs):
    """
    计算检索评估指标
    参数:
        true_docs (list): 真实文档列表
        retrieved_docs (list): 检索到的文档列表
    返回值:
        dict: 包含Precision@5、Recall@5和MRR指标的字典
    """
    y_true = [1 if doc in true_docs else 0 for doc in retrieved_docs]  # True
    y_pred = [1] * len(retrieved_docs)  # all

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred)
    mrr = 1 / (true_docs.index(retrieved_docs[0]) + 1) if retrieved_docs[0] in true_docs else 0
    return {
        "Precision@5": precision,
        "Recall@5": recall,
        "MRR": mrr}


def load_bert_scorer():
    """
    加载并初始化中文BERT评分器
    返回值:
        BERTScorer: 配置好的BERT评分器实例
    """
    # 加载自定义模型和分词器
    model_path = 'bert-base-chinese'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 手动加载模型配置
    num_layers = model2layers.get(os.path.basename(model_path), 12)
    model_type = lang2model.get('zh', 'bert-base-chinese')
    model = AutoModel.from_pretrained(model_path)

    return BERTScorer(
        model_type=model_type,
        num_layers=num_layers,
        lang='zh',
        use_fast_tokenizer=True,
        rescale_with_baseline=False,
        device=model.device)


def evaluate_generation(precision, reference, context):
    """
    计算文本生成质量评估指标
    参数:
        precision (str): 生成的文本（预测文本）
        reference (str): 参考文本（真实文本）
        context (list): 上下文文档列表
    返回值:
        dict: 包含ROUGE-1、ROUGE-L、BERTScore和Faithfulness指标的字典
    """
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge = scorer.score(reference, precision)

    # BERTScore
    bert_scorer = load_bert_scorer()
    _, _, bert_f1 = bert_scorer.score([precision], [reference])

    # 忠实度（简易版：检测关键实体是否一致）
    random.seed(42)
    key_entities = set([word for word in jieba.lcut(reference) if word not in stopwords])
    precision_entities = set([word for word in jieba.lcut(precision) if word not in stopwords])
    faithful = sum(1 for e in key_entities if e in precision_entities) / max(1, len(key_entities))

    return {
        'ROUGE-1': rouge['rouge1'].fmeasure,
        'ROUGE-L': rouge['rougeL'].fmeasure,
        'BERTScore': bert_f1.mean().item(),
        'Faithfulness': faithful}


def preprocess_text(text):
    """
    统一的文本预处理函数
    参数:
        text (str): 待处理的原始文本
    返回值:
        str: 预处理后的文本
    """
    # 转换为小写
    text = text.lower()
    text = re.sub(r'\n', '', text)
    # 移除标点符号
    text = re.sub(r'[- *]+', '，', text).replace("，，|：，", '，')
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)
    return text


# 4. 主评测流程
results = []
latencies = []

# 提取文件中的reference_docs列的内容列表
eval_data['reference_docs'] = eval_data['reference_docs'].apply(ast.literal_eval)

# for i in tqdm(range(len(eval_data[:2]))):
for i in tqdm(range(len(eval_data))):
    item = eval_data.iloc[i]
    print(f"当前处理第{i}条数据， 查询：{item['query']}")
    start_time = time.time()
    query = item['query']
    reference = ''.join(item['reference_docs'])
    reference_answer = item['reference_answer']
    response = launch_agent(query, agent, 1001)
    print(f"response:{response}")
    if '<think>' in response and '</think>' in response:
        answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    elif '<think>' in response and '</think>\n\n答：' in response:
        answer = re.sub(r"<think>.*?</think>\n\n答：", "", response, flags=re.DOTALL)
    else:
        answer = response
    # 执行RAG流程
    retrieved_docs = rag_qa(item['query'])

    # 记录延迟
    latency = time.time() - start_time
    latencies.append(latency)
    # 评估各模块
    retrieval_metrics = evaluate_retrieval(retrieved_docs, retrieved_docs)
    # 文本预处理过程
    reference_processed = preprocess_text(reference)
    precision_processed = preprocess_text(answer)
    generation_metrics = evaluate_generation(precision_processed, reference_processed, retrieved_docs)

    results.append({
        **retrieval_metrics,
        **generation_metrics})

# 5. 汇总结果
aggregates = {
    "Retrieval": {
        "Precision@5": np.mean([r['Precision@5'] for r in results]),
        "Recall@5": np.mean([r['Recall@5'] for r in results]),
        "MRR": np.mean([r['MRR'] for r in results])},
    "Generation": {
        "ROUGE-1": np.mean([r['ROUGE-1'] for r in results]),
        "ROUGE-L": np.mean([r['ROUGE-L'] for r in results]),
        "BERTScore": np.mean([r['BERTScore'] for r in results]),
        "Faithfulness": np.mean([r['Faithfulness'] for r in results])},
    "Performance": {
        "Avg_Latency": np.mean(latencies),
        "QPS": 1 / np.mean(latencies)
    }
}

print('=' * 50 + '\n综合评测结果：\n' + '=' * 50)
print(json.dumps(aggregates, indent=2))
