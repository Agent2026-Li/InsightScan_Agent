import time

from Agent.ReAct import ReActAgent
from Models.Factory import ChatModelFactory
from Tools.Tools import *
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from flask import Flask, request, jsonify
import re

app = Flask(__name__)


def launch_agent(query, agent, uuid=1001):
    chat_history = ChatMessageHistory()
    reply = agent.execute(query, uuid, chat_history=chat_history, verbose=True)
    return reply

# 添加根路径响应
@app.route('/')
def home():
    return "<h1>扫智通Agent服务已启动</h1><p>请通过指定API接口进行访问，如：/chat</p>", 200

@app.route("/predict", methods=["POST"])
def main():
    try:
        data = request.get_json()
        query = data.get("query", "")
        uuid = data.get('uuid', 1001)
        if not query:
            return jsonify({"error": "请输入文本"}), 400
        llm = ChatModelFactory().get_model()

        tools = [
            rag_qa_tool,
            report_generation_tool,
        ]

        agent = ReActAgent(
            llm=llm,
            tools=tools,
            main_prompt_file_path="prompts/main.txt"
        )

        response = launch_agent(query, agent, uuid)

        if '<think>' in response and "</think>" in response:
            answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        elif '<think>' in response and "</think>\n\n答：" in response:
            answer = re.sub(r"<think>.*?</think>\n\n答：", "", response, flags=re.DOTALL)
        else:
            answer = response
        print(f'answer:{answer}')
        print('-----------------------------------------------------------------------')
        return jsonify({"answer": answer}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
