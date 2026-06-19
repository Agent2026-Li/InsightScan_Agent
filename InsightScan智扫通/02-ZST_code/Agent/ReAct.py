import os
import re
from typing import List, Optional, Tuple

import pandas as pd
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputToolsParser, StrOutputParser, PydanticOutputParser
from langchain_core.tools import BaseTool, render_text_description  # 👈 核心修复：BaseTool 移到了这里
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from pydantic import ValidationError

from Agent.Action import Action


class ReActAgent:
    # JSON提取：从RST文本中提取JSON代码块内容
    @staticmethod
    def extract_json_from_action_rst(text: str) -> str | None:
        """
        从RST格式文本中提取JSON代码块内容
        """
        pattern = re.compile(r'```json(.*?)```', flags=re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            json_str = matches[-1]
            return json_str
        return None

    # 观察格式化：清理响应并追加动作和观察结果
    @staticmethod
    def format_observations(response: str, action: Action, observation: str) -> str:
        """
        格式化观察结果，去除响应中的JSON代码块，并将动作和观察结果追加到响应末尾
        """
        rst = re.sub(r"```json(.*?)```", "", response, flags=re.DOTALL)
        rst += "\n" + str(action) + "\n返回结果为：\n" + observation
        return rst

    # 初始化流程
    def __init__(self, llm: BaseChatModel, tools: List[BaseTool], main_prompt_file_path: str):
        """
        初始化智能体构造器
        """
        self.llm = llm
        self.tools = tools
        self.main_prompt_file_path = main_prompt_file_path

        # 实例化基础输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=Action)

        # 初始化提示模版和构建处理链
        self.__init_prompts()
        self.__init_chains()

    # 初始化提示词模版
    def __init_prompts(self):
        """
        初始化提示模板
        """
        with open(os.path.join(os.getcwd(), self.main_prompt_file_path), 'r', encoding='utf-8') as f:
            self.prompt = ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),  # 聊天历史占位符
                HumanMessagePromptTemplate.from_template(f.read()),
            ]).partial(
                tools=render_text_description(self.tools),
                tool_names=",".join([tool.name for tool in self.tools]),
                format_instructions=self.output_parser.get_format_instructions()
            )

    # 初始化任务链主对象
    def __init_chains(self):
        """
        初始化主链对象
        """
        self.main_chains = (self.prompt | self.llm | StrOutputParser())

    # 工具管理，确认工具是否存在
    def confirm_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        根据工具名称查找并返回对应的工具对象
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    # 单步执行，生成动作并解析响应
    def step_by_step(self, task, short_term_memory=None, chat_history=None, verbose=False) -> Tuple[Action, str]:
        """
        执行逐步任务处理
        """
        inputs = {
            "input": task,
            "agent_scratchpad": "\n".join(short_term_memory) if short_term_memory else "",
            "chat_history": chat_history.messages if chat_history else [],
        }
        response = ""
        for token in self.main_chains.stream(inputs):
            response += token
        print(f'response:{response}')

        action_rst_json = self.extract_json_from_action_rst(response)
        print(f'json_action:{action_rst_json}')

        # 鲁棒解析逻辑：
        try:
            target_text = action_rst_json if action_rst_json else response
            action = self.output_parser.parse(target_text)
        except Exception as e:
            # 完美的 ReAct 降级策略：解析失败时赋予“未知”动作，反馈给模型进行自我修复
            print(f"解析动作失败: {e}. 将错误信息反馈给模型进行自我反思。")
            action = Action(tool_name="未知", args={"query": response})

        print(f'action:{action}')
        return action, response

    # 执行动作，验证并调用工具返回观察结果
    def exec_actions(self, action: Action) -> str:
        """
        执行指定的动作并返回观察结果
        """
        tool = self.confirm_tool(action.tool_name)

        if tool is None:
            observation = (
                f"Error: 没找到待执行的工具或者指令 '{action.tool_name}'。 "
                f"或者是你的 JSON 格式不符合规范，请严格按照要求的 json 格式重新输出。"
            )
        else:
            try:
                print(f'tool 执行参数：{action.args}')
                observation = tool.run(action.args)
                print(f'tool 执行结果：{observation}')
            except ValidationError as e:
                observation = f"执行动作时参数非法：{str(e)}, args:{action.args}"
            except Exception as e:
                observation = f"Error in exec_actions: {str(e)}, args:{action.args}"
        return observation

    # 任务执行：循环执行 -> 推理 -> 执行动作 -> 观察直到完成
    def execute(self, task: str, uuid: int, chat_history: ChatMessageHistory, verbose=False) -> str:
        """
        执行任务的主要方法
        """
        short_term_memory = []

        action, response = self.step_by_step(task=task, short_term_memory=short_term_memory,
                                             chat_history=chat_history, verbose=verbose)
        print(f'execute方法: action:{action}, response:{response}')

        answer = self.exec_actions(action)
        print(f'answer:{answer}')

        short_term_memory.append(self.format_observations(response, action, answer))

        if chat_history is not None:
            chat_history.add_user_message(task)
            chat_history.add_ai_message(answer)

        return answer