from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any

# NOTE: Action 类用于定义 AI Agent 生成的动作指令，采用 Pydantic 进行类型校验和规范化
class Action(BaseModel):
    tool_name: str = Field(description="The name of a tool") # 调用的工具名称
    # args 使用 Any 类型以增强对 LLM 输出不同格式（如 list/null）的兼容性
    args: Optional[Any] = Field(description="Input args of a tool, containing names and values", default={})
    
    @model_validator(mode='before')
    @classmethod
    def validate_all(cls, data: Any) -> Any:
        # HACK: 处理小规模 LLM (如 1.5b) 在生成 JSON 时可能出现的各种非标准偏移
        if not isinstance(data, dict):
            return data
            
        # 1. 规范化 tool_name (处理 LLM 误将 JSON Schema 描述作为输出的情况)
        tool_name = data.get('tool_name')
        if isinstance(tool_name, dict):
            # 兼容模式：如果 tool_name 是字典，尝试从 title 或 description 中提取具体的工具名称
            data['tool_name'] = tool_name.get('title') or tool_name.get('description') or str(tool_name)
        
        # 2. 规范化 args (处理 LLM 偶尔将 positional args 输出为 list 或 null 的情况)
        args = data.get('args')
        if isinstance(args, list):
            if len(args) > 0:
                if isinstance(args[0], dict):
                    # 如果列表内第一个元素是字典，则提取该字典为 args
                    data['args'] = args[0]
                else:
                    # 如果列表内是字符串或其他，默认将其映射为 'query' 字段（针对本项目工具的通用约定）
                    data['args'] = {"query": str(args[0])}
            else:
                # 将空列表统一转换为空字典，避免后续 Pydantic 校验失败
                data['args'] = {}
        elif args is None:
            # 确保 args 始终为字典，避免后续调用 tool.run() 时由于 NoneType 报错
            data['args'] = {}
            
        return data

    def __str__(self):
        # 自定义字符串表示，方便在 ReAct 循环的日志中直观追踪 Agent 的决策过程
        rst = f"Action(tool_name={self.tool_name}"
        if isinstance(self.args, dict):
            for k, v in self.args.items():
                rst += f", {k}={v}"
        else:
            rst += f", args={self.args}"
        rst += ")"
        return rst
