import streamlit as st
import requests

# 1. 页面配置（小彩蛋：设置网页标签和图标）
st.set_page_config(
    page_title="InsightScan 智扫通 | 扫地机器人 Agent 智能看版",
    page_icon="🤖",
    layout="wide"  # 宽屏布局，更具现代感
)

# 2. 自定义 CSS 样式（让 UI 细节更精致）
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .badge {
        background-color: #E2E8F0;
        color: #475569;
        padding: 0.25rem 0.6rem;
        border-radius: 0.375rem;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 顶部区域布局
st.markdown('<p class="main-title">🤖 InsightScan 智扫通</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">基于 LangChain ReAct 框架的扫地机器人智能问答与报告生成 Agent 系统</p>',
            unsafe_allow_html=True)

# 核心技术标签
st.markdown("""
    <span class="badge">LangChain</span>
    <span class="badge">ReAct Agent</span>
    <span class="badge">Qwen3</span>
    <span class="badge">ChromaDB RAG</span>
    <span class="badge">Flask API</span>
""", unsafe_allow_html=True)
st.write("")

# 4. 左右分栏布局（左侧操作，右侧展示结果，极具生产线看板风范）
col_left, col_right = st.columns([5, 5], gap="large")

with col_left:
    st.subheader("📥 交互输入控制台")

    # 将输入框用容器包裹，视觉上更有整体感
    with st.container(border=True):
        sentence = st.text_area(
            "请输入您的指令或设备疑问：",
            placeholder="例如：\n1. 扫地机器人显示雷达传感器报错怎么办？\n2. 帮我生成用户ID 1001在5月份的使用分析报告。",
            height=130
        )

        # 按钮样式微调（Streamlit 默认会拉伸）
        generate_btn = st.button("🚀 调度 Agent 执行", type="primary", use_container_width=True)

    # 说明书挪到左下方
    with st.expander("📖 查看系统架构与操作指南", expanded=True):
        st.markdown("""
        1. **设备知识库问答 (RAG)**：输入关于设备报错、清扫说明、硬件维护的疑问，Agent 将自动检索 ChromaDB 知识库并作答。
        2. **端到端报告生成 (Data Agent)**：输入包含 `用户ID` 和 `月份` 的指令，系统将触发参数提取并联动底层 Pandas 业务数据渲染生成深度分析。
        """)

with col_right:
    st.subheader("🖥️ Agent 实时执行结果")

    if generate_btn:
        if not sentence.strip():
            st.warning("⚠️ 请输入有效的句子再点击执行！")
        else:
            # 增加平滑的等待动效
            with st.spinner("🤖 Agent 正在思考、规划并调用工具链..."):
                try:
                    # 调用 Flask API
                    response = requests.post(
                        "http://192.168.71.105:8002/predict",
                        json={"query": sentence, "uuid": 1001},
                        timeout=15  # 增加超时保护
                    )
                    response.raise_for_status()
                    result = response.json()

                    if 'answer' in result:
                        # 使用大模型标准的 Chat Message 气泡组件进行高大上的结果渲染
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown("### ✨ 最终交付结果")
                            st.write(result['answer'])
                            st.balloons()  # 成功后放个小气球特效，演示极佳
                    else:
                        st.error(f"❌ 系统内部错误：{result.get('error', '未知错误')}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ 连到后端的请求超时了，请检查后端大模型服务是否卡死。")
                except requests.exceptions.RequestException as e:
                    st.error(
                        f'🚨 无法连接到后端的 Flask API，请确保服务器 `http://192.168.71.105:8002` 处于运行状态。\n错误详情: {str(e)}')
    else:
        # 未点击生成时的占位提示
        with st.container(border=True):
            st.info("💡 请在左侧控制台输入指令并点击“调度 Agent 执行”，结果会在此实时渲染输出。")