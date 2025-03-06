#   conda activate vs
#   streamlit run 1.py

import streamlit as st  
from openai import OpenAI
from langchain.memory import ConversationBufferMemory
import time
import os

# 必须放在最前面设置页面配置
st.set_page_config(page_title="Deepseek GPT Chat", layout="wide")

# 弹窗提示：在第一次打开页面时显示
if "pop_up_closed" not in st.session_state:
    st.session_state.pop_up_closed = False

if not st.session_state.pop_up_closed:
    with st.container():
        st.info("欢迎使用 Deepseek Chat！\n\n请注意：本系统仅用于演示，数据可能存在延迟或不准确，请谨慎使用。")
        if st.button("关闭提示"):
            st.session_state.pop_up_closed = True
    st.markdown("---")  # 分割线

# 子类化 ConversationBufferMemory，覆盖 save_context 方法，允许 outputs 缺省
class DeepseekConversationBufferMemory(ConversationBufferMemory):
    def save_context(self, inputs, outputs=None):
        if outputs is None:
            outputs = {}  
        return super().save_context(inputs, outputs)

# 配置 deepseek 模型（请确保 api_key 和 base_url 设置正确）
client = OpenAI(
    api_key=os.environ['KEY'],
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    timeout=1800
)

memory = DeepseekConversationBufferMemory()

st.title("聊天界面")

# 会话状态初始化：支持多个会话
if "sessions" not in st.session_state:
    st.session_state.sessions = {}  # 格式：{会话名: [{"role": ..., "content": ...}, ...]}
if "current_session" not in st.session_state:
    st.session_state.current_session = None

# 侧边栏：会话管理
st.sidebar.header("会话管理")

# 新建会话按钮
if st.sidebar.button("新建会话"):
    new_session_name = f"会话_{len(st.session_state.sessions) + 1}"
    st.session_state.sessions[new_session_name] = []
    st.session_state.current_session = new_session_name

# 如果已有会话，则提供选择下拉框
session_names = list(st.session_state.sessions.keys())
if session_names:
    default_index = session_names.index(st.session_state.current_session) if st.session_state.current_session in session_names else 0
    session_choice = st.sidebar.selectbox("选择会话", session_names, index=default_index)
    st.session_state.current_session = session_choice

    # 侧边栏显示当前会话历史
    def update_sidebar_display():
        session_chat = st.session_state.sessions.get(st.session_state.current_session, [])
        chat_md = ""
        for msg in session_chat:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["assistant", "assistant_temp"]:
                chat_md += f"**Assistant**：{content}\n\n---\n\n"
            else:
                chat_md += f"**User**：{content}\n\n"
        st.session_state.chat_md = chat_md
    update_sidebar_display()
    st.sidebar.markdown("**当前会话历史**：")
    st.sidebar.markdown(st.session_state.chat_md)
else:
    st.sidebar.write("暂无会话，请点击左侧按钮新建会话。")

# 如果还未创建会话，则不显示输入区域及聊天记录
if st.session_state.current_session is None:
    st.info("请点击左侧【新建会话】按钮后开始聊天。")
else:
    # 主区域：聊天显示区域
    chat_placeholder = st.empty()

    def update_chat_display_main():
        """更新当前会话的聊天记录显示"""
        session_chat = st.session_state.sessions.get(st.session_state.current_session, [])
        chat_md = ""
        for msg in session_chat:
            role = msg.get("role")
            content = msg.get("content")
            if role in ["assistant", "assistant_temp"]:
                chat_md += f"**Assistant**：{content}\n\n---\n\n"
            else:
                chat_md += f"**User**：{content}\n\n"
        chat_placeholder.markdown(chat_md)

    update_chat_display_main()

    # 使用 st.form 包裹输入框和发送按钮，开启 clear_on_submit 自动清空输入框
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("请输入您的消息：", key="user_input")
        submitted = st.form_submit_button("发送")
        
        if submitted and user_input.strip():
            # 保存用户输入
            st.session_state.sessions[st.session_state.current_session].append({"role": "user", "content": user_input})
            update_chat_display_main()
            
            display_text = ""
            try:
                # 调用 deepseek 流式接口，使用 messages 参数（根据实际 API 要求可调整）
                stream = client.chat.completions.create(
                    model="deepseek-r1-250120",
                    messages=[{"role": "user", "content": user_input}],
                    stream=True,
                    temperature=0.7
                )
            except Exception as e:
                st.error(f"调用接口出错：{e}")
            else:
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    display_text += token
                    current_chat = st.session_state.sessions[st.session_state.current_session]
                    if current_chat and current_chat[-1].get("role") == "assistant_temp":
                        current_chat[-1]["content"] = display_text
                    else:
                        current_chat.append({"role": "assistant_temp", "content": display_text})
                    update_chat_display_main()
                    time.sleep(0.01)  # 根据需要调整刷新频率

                current_chat = st.session_state.sessions[st.session_state.current_session]
                if current_chat and current_chat[-1].get("role") == "assistant_temp":
                    current_chat[-1]["role"] = "assistant"
                update_chat_display_main()
                memory.save_context({"input": user_input}, {"output": display_text})
