import streamlit as st
import sys
import os

# 把项目根目录加入 Python 搜索路径，这样才能找到 src 包
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.rag.chain import get_rag_chain

st.set_page_config(page_title="⚖️ 法律助手 Pro", page_icon="⚖️")
st.title("⚖️ 法律智能助手 (工程版)")

@st.cache_resource
def load_chain_cached():
    return get_rag_chain()

# 加载链条
try:
    rag_chain = load_chain_cached()
except Exception as e:
    st.error(f"加载失败，请检查数据库路径。错误信息: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入法律问题..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("AI 正在检索..."):
            stream = rag_chain.stream(prompt)
            response = st.write_stream(stream)
            
    st.session_state.messages.append({"role": "assistant", "content": response})