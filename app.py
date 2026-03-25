import streamlit as st
from chain import get_rag_chain

st.set_page_config(page_title="⚖️ 法律助手 Pro", page_icon="⚖️")
st.title("⚖️ 法律智能助手（工程版）")


@st.cache_resource
def load_chain_cached():
    return get_rag_chain()


try:
    rag_chain = load_chain_cached()
except Exception as e:
    import traceback

    st.error("❌ 系统初始化失败")
    st.code(traceback.format_exc())
    st.stop()
import streamlit as st
from src.rag.chain import answer_with_sources

st.set_page_config(page_title="⚖️ 法律助手 Pro", page_icon="⚖️")
st.title("⚖️ 法律智能助手（工程版）")
st.caption("支持法条依据展示、低置信度保守回答。")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("查看法条依据"):
                for idx, source in enumerate(message["sources"], start=1):
                    st.markdown(f"**资料 {idx}｜{source['article']}**")
                    st.caption(f"来源：{source['source']}")
                    st.write(source["content"])

if prompt := st.chat_input("请输入法律问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在检索并生成答案..."):
            result = answer_with_sources(prompt)
            st.markdown(result["answer"])

            badge = "🟡 中等置信度" if result["confidence"] == "medium" else "🔴 低置信度"
            st.caption(f"问题类型：{result['question_type']} ｜ 当前回答状态：{badge}")

            with st.expander("查看法条依据"):
                for idx, source in enumerate(result["sources"], start=1):
                    st.markdown(f"**资料 {idx}｜{source['article']}**")
                    st.caption(f"来源：{source['source']}")
                    st.write(source["content"])

    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "question_type": result["question_type"],
    }
)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入法律问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI 正在检索并生成答案..."):
            response = st.write_stream(rag_chain.stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})