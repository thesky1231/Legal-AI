from langchain_openai import ChatOpenAI

print("正在连接本地 vLLM...")

# 关键点来了！👇
llm = ChatOpenAI(
    model="qwen2.5",               # 这个名字要跟你启动命令里的 --served-model-name 一致
    openai_api_key="EMPTY",        # 本地跑不需要 key，但库要求必须填一个，随便填
    openai_api_base="http://localhost:8000/v1", # 【重点】指向你的 WSL 地址
    temperature=0.7
)

# 见证奇迹的时刻：用法跟以前一模一样！
try:
    response = llm.invoke("你好，你是谁？请做个自我介绍。")
    print("\n====== 调用成功！ ======")
    print(response.content)
except Exception as e:
    print("\n====== 连接失败 ======")
    print("请检查 WSL 里的 vLLM 服务是不是还活着（有没有被关掉）")
    print(e)