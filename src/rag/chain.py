import os
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(CURRENT_DIR)), "db")

def get_retriever():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库路径不存在: {DB_PATH}，请先运行 create_db.py")
    
    
    client = chromadb.PersistentClient(path=DB_PATH)
    vectorstore = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="law_data"
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


def get_rag_chain():
    retriever = get_retriever()

    template = """
    你是一个法律助手。请根据以下参考资料回答问题：
    【参考资料】：
    {context}
    【问题】：
    {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOllama(model="qwen2.5:7b", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain
