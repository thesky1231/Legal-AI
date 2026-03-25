import os
from functools import lru_cache
from typing import Any, List, Dict

import chromadb
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    RERANK_MODEL,
    RERANK_DEVICE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
    VECTOR_RECALL_K,
    BM25_RECALL_K,
    FINAL_TOP_K,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@lru_cache(maxsize=1)
def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
            "local_files_only": True,
        },
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )


class RerankRetriever(BaseRetriever):
    vector_retriever: Any
    bm25_retriever: Any
    reranker: Any
    top_k: int = FINAL_TOP_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:
        vec_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        unique_docs = {}
        for doc in vec_docs + bm25_docs:
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc

        candidates = list(unique_docs.values())
        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)

        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in scored_docs[: self.top_k]]


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库路径不存在：{DB_PATH}，请先运行 ingest.py")

    client = chromadb.PersistentClient(path=DB_PATH)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=build_embeddings(),
    )


@lru_cache(maxsize=1)
def load_all_docs_from_collection() -> List[Document]:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库路径不存在：{DB_PATH}，请先运行 ingest.py")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    raw = collection.get(include=["documents", "metadatas"])
    documents = raw.get("documents", [])
    metadatas = raw.get("metadatas", [])

    return [
        Document(page_content=doc, metadata=meta or {})
        for doc, meta in zip(documents, metadatas)
    ]


@lru_cache(maxsize=1)
def get_vector_retriever():
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": VECTOR_RECALL_K})


@lru_cache(maxsize=1)
def get_bm25_retriever():
    docs = load_all_docs_from_collection()
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = BM25_RECALL_K
    return bm25_retriever


def get_hybrid_candidates(query: str) -> List[Document]:
    vector_retriever = get_vector_retriever()
    bm25_retriever = get_bm25_retriever()

    vec_docs = vector_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    unique_docs = {}
    for doc in vec_docs + bm25_docs:
        if doc.page_content not in unique_docs:
            unique_docs[doc.page_content] = doc
    return list(unique_docs.values())


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)


def get_retriever():
    vector_retriever = get_vector_retriever()
    bm25_retriever = get_bm25_retriever()
    reranker = get_reranker()

    return RerankRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        top_k=FINAL_TOP_K,
    )


def retrieve(query: str, top_k: int = FINAL_TOP_K) -> List[Document]:
    retriever = get_retriever()
    docs = retriever.invoke(query)
    return docs[:top_k]


def format_docs(docs: List[Document]) -> str:
    parts = []
    for idx, doc in enumerate(docs, start=1):
        article = doc.metadata.get("article", "未知法条")
        source = doc.metadata.get("source", "未知来源")
        parts.append(
            f"【资料{idx}】\n"
            f"法条: {article}\n"
            f"来源: {source}\n"
            f"内容: {doc.page_content}"
        )
    return "\n\n".join(parts)


def build_answer_prompt() -> ChatPromptTemplate:
    template = """
你是一个法律检索问答助手。请严格依据提供的法条资料回答，不要使用资料之外的知识。

回答要求：
1. 先给出简明结论；
2. 再给出对应法条依据；
3. 如果证据不足，请明确回答“根据当前检索到的法条，暂时无法确定”；
4. 不要编造法条，不要虚构案件事实；
5. 输出尽量结构化，格式为：
【结论】
...
【依据】
...
【说明】
...

【参考资料】
{context}

【用户问题】
{question}
"""
    return ChatPromptTemplate.from_template(template)

from config import (
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    RERANK_MODEL,
    RERANK_DEVICE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
    VECTOR_RECALL_K,
    BM25_RECALL_K,
    FINAL_TOP_K,
    OLLAMA_BASE_URL,
)

def get_model() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )

def is_low_confidence(query: str, docs: List[Document]) -> bool:
    if not docs:
        return True

    query_len = len(query.strip())
    joined = "".join(doc.page_content for doc in docs[:2])

    # 很粗但实用：如果召回内容过短，或者问题很长但证据很少，先走保守策略
    if len(joined.strip()) < 60:
        return True

    if query_len >= 20 and len(joined.strip()) < 120:
        return True

    return False


def answer_with_sources(query: str, top_k: int = FINAL_TOP_K) -> Dict:
    docs = retrieve(query, top_k=top_k)

    sources = [
        {
            "article": doc.metadata.get("article", "未知法条"),
            "source": doc.metadata.get("source", "未知来源"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    if is_low_confidence(query, docs):
        return {
            "answer": (
                "【结论】\n根据当前检索到的法条，暂时无法确定。\n\n"
                "【依据】\n当前证据不足，建议补充更具体的案件事实或问题描述。\n\n"
                "【说明】\n系统仅基于已检索法条作答，为避免误导，当前不做超出证据范围的判断。"
            ),
            "sources": sources,
            "confidence": "low",
        }

    prompt = build_answer_prompt()
    model = get_model()

    chain = (
        {"context": lambda _: format_docs(docs), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    answer = chain.invoke(query)
    return {
        "answer": answer,
        "sources": sources,
        "confidence": "medium",
    }


def get_rag_chain():
    retriever = get_retriever()
    prompt = build_answer_prompt()
    model = get_model()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain