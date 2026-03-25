import os
import re
from functools import lru_cache
from typing import Any, List, Dict, Literal

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
    OLLAMA_BASE_URL,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

QuestionType = Literal[
    "direct_answer",
    "definition",
    "confusing",
    "complex_reasoning",
    "should_refuse",
]


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
    if not docs:
        raise ValueError("当前数据库中没有可用于 BM25 的文档，请先执行 ingest 构建知识库。")

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


# =========================
# 1. 问题分类
# =========================
def classify_question(query: str) -> QuestionType:
    q = query.strip()

    # 1. should_refuse：不应直接下案件定性或量刑结论
    refuse_patterns = [
        r"能不能直接判断",
        r"能不能直接认定",
        r"一定会判几年",
        r"一定构成",
        r"一定就是",
        r"能不能断定",
        r"能不能直接下结论",
        r"能不能仅凭",
        r"只看.*能不能",
        r"是不是一定",
        r"一定会怎么判",
        r"能不能只根据.*判断",
    ]
    if any(re.search(p, q) for p in refuse_patterns):
        return "should_refuse"

    # 2. confusing：易混淆、比较、区别
    confusing_keywords = [
        "区别",
        "不同",
        "区分",
        "边界",
        "混淆",
        "相比",
        "比较",
    ]
    confusing_pairs = [
        ("抢劫", "抢夺"),
        ("诈骗", "合同诈骗"),
        ("故意伤害", "故意杀人"),
        ("非法拘禁", "绑架"),
        ("危险驾驶", "交通肇事"),
        ("盗窃", "故意毁坏财物"),
    ]
    if any(k in q for k in confusing_keywords):
        for a, b in confusing_pairs:
            if a in q and b in q:
                return "confusing"

    # “A和B有什么区别”这类也算 confusing
    for a, b in confusing_pairs:
        if a in q and b in q:
            return "confusing"

    # 3. definition：概念定义、术语解释
    definition_keywords = [
        "是什么",
        "怎么理解",
        "叫什么",
        "定义",
        "属于什么",
        "在法律上叫什么",
        "一般指什么",
        "如何理解",
        "是什么意思",
    ]
    if any(k in q for k in definition_keywords):
        return "definition"
    if "从犯" in q and ("怎么处理" in q or "一般怎么处理" in q):
        return "definition"
    # 4. complex_reasoning：多条件、多角色、多情节
    complex_keywords = [
        "未成年人",
        "主犯",
        "从犯",
        "共同犯罪",
        "正当防卫",
        "防卫过当",
        "未遂",
        "自首",
        "量刑",
        "怎么处理",
        "如何处理",
        "如何认定",
        "如何评价",
        "结合",
    ]
    if sum(1 for k in complex_keywords if k in q) >= 2:
        return "complex_reasoning"

    # 默认 direct_answer
    return "direct_answer"

# =========================
# 2. 类型对应策略
# =========================
def get_top_k_for_question_type(qtype: QuestionType) -> int:
    mapping = {
        "direct_answer": 3,
        "definition": 4,
        "confusing": 7,
        "complex_reasoning": 5,
        "should_refuse": 3,
    }
    return mapping[qtype]


def build_refusal_answer() -> str:
    return (
        "【结论】\n"
        "根据当前检索到的法条，暂时无法直接作出确定性结论。\n\n"
        "【依据】\n"
        "此类问题通常需要结合具体案件事实、行为方式、主观故意、结果后果以及证据情况综合分析，"
        "仅凭当前问题描述或少量法条，不能直接下最终定性或量刑结论。\n\n"
        "【说明】\n"
        "为避免误导，系统在证据不足或案件事实不完整时采取保守回答策略。建议补充更具体的案件事实。"
    )


def build_direct_prompt() -> ChatPromptTemplate:
    template = """
你是一个法律检索问答助手。请严格依据提供的法条资料回答，不要使用资料之外的知识。

回答要求：
1. 先给出简明结论；
2. 再给出对应法条依据；
3. 不要编造法条，不要虚构案件事实；
4. 输出尽量结构化，格式为：
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


def build_confusing_prompt() -> ChatPromptTemplate:
    template = """
你是一个法律检索问答助手。当前问题属于“易混淆法律概念/罪名区分”问题，请严格依据提供的法条资料回答。

回答要求：
1. 不要武断地下最终案件结论；
2. 必须分别说明两个概念/罪名各自的核心特征；
3. 必须分别列出对应的法条依据，不能只给一个；
4. 必须明确指出二者最关键的区分点；
5. 如资料不足以完整比较，明确说明“当前资料不足以完整区分”；
6. 输出格式为：
【比较对象】
A：...
B：...
【核心区别】
...
【A 的依据法条】
...
【B 的依据法条】
...
【说明】
...

【参考资料】
{context}

【用户问题】
{question}
"""
    return ChatPromptTemplate.from_template(template)


def build_complex_prompt() -> ChatPromptTemplate:
    template = """
你是一个法律检索问答助手。当前问题属于“复杂推理/多条件分析”问题，请严格依据提供的法条资料回答。

回答要求：
1. 优先给出保守、稳健的分析；
2. 明确说明还需要结合哪些案件事实；
3. 不要作出绝对化结论；
4. 必须列出法条依据；
5. 输出格式为：
【初步结论】
...
【需要结合的事实】
...
【依据法条】
...
【说明】
...

【参考资料】
{context}

【用户问题】
{question}
"""
    return ChatPromptTemplate.from_template(template)

def build_definition_prompt() -> ChatPromptTemplate:
    template = """
你是一个法律检索问答助手。当前问题属于“法律概念/术语定义”问题，请严格依据提供的法条资料回答。

回答要求：
1. 先直接解释这个概念是什么；
2. 再给出对应法条依据；
3. 不要过度保守，不要把普通定义题误判成无法回答；
4. 不要编造法条，不要虚构案件事实；
5. 输出格式为：
【概念解释】
...
【依据法条】
...
【说明】
...

【参考资料】
{context}

【用户问题】
{question}
"""
    return ChatPromptTemplate.from_template(template)

def get_prompt_by_type(qtype: QuestionType) -> ChatPromptTemplate:
    if qtype == "definition":
        return build_definition_prompt()
    if qtype == "confusing":
        return build_confusing_prompt()
    if qtype == "complex_reasoning":
        return build_complex_prompt()
    return build_direct_prompt()


def get_model() -> ChatOllama:
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
    )


# =========================
# 3. 风险判断
# =========================
def is_low_confidence(query: str, docs: List[Document], qtype: QuestionType) -> bool:
    if not docs:
        return True

    joined = "".join(doc.page_content for doc in docs[:2]).strip()

    if len(joined) < 60:
        return True

    # 定义题不要过度保守
    if qtype == "definition":
        return len(joined) < 80

    # 复杂题要求更高证据量
    if qtype == "complex_reasoning" and len(joined) < 120:
        return True

    # 混淆题要求候选稍充分
    if qtype == "confusing" and len(joined) < 100:
        return True

    return False


# =========================
# 4. 路由后的回答主函数
# =========================
def answer_with_sources(query: str, top_k: int | None = None) -> Dict:
    qtype = classify_question(query)
    chosen_top_k = top_k if top_k is not None else get_top_k_for_question_type(qtype)

    docs = retrieve(query, top_k=chosen_top_k)

    sources = [
        {
            "article": doc.metadata.get("article", "未知法条"),
            "source": doc.metadata.get("source", "未知来源"),
            "content": doc.page_content,
        }
        for doc in docs
    ]

    # should_refuse 直接走保守策略优先
    if qtype == "should_refuse":
        return {
            "answer": build_refusal_answer(),
            "sources": sources,
            "confidence": "low",
            "question_type": qtype,
        }

    # 复杂问题、证据不足时也保守
    if is_low_confidence(query, docs, qtype):
        return {
            "answer": (
                "【结论】\n根据当前检索到的法条，暂时无法确定。\n\n"
                "【依据】\n当前证据不足，建议补充更具体的案件事实或问题描述。\n\n"
                "【说明】\n系统仅基于已检索法条作答，为避免误导，当前不做超出证据范围的判断。"
            ),
            "sources": sources,
            "confidence": "low",
            "question_type": qtype,
        }

    prompt = get_prompt_by_type(qtype)
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
        "confidence": "medium" if qtype != "complex_reasoning" else "low",
        "question_type": qtype,
    }


def get_rag_chain():
    prompt = build_direct_prompt()
    model = get_model()
    retriever = get_retriever()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return rag_chain