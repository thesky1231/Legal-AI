import json
import os
import re
from typing import List, Dict, Callable

from tqdm import tqdm
from langchain_core.documents import Document

from src.rag.chain import (
    get_bm25_retriever,
    get_vectorstore,
    get_hybrid_candidates,
    get_reranker,
)
from config import GOLDEN_DATASET_PATH


def normalize(text: str) -> str:
    text = re.sub(r"[，。！？【】（）()、；：“”‘’《》〈〉,.!?;:\"'\[\]\s]+", "", text)
    return text.strip()


def contains_ground_truth(retrieved_docs: List[Document], ground_truth: str) -> bool:
    context_text = "".join(doc.page_content for doc in retrieved_docs)
    return normalize(ground_truth) in normalize(context_text)


def rerank_docs(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored = list(zip(docs, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:top_k]]


# 这些对象在整个评测过程中只初始化一次
VECTORSTORE = get_vectorstore()
BM25_RETRIEVER = get_bm25_retriever()


def vector_search(query: str, k: int) -> List[Document]:
    return VECTORSTORE.similarity_search(query, k=k)


def bm25_search(query: str, k: int) -> List[Document]:
    BM25_RETRIEVER.k = k
    return BM25_RETRIEVER.invoke(query)


def hybrid_search(query: str, k: int) -> List[Document]:
    candidates = get_hybrid_candidates(query)
    return candidates[:k]


def hybrid_rerank_search(query: str, k: int) -> List[Document]:
    candidates = get_hybrid_candidates(query)
    return rerank_docs(query, candidates, k)


def evaluate_strategy(
    dataset: List[Dict],
    strategy_name: str,
    search_fn: Callable[[str, int], List[Document]],
):
    results = {}

    for k in [3, 5]:
        correct = 0
        for item in tqdm(dataset, desc=f"{strategy_name} Recall@{k}"):
            question = item["question"]
            ground_truth = item["ground_truth"]

            docs = search_fn(question, k)
            if contains_ground_truth(docs, ground_truth):
                correct += 1

        recall = correct / len(dataset)
        results[f"Recall@{k}"] = recall

    return results


def main():
    if not os.path.exists(GOLDEN_DATASET_PATH):
        raise FileNotFoundError(f"找不到测试集：{GOLDEN_DATASET_PATH}")

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    strategies = {
        "BM25": bm25_search,
        "Vector": vector_search,
        "Hybrid": hybrid_search,
        "Hybrid + Rerank": hybrid_rerank_search,
    }

    summary = {}
    for name, fn in strategies.items():
        summary[name] = evaluate_strategy(dataset, name, fn)

    print("\n" + "=" * 50)
    print("📊 检索评测结果")
    print("=" * 50)
    for name, metrics in summary.items():
        print(
            f"{name:<18} "
            f"Recall@3={metrics['Recall@3']:.2%}  "
            f"Recall@5={metrics['Recall@5']:.2%}"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()