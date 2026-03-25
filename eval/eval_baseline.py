import csv
import json
import os
import re
from datetime import datetime
from typing import List, Dict, Callable, Tuple

from tqdm import tqdm
from langchain_core.documents import Document

from src.rag.chain import (
    get_bm25_retriever,
    get_vectorstore,
    get_hybrid_candidates,
    get_reranker,
)


DEFAULT_DATASET_PATH = os.path.join("data", "golden_dataset.json")
RESULTS_DIR = os.path.join("eval", "results")


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


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


# 整个评测过程中只初始化一次
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


def doc_titles(docs: List[Document]) -> List[str]:
    return [doc.metadata.get("article", "未知法条") for doc in docs]


def evaluate_strategy(
    dataset: List[Dict],
    strategy_name: str,
    search_fn: Callable[[str, int], List[Document]],
) -> Tuple[Dict, List[Dict]]:
    metrics = {}
    error_cases = []

    for k in [3, 5]:
        correct = 0
        current_k_errors = []

        for item in tqdm(dataset, desc=f"{strategy_name} Recall@{k}"):
            question = item["question"]
            ground_truth = item["ground_truth"]
            qtype = item.get("type", "unknown")
            note = item.get("note", "")

            docs = search_fn(question, k)
            hit = contains_ground_truth(docs, ground_truth)

            if hit:
                correct += 1
            else:
                current_k_errors.append(
                    {
                        "strategy": strategy_name,
                        "k": k,
                        "question": question,
                        "type": qtype,
                        "note": note,
                        "ground_truth": ground_truth,
                        "retrieved_articles": doc_titles(docs),
                        "retrieved_contents_preview": [
                            doc.page_content[:120] for doc in docs
                        ],
                    }
                )

        recall = correct / len(dataset) if dataset else 0.0
        metrics[f"Recall@{k}"] = recall

        # 只把更严格的 Recall@3 全量保留；Recall@5 也保留，方便分析
        error_cases.extend(current_k_errors)

    return metrics, error_cases


def save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, summary: Dict[str, Dict[str, float]]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "recall@3", "recall@5"])
        for strategy, metrics in summary.items():
            writer.writerow(
                [
                    strategy,
                    f"{metrics.get('Recall@3', 0.0):.4f}",
                    f"{metrics.get('Recall@5', 0.0):.4f}",
                ]
            )


def main():
    dataset_path = os.getenv("EVAL_DATASET_PATH", DEFAULT_DATASET_PATH)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到测试集：{dataset_path}")

    ensure_results_dir()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    strategies = {
        "BM25": bm25_search,
        "Vector": vector_search,
        "Hybrid": hybrid_search,
        "Hybrid + Rerank": hybrid_rerank_search,
    }

    summary = {}
    all_errors = []

    for name, fn in strategies.items():
        metrics, errors = evaluate_strategy(dataset, name, fn)
        summary[name] = metrics
        all_errors.extend(errors)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    result_json_path = os.path.join(
        RESULTS_DIR, f"{dataset_name}_recall_results_{timestamp}.json"
    )
    result_csv_path = os.path.join(
        RESULTS_DIR, f"{dataset_name}_recall_results_{timestamp}.csv"
    )
    error_json_path = os.path.join(
        RESULTS_DIR, f"{dataset_name}_error_analysis_{timestamp}.json"
    )

    result_payload = {
        "dataset_path": dataset_path,
        "dataset_size": len(dataset),
        "generated_at": timestamp,
        "summary": summary,
    }

    error_payload = {
        "dataset_path": dataset_path,
        "dataset_size": len(dataset),
        "generated_at": timestamp,
        "errors": all_errors,
    }

    save_json(result_json_path, result_payload)
    save_csv(result_csv_path, summary)
    save_json(error_json_path, error_payload)

    print("\n" + "=" * 60)
    print("📊 检索评测结果")
    print("=" * 60)
    print(f"测试集: {dataset_path}")
    print(f"样本数: {len(dataset)}")
    print("-" * 60)
    for name, metrics in summary.items():
        print(
            f"{name:<18} "
            f"Recall@3={metrics['Recall@3']:.2%}  "
            f"Recall@5={metrics['Recall@5']:.2%}"
        )
    print("-" * 60)
    print(f"结果 JSON: {result_json_path}")
    print(f"结果 CSV : {result_csv_path}")
    print(f"错题分析 : {error_json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()