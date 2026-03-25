import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Any

from src.rag.chain import answer_with_sources


DEFAULT_DATASET_PATH = os.path.join("data", "answer_eval_dataset.json")
RESULTS_DIR = os.path.join("eval", "results")


def ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def normalize_article_name(text: str) -> str:
    return text.replace(" ", "").replace("　", "").strip()


def contains_expected_article(returned_sources: List[Dict[str, Any]], expected_articles: List[str]) -> bool:
    if not expected_articles:
        return True

    returned_titles = [
        normalize_article_name(item.get("article", ""))
        for item in returned_sources
    ]
    expected_titles = [normalize_article_name(x) for x in expected_articles]

    for exp in expected_titles:
        if any(exp in title or title in exp for title in returned_titles):
            return True
    return False


def looks_like_refusal(answer: str) -> bool:
    refusal_keywords = [
        "暂时无法确定",
        "证据不足",
        "需要结合具体案件事实",
        "不能直接判断",
        "不宜直接下结论",
        "需要进一步分析",
        "无法直接得出",
    ]
    return any(keyword in answer for keyword in refusal_keywords)


def judge_answer_correct(answer: str, reference_answer: str, should_refuse: bool) -> int:
    """
    第一版用启发式，不追求完美，先追求可运行、可对比。
    """
    if should_refuse:
        return 1 if looks_like_refusal(answer) else 0

    # 非拒答类：只要不是明显空回答/明显拒答，就先记为正确
    if not answer or len(answer.strip()) < 10:
        return 0

    if looks_like_refusal(answer):
        return 0

    return 1


def judge_citation_correct(returned_sources: List[Dict[str, Any]], expected_articles: List[str], should_refuse: bool) -> int:
    if should_refuse:
        # 拒答题不强制要求法条命中
        return 1
    return 1 if contains_expected_article(returned_sources, expected_articles) else 0


def judge_hallucination(answer: str, should_refuse: bool) -> int:
    """
    第一版简化：该拒答却没有拒答，且给出强结论，记作 hallucination=1。
    非拒答题先不做复杂 NLP 判断。
    """
    strong_claim_keywords = [
        "一定",
        "必然",
        "明确构成",
        "可以直接认定",
        "必定",
    ]

    if should_refuse and not looks_like_refusal(answer):
        if any(keyword in answer for keyword in strong_claim_keywords):
            return 1
        return 1

    return 0


def judge_refusal_appropriate(answer: str, should_refuse: bool) -> int:
    if should_refuse:
        return 1 if looks_like_refusal(answer) else 0
    else:
        return 0 if looks_like_refusal(answer) else 1


def evaluate_one(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    reference_answer = item["reference_answer"]
    expected_articles = item.get("expected_articles", [])
    should_refuse = item.get("should_refuse", False)

    result = answer_with_sources(question)
    answer = result.get("answer", "")
    sources = result.get("sources", [])
    confidence = result.get("confidence", "unknown")

    answer_correct = judge_answer_correct(answer, reference_answer, should_refuse)
    citation_correct = judge_citation_correct(sources, expected_articles, should_refuse)
    hallucination = judge_hallucination(answer, should_refuse)
    refusal_appropriate = judge_refusal_appropriate(answer, should_refuse)

    overall_score = answer_correct + citation_correct + refusal_appropriate - hallucination

    return {
        "id": item.get("id", ""),
        "question": question,
        "type": item.get("type", "unknown"),
        "note": item.get("note", ""),
        "should_refuse": should_refuse,
        "reference_answer": reference_answer,
        "expected_articles": expected_articles,
        "generated_answer": answer,
        "returned_articles": [s.get("article", "") for s in sources],
        "confidence": confidence,
        "answer_correct": answer_correct,
        "citation_correct": citation_correct,
        "hallucination": hallucination,
        "refusal_appropriate": refusal_appropriate,
        "overall_score": overall_score,
    }


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {}

    answer_correct_rate = sum(x["answer_correct"] for x in results) / total
    citation_correct_rate = sum(x["citation_correct"] for x in results) / total
    hallucination_rate = sum(x["hallucination"] for x in results) / total
    refusal_appropriate_rate = sum(x["refusal_appropriate"] for x in results) / total
    avg_overall_score = sum(x["overall_score"] for x in results) / total

    by_type = {}
    for item in results:
        t = item["type"]
        by_type.setdefault(t, [])
        by_type[t].append(item)

    by_type_summary = {}
    for t, items in by_type.items():
        n = len(items)
        by_type_summary[t] = {
            "count": n,
            "answer_correct_rate": sum(x["answer_correct"] for x in items) / n,
            "citation_correct_rate": sum(x["citation_correct"] for x in items) / n,
            "hallucination_rate": sum(x["hallucination"] for x in items) / n,
            "refusal_appropriate_rate": sum(x["refusal_appropriate"] for x in items) / n,
            "avg_overall_score": sum(x["overall_score"] for x in items) / n,
        }

    return {
        "total": total,
        "answer_correct_rate": answer_correct_rate,
        "citation_correct_rate": citation_correct_rate,
        "hallucination_rate": hallucination_rate,
        "refusal_appropriate_rate": refusal_appropriate_rate,
        "avg_overall_score": avg_overall_score,
        "by_type": by_type_summary,
    }


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fields = [
        "id",
        "question",
        "type",
        "note",
        "should_refuse",
        "confidence",
        "answer_correct",
        "citation_correct",
        "hallucination",
        "refusal_appropriate",
        "overall_score",
        "expected_articles",
        "returned_articles",
        "reference_answer",
        "generated_answer",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            row_copy = row.copy()
            row_copy["expected_articles"] = " | ".join(row_copy.get("expected_articles", []))
            row_copy["returned_articles"] = " | ".join(row_copy.get("returned_articles", []))
            writer.writerow(row_copy)


def main():
    dataset_path = os.getenv("ANSWER_EVAL_DATASET_PATH", DEFAULT_DATASET_PATH)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到回答级评测数据集：{dataset_path}")

    ensure_results_dir()

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        results.append(evaluate_one(item))

    summary = summarize(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

    json_path = os.path.join(RESULTS_DIR, f"{dataset_name}_answer_eval_{timestamp}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{dataset_name}_answer_eval_{timestamp}.csv")

    payload = {
        "dataset_path": dataset_path,
        "generated_at": timestamp,
        "summary": summary,
        "details": results,
    }

    save_json(json_path, payload)
    save_csv(csv_path, results)

    print("\n" + "=" * 60)
    print("🧪 回答级评测结果")
    print("=" * 60)
    print(f"测试集: {dataset_path}")
    print(f"样本数: {summary.get('total', 0)}")
    print("-" * 60)
    print(f"Answer Correct Rate     : {summary.get('answer_correct_rate', 0):.2%}")
    print(f"Citation Correct Rate   : {summary.get('citation_correct_rate', 0):.2%}")
    print(f"Hallucination Rate      : {summary.get('hallucination_rate', 0):.2%}")
    print(f"Refusal Appropriate Rate: {summary.get('refusal_appropriate_rate', 0):.2%}")
    print(f"Avg Overall Score       : {summary.get('avg_overall_score', 0):.2f}")
    print("-" * 60)

    by_type = summary.get("by_type", {})
    for t, s in by_type.items():
        print(
            f"[{t}] "
            f"count={s['count']} "
            f"answer={s['answer_correct_rate']:.2%} "
            f"citation={s['citation_correct_rate']:.2%} "
            f"hallucination={s['hallucination_rate']:.2%} "
            f"refusal={s['refusal_appropriate_rate']:.2%} "
            f"avg_score={s['avg_overall_score']:.2f}"
        )

    print("-" * 60)
    print(f"结果 JSON: {json_path}")
    print(f"结果 CSV : {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()