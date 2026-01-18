import json
import sys
import os
from tqdm import tqdm # 如果报错，请运行 pip install tqdm

# 把项目根目录加入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.rag.chain import get_retriever

def evaluate_recall():
    print("🚀 开始评估 Baseline (Recall@3)...")
    
    # 路径指向你的测试集
    dataset_path = "./data/golden_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ 错误：找不到测试集 {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # 获取检索器 (只查库，不生成)
    retriever = get_retriever()
    
    correct_count = 0
    total_count = len(dataset)
    
    print(f"共加载 {total_count} 条测试数据，正在检索...")

    for item in tqdm(dataset):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        # 1. 检索
        retrieved_docs = retriever.invoke(question)
        
        # 2. 检查：标准答案是不是在检索出来的文档里？
        is_hit = False
        context_text = "".join([doc.page_content for doc in retrieved_docs])
        
        # 简单粗暴的字符串包含匹配
        if ground_truth in context_text:
            is_hit = True
        
        if is_hit:
            correct_count += 1
        else:
            # 打印第一条错题，方便调试（不想看可以注释掉）
            # print(f"\n❌ Miss: {question}")
            pass

    recall_rate = correct_count / total_count
    print("\n" + "="*30)
    print(f"📊 评估报告")
    print(f"✅ Recall@3: {recall_rate:.2%}")
    print("="*30)

if __name__ == "__main__":
    evaluate_recall()