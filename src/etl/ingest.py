import os
import re
import shutil
from typing import List

import chromadb
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    PDF_PATH,
    DB_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
)


def build_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": NORMALIZE_EMBEDDINGS},
    )


def clean_text(text: str) -> str:
    """
    文本清洗：
    1. 去除 PDF 噪声页眉页脚
    2. 去除律师事务所广告信息
    3. 压缩多余空行
    """
    text = text.replace("|", " ").replace("｜", " ")

    noise_patterns = [
        r"京师律师事务所",
        r"JINGSH LAW FIRM",
        r"北京市京师律师事务所",
        r"赵荔律师团队",
        r"中国刑事辩护网提供",
        r"--- PAGE \d+ ---",
        r"\d+\s*/\s*101",
        r"\\",
        r"(?:北京市)+.*?京师律师大厦.*?100025.*?电话[:：][\d-]+",
        r"(?:北京市)+朝阳区.+?电话[:：][\d-]+",
        r"电话[:：]010-[\d-]+",
    ]

    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.S | re.I)

    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def split_law_text(full_text: str) -> List[Document]:
    """
    按“第X条 / 第X条之一”切分法条
    """
    print("✂️ 正在进行法条级切分...")

    pattern = r"(第\s*[零一二三四五六七八九十百千万0-9]+(?:[之\s]*[一二三四五六七八九十百千万0-9]+)?\s*条)"
    segments = re.split(pattern, full_text)

    documents: List[Document] = []

    for i in range(1, len(segments), 2):
        if i + 1 >= len(segments):
            continue

        article_header = segments[i].strip()
        content_body = segments[i + 1].strip()
        full_chunk = clean_text(article_header + content_body)

        if len(full_chunk) < 5:
            continue

        clean_title = article_header
        match_title = re.match(r"\s*【([^】]+)】", content_body)
        if match_title:
            clean_title += " " + match_title.group(1)

        doc = Document(
            page_content=full_chunk,
            metadata={
                "source": os.path.basename(PDF_PATH),
                "article": clean_title.replace(" ", ""),
            },
        )
        documents.append(doc)

    print(f"✅ 切分完成，共提取 {len(documents)} 条法律条款。")
    if len(documents) > 460:
        print("🎉 检测到较完整的子法条切分结果。")

    return documents


def ingest() -> None:
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"找不到 PDF 文件：{PDF_PATH}")

    print(f"📖 正在读取 {PDF_PATH} ...")
    loader = PyPDFLoader(PDF_PATH)
    raw_pages = loader.load()
    full_text = "\n".join(page.page_content for page in raw_pages)

    cleaned_text = clean_text(full_text)
    chunks = split_law_text(cleaned_text)

    print("⏳ 正在加载 embedding 模型 ...")
    embeddings = build_embeddings()

    print("💾 正在重建向量数据库 ...")
    if os.path.exists(DB_PATH):
        for name in os.listdir(DB_PATH):
            path = os.path.join(DB_PATH, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except Exception as e:
                print(f"⚠️ 清理旧文件失败: {path}, error={e}")
    else:
        os.makedirs(DB_PATH, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding + Ingest"):
        batch = chunks[i : i + batch_size]
        vectorstore.add_documents(batch)

    print("🎉 入库成功！")


if __name__ == "__main__":
    ingest()