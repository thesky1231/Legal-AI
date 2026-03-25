import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
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


def check_db():
    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库文件夹不存在：{DB_PATH}，请先运行 ingest.py")
        return

    print(f"🕵️ 正在检查数据库：{DB_PATH}")
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        collections = client.list_collections()
        print(f"📂 当前集合：{[c.name for c in collections]}")

        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=build_embeddings(),
        )

        all_data = vectorstore.get()
        count = len(all_data.get("ids", []))
        print(f"📊 集合 '{COLLECTION_NAME}' 文档数：{count}")

        if count == 0:
            print("⚠️ 数据库是空的，请检查 ingest.py 的切分与入库逻辑。")
            return

        print("✅ 数据库中存在数据，开始测试检索 ...")
        docs = vectorstore.similarity_search("故意杀人", k=3)

        for i, doc in enumerate(docs, start=1):
            print(f"\n--- 检索结果 {i} ---")
            print(f"article: {doc.metadata.get('article', '未知')}")
            print(f"source: {doc.metadata.get('source', '未知')}")
            print(doc.page_content[:200], "...")
    except Exception as e:
        print(f"❌ 检查数据库时发生错误：{e}")


if __name__ == "__main__":
    check_db()