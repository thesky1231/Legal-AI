from functools import lru_cache
from typing import List, Dict, Any
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.rag.chain import get_rag_chain, retrieve, answer_with_sources
from config import SERVER_HOST, SERVER_PORT
from logger import get_logger


logger = get_logger("server")
app = FastAPI(title="法律助手 Pro 后端", version="1.1.0")


@lru_cache(maxsize=1)
def load_chain():
    return get_rag_chain()


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="用户问题")


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="检索问题")
    top_k: int = Field(3, ge=1, le=20, description="返回条数")


@app.get("/health")
def health():
    logger.info("health check")
    return {"status": "ok", "service": "legal-ai-backend"}


@app.post("/api/retrieve")
def api_retrieve(request: RetrieveRequest):
    start = time.perf_counter()
    try:
        logger.info(f"/api/retrieve query={request.query!r} top_k={request.top_k}")
        docs = retrieve(request.query, top_k=request.top_k)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs, start=1):
            results.append(
                {
                    "rank": idx,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        logger.info(f"/api/retrieve success count={len(results)} latency_ms={latency_ms}")
        return {
            "query": request.query,
            "top_k": request.top_k,
            "count": len(results),
            "latency_ms": latency_ms,
            "results": results,
        }
    except Exception as e:
        import traceback
        logger.error(f"/api/retrieve failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def api_chat(request: ChatRequest):
    start = time.perf_counter()
    try:
        logger.info(f"/api/chat query={request.query!r}")
        result = answer_with_sources(request.query)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            f"/api/chat success confidence={result['confidence']} "
            f"sources={len(result['sources'])} latency_ms={latency_ms}"
        )

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "latency_ms": latency_ms,
        }
    except Exception as e:
        import traceback
        logger.error(f"/api/chat failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def api_chat(request: ChatRequest):
    start = time.perf_counter()
    try:
        logger.info(f"/api/chat query={request.query!r}")
        result = answer_with_sources(request.query)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            f"/api/chat success type={result['question_type']} "
            f"confidence={result['confidence']} "
            f"sources={len(result['sources'])} latency_ms={latency_ms}"
        )

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "question_type": result["question_type"],
            "latency_ms": latency_ms,
        }
    except Exception as e:
        import traceback
        logger.error(f"/api/chat failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)