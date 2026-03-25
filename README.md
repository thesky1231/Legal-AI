# Legal-AI

一个面向法律场景的 RAG 问答系统，支持法律知识入库、检索增强生成、法条依据返回、低置信度保守回答，以及基于 Docker 的服务化部署。

## 项目简介
本项目以法律问答为场景，构建了一个端到端的 RAG 系统：从法律 PDF 中抽取法条、构建本地知识库、执行检索增强生成，并在回答中返回法条依据与来源片段。项目同时支持 FastAPI 后端、Streamlit 前端，以及 Docker 化部署。

## 核心能力
- 法律 PDF 文本清洗与法条级切分
- Embedding 向量化入库与 Chroma 持久化
- Dense Retrieval / BM25 / Hybrid / Rerank 检索链路
- 法条依据展示与低置信度保守回答
- FastAPI / Streamlit / Docker 完整链路

## 评测结果
- BM25：Recall@3 = 0.00%，Recall@5 = 0.00%
- Vector：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid + Rerank：Recall@3 = 92.50%，Recall@5 = 92.50%

结论：当前法律条文测试集上，Dense Retrieval 已具备较强召回能力；Hybrid 与 Rerank 暂未带来额外增益。

## 项目亮点
- 按法律文本天然结构进行法条级切分，而非固定 chunk size
- 自建黄金测试集与自动化评测脚本，形成数据驱动优化闭环
- 回答同时返回法条依据与来源片段，增强可解释性
- 增加低置信度保守回答机制，降低幻觉风险
- 完成 Docker 化部署，并解决模型缓存挂载、宿主机模型服务访问、向量数据库环境兼容等工程问题
