# Legal-AI

一个面向法律场景的 RAG 问答系统，支持法律知识入库、检索增强生成、法条依据返回、问题分类与策略路由、回答级评测，以及基于 FastAPI + Docker 的服务化部署。

---

## 1. 项目简介

通用大模型直接回答法律问题时，常见问题包括：

- 回答缺乏法条依据
- 易出现幻觉或过度自信结论
- 难以验证回答来源
- 难以量化系统效果

本项目以法律条文问答为场景，构建了一个端到端的法律 RAG 系统：

- 从法律 PDF 中抽取法条，构建本地知识库
- 完成检索级评测与回答级评测
- 面向高风险问题设计轻量问题分类与策略路由
- 返回答案的同时展示法条依据
- 在证据不足或不应直接下结论时进行保守回答
- 支持 FastAPI 后端、Streamlit 前端以及 Docker / Compose 部署

---

## 2. 核心能力

### 2.1 法律知识库构建
- 读取法律 PDF 文本
- 清洗页眉页脚、广告等噪声
- 按法条粒度进行切分
- 生成 embedding 并写入 Chroma 持久化数据库

### 2.2 检索增强问答
- Dense Retrieval
- BM25 Retrieval
- Hybrid Retrieval
- CrossEncoder Rerank
- 基于检索结果生成结构化法律回答

### 2.3 问题分类与策略路由
- `direct_answer`：法条直查型
- `definition`：法律概念 / 术语定义型
- `confusing`：易混淆罪名 / 概念区分型
- `complex_reasoning`：多条件、多角色、多情节分析型
- `should_refuse`：不应直接下案件结论 / 量刑结论型

### 2.4 可解释与风险控制
- 返回回答结论
- 返回法条依据与原文片段
- 支持低置信度保守回答
- 支持应拒答题模板化处理，降低幻觉风险

### 2.5 工程化与部署
- FastAPI 提供 `/health`、`/api/retrieve`、`/api/chat`
- Streamlit 提供交互前端
- Docker / Compose 支持容器化构建与运行
- 结构化日志、环境变量配置、最小化接口测试脚本

---

## 3. 项目结构

```bash
.
├── app.py
├── config.py
├── logger.py
├── server.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── data/
│   ├── law.pdf
│   ├── golden_dataset.json
│   ├── golden_dataset_v2.json
│   └── answer_eval_dataset.json
├── db/
├── db_docker/
├── eval/
│   ├── eval_baseline.py
│   ├── eval_answer.py
│   └── results/
├── scripts/
│   ├── ingest.bat
│   ├── serve.bat
│   └── down.bat
├── tests/
│   └── test_api.py
└── src/
    ├── etl/
    │   └── ingest.py
    └── rag/
        └── chain.py
```

---

## 4. 系统流程

### 4.1 数据入库流程
1. 加载法律 PDF
2. 文本清洗
3. 按法条粒度切分
4. 使用 `BAAI/bge-m3` 生成向量
5. 写入 Chroma 持久化数据库

### 4.2 问答流程
1. 用户输入法律问题
2. 问题分类（直查 / 定义 / 混淆 / 复杂 / 应拒答）
3. 执行检索召回
4. 按题型选择不同 prompt 与回答策略
5. 生成结构化回答
6. 返回答案 + 法条依据 + 置信度 + 问题类型

---

## 5. 技术栈

- Python
- FastAPI
- Streamlit
- LangChain
- Chroma
- HuggingFace Embeddings
- Sentence Transformers
- Ollama
- Docker
- Docker Compose

---

## 6. 评测结果

### 6.1 检索级评测

基于两套自建黄金测试集，对 Dense / BM25 / Hybrid / Rerank 进行对比。

#### 数据集 1：`golden_dataset.json`
- BM25：Recall@3 = 0.00%，Recall@5 = 0.00%
- Vector：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid + Rerank：Recall@3 = 92.50%，Recall@5 = 92.50%

#### 数据集 2：`golden_dataset_v2.json`
- BM25：Recall@3 = 0.00%，Recall@5 = 0.00%
- Vector：Recall@3 = 95.00%，Recall@5 = 95.00%
- Hybrid：Recall@3 = 95.00%，Recall@5 = 95.00%
- Hybrid + Rerank：在复杂集上未带来稳定增益

**结论：**
当前法律条文问答场景下，Dense Retrieval 是最稳定的主方案；BM25 暂未体现增益，Rerank 也不是天然正收益。

### 6.2 回答级评测

基于自建 `answer_eval_dataset.json`，围绕以下四个维度进行回答级评测：

- Answer Correct
- Citation Correct
- Hallucination Rate
- Refusal Appropriate

#### 当前结果
- Answer Correct Rate：100.00%
- Citation Correct Rate：100.00%
- Hallucination Rate：0.00%
- Refusal Appropriate Rate：100.00%

**结论：**
通过问题分类与策略路由，系统在当前自建回答级测试集上实现了稳定的答案质量与风险控制表现。

---

## 7. 项目亮点

- 基于法律文本结构进行法条级切分，而非固定 chunk
- 同时完成检索级评测与回答级评测，形成完整评测闭环
- 不盲目堆模块，而是通过评测发现 Dense Retrieval 更适合当前场景
- 针对高风险问题引入轻量问题分类与策略路由，将回答级评测中的幻觉率压降至 0
- 支持依据返回、低置信度保守回答和应拒答场景治理
- 完成 FastAPI + Streamlit + Docker / Compose 的完整工程链路
- 在容器部署中解决了模型缓存挂载、宿主机 Ollama 访问、Chroma 持久化兼容等工程问题

---

## 8. 本地运行

### 8.1 构建知识库
```bash
python -m src.etl.ingest
```

### 8.2 检查数据库
```bash
python debug_db.py
```

### 8.3 运行检索评测
```bash
python -m eval.eval_baseline
```

### 8.4 运行回答级评测
```bash
python -m eval.eval_answer
```

### 8.5 启动后端
```bash
python server.py
```

### 8.6 启动前端
```bash
streamlit run app.py
```

---

## 9. Docker / Compose 运行

### 9.1 构建知识库
```bash
docker compose run --rm legal-ai-ingest
```

### 9.2 启动后端
```bash
docker compose up legal-ai-backend
```

### 9.3 关闭服务
```bash
docker compose down
```

---

## 10. API 接口

### 健康检查
- `GET /health`

### 检索接口
- `POST /api/retrieve`

### 问答接口
- `POST /api/chat`

接口返回：
- answer
- sources
- confidence
- question_type
- latency_ms

---

## 11. 后续方向

- 接入更大规模回答级数据集，继续验证泛化能力
- 按题型进一步扩展策略路由
- 引入轻量 Agent 处理更复杂的多步法律问题
- 接入更系统的自动化评测框架（如 RAGAS）
- 增加结果可视化与更细粒度监控指标
