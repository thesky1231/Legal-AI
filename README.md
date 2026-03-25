# Legal-AI

一个面向法律场景的 **RAG 问答系统**，支持法律知识入库、检索增强生成、法条依据返回、低置信度保守回答、回答级评测，以及基于 Docker / Compose 的服务化部署。

> 这个项目的核心目标不是“让模型看起来会回答”，而是让系统在 **有依据、可解释、可评估、可部署** 的前提下回答法律问题。

---

## 项目亮点

- **法律知识库构建**：完成 PDF 清洗、法条级切分、Embedding 向量化入库与 Chroma 持久化存储  
- **检索链路评测**：实现 Dense Retrieval、BM25、Hybrid Retrieval、CrossEncoder Rerank 等多种方案，并基于两套自建黄金测试集完成自动化评测  
- **回答级评测**：围绕答案正确性、法条引用准确性、幻觉率与拒答合理性进行量化分析，并输出 JSON / CSV  
- **策略路由优化**：基于回答级评测发现系统在高风险问题上的短板，进一步引入轻量问题分类与策略路由  
- **风险控制能力**：在当前自建回答级测试集上实现 **100% Answer Correct / Citation Correct / Refusal Appropriate，Hallucination Rate 0%**  
- **工程化交付**：完成 FastAPI 服务化、Streamlit 前端、Docker / Compose 容器化部署、日志与接口测试，形成从本地验证到工程交付的完整闭环  

---

## 项目定位

这个项目不是单纯的聊天 demo，而是一个偏 **AI 应用工程 / RAG 工程 / LLM 应用开发 / AI 后端** 方向的项目原型。

它重点体现的是：

- 如何把法律问答问题转化为 **检索增强 + 风险控制** 的系统设计问题
- 如何通过 **评测驱动** 而不是凭感觉去优化 RAG
- 如何把一个本地可运行的项目，推进到 **API 服务化 + Docker 部署** 的工程形态

---

## 评测结果

### 1. 检索级评测

基于两套自建黄金测试集，对不同检索方案进行了对比评测。

#### 测试集 1：`golden_dataset.json`
- BM25：Recall@3 = 0.00%，Recall@5 = 0.00%
- Vector：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid：Recall@3 = 92.50%，Recall@5 = 92.50%
- Hybrid + Rerank：Recall@3 = 92.50%，Recall@5 = 92.50%

#### 测试集 2：`golden_dataset_v2.json`
- BM25：Recall@3 = 0.00%，Recall@5 = 0.00%
- Vector：Recall@3 = 95.00%，Recall@5 = 95.00%
- Hybrid：Recall@3 = 95.00%，Recall@5 = 95.00%
- Hybrid + Rerank：不同阶段实验中出现过负优化风险

#### 结论
在当前法律条文问答场景下，**Dense Retrieval 表现最稳定**。  
这也说明检索优化不能靠堆模块，而必须基于数据验证。

---

### 2. 回答级评测

在检索级评测基础上，进一步构建回答级评测框架，从以下四个维度评估系统表现：

- `answer_correct`：答案是否正确
- `citation_correct`：法条引用是否正确
- `hallucination`：是否出现明显胡编
- `refusal_appropriate`：证据不足时是否合理拒答/保守回答

#### 当前结果：`answer_eval_dataset.json`
- Answer Correct Rate：**100.00%**
- Citation Correct Rate：**100.00%**
- Hallucination Rate：**0.00%**
- Refusal Appropriate Rate：**100.00%**
- Avg Overall Score：**3.00**

#### 题型分布结果
- `direct_answer`：100%
- `definition`：100%
- `complex_reasoning`：100%
- `confusing`：100%
- `rewrite`：100%
- `should_refuse`：100%

#### 结论
通过引入轻量问题分类与策略路由，系统在高风险场景下的稳健性明显提升，尤其是将回答级评测中的 **hallucination rate 压降至 0**。

---

## 核心设计思路

### 1. 为什么是法条级切分，而不是固定 chunk
法律文本天然具有清晰的结构边界。  
按法条切分相比固定 chunk size 更适合：

- 检索时命中完整法条语义单元
- 回答时返回明确出处
- 后续做法条依据解释与风险控制

### 2. 为什么不直接让大模型回答
因为法律场景对依据要求很高，纯大模型直接回答容易出现：

- 幻觉
- 无依据结论
- 难以验证回答来源
- 难以评估系统效果

RAG 的价值就在于让回答建立在**可检索、可引用、可解释**的法条证据之上。

### 3. 为什么引入回答级评测
仅有检索 Recall 还不够，因为真正有业务价值的是：

- 最终答案是否正确
- 引用法条是否准确
- 系统是否会胡编
- 证据不足时能不能稳住

所以项目后期重点从“检得到”升级到了“答得对”。

### 4. 为什么要做问题分类与策略路由
回答级评测表明，系统真正容易出问题的不是普通直查题，而是：

- 应拒答题
- 易混淆题
- 复杂分析题

因此项目后续不再使用统一 pipeline 处理所有问题，而是按问题类型采用不同策略，提高系统稳健性。

---

## 问题分类与策略路由

当前系统对问题做轻量分类，并采用差异化处理策略：

### 1. `direct_answer`
适用于法条直查类问题  
策略：
- Dense Retrieval
- 简洁直接回答
- 返回法条依据

### 2. `definition`
适用于法律概念、术语解释类问题  
策略：
- 先解释概念
- 再给法条依据
- 避免过度保守

### 3. `confusing`
适用于近义罪名、易混淆问题  
策略：
- 强制比较两个概念
- 明确区分点
- 尽可能分别给出对应法条依据

### 4. `complex_reasoning`
适用于多法条、多条件、复杂分析问题  
策略：
- 更保守的回答方式
- 明确指出需要补充的案件事实
- 避免武断下案件结论

### 5. `should_refuse`
适用于不应直接定性或量刑的问题  
策略：
- 优先使用保守回答模板
- 不直接作出确定性结论
- 引导用户补充具体事实

---

## 系统架构

### 数据入库流程
1. 读取法律 PDF
2. 文本清洗（去除页眉页脚、广告噪声等）
3. 按法条粒度切分
4. 生成 embedding
5. 写入 Chroma 持久化数据库

### 问答流程
1. 用户输入法律问题
2. 执行检索召回
3. 根据问题类型进行路由
4. 按不同策略组织 prompt
5. 生成回答
6. 返回答案 + 法条依据 + 置信度 + 问题类型

---

## 技术栈

### 后端
- Python
- FastAPI
- Pydantic

### 检索与模型
- LangChain
- Chroma
- HuggingFace Embeddings
- Sentence Transformers
- Ollama

### 前端
- Streamlit

### 工程化
- Docker
- Docker Compose
- 日志系统
- 接口测试脚本

---

## 项目结构

```bash
.
├── app.py                     # Streamlit 前端
├── config.py                  # 项目配置
├── logger.py                  # 日志模块
├── server.py                  # FastAPI 后端
├── requirements.txt           # Python 依赖
├── Dockerfile                 # Docker 镜像构建文件
├── docker-compose.yml         # Docker Compose 编排
├── README.md
├── data/
│   ├── law.pdf
│   ├── golden_dataset.json
│   ├── golden_dataset_v2.json
│   └── answer_eval_dataset.json
├── db/                        # 本地知识库
├── db_docker/                 # 容器环境知识库
├── eval/
│   ├── eval_baseline.py       # 检索级评测
│   ├── eval_answer.py         # 回答级评测
│   └── results/               # 评测结果输出
├── scripts/
│   ├── ingest.bat
│   ├── serve.bat
│   └── down.bat
├── tests/
│   └── test_api.py
└── src/
    ├── etl/
    │   └── ingest.py          # 法律文本清洗、切分、入库
    └── rag/
        └── chain.py           # 检索链路、路由与问答逻辑
```

---

## 本地运行

### 1. 构建知识库
```bash
python -m src.etl.ingest
```

### 2. 检查数据库
```bash
python debug_db.py
```

### 3. 跑检索级评测
```bash
python -m eval.eval_baseline
```

### 4. 跑回答级评测
```bash
python -m eval.eval_answer
```

### 5. 启动后端
```bash
python server.py
```

### 6. 启动前端
```bash
streamlit run app.py
```

### 7. 跑接口测试
```bash
python tests/test_api.py
```

---

## Docker / Compose 运行

### 1. 构建镜像
```bash
docker build -t legal-ai-backend .
```

### 2. 容器内构建知识库
```bash
docker compose run --rm legal-ai-ingest
```

### 3. 启动后端服务
```bash
docker compose up legal-ai-backend
```

### 4. 关闭服务
```bash
docker compose down
```

---

## API 接口

### 健康检查
- `GET /health`

### 检索接口
- `POST /api/retrieve`

请求示例：
```json
{
  "query": "故意杀人罪如何认定？",
  "top_k": 3
}
```

### 问答接口
- `POST /api/chat`

请求示例：
```json
{
  "query": "故意杀人罪如何认定？"
}
```

返回内容包括：
- `answer`
- `sources`
- `confidence`
- `question_type`
- `latency_ms`

---

## 工程亮点

- 使用 **法条级切分** 替代固定 chunk，提升法律场景检索与解释能力  
- 同时做了 **检索级评测 + 回答级评测**，形成完整优化闭环  
- 基于回答级评测发现系统在高风险问题上的短板，并进一步引入 **轻量问题分类与策略路由**  
- 在当前回答级测试集上实现 **100% Answer Correct / Citation Correct / Refusal Appropriate，Hallucination Rate 0%**  
- 完成 FastAPI、Streamlit、Docker、Compose、日志、测试等完整工程链路  
- 在 Docker 化过程中，解决了模型缓存挂载、宿主机模型服务访问、向量数据库环境兼容等工程问题  

---

## 适配岗位方向

这个项目更适合展示以下方向的能力：

- AI 应用工程
- RAG 工程
- LLM 应用开发
- AI 后端开发
- 智能问答系统开发

---

## 项目收获

这个项目让我真正学到的，不只是怎么把 RAG 跑起来，而是：

- 如何把一个 AI demo 做成可解释、可评测、可部署的系统原型
- 如何通过评测发现真实短板，而不是靠感觉优化
- 如何在高风险场景下引入策略路由和保守回答机制
- 如何处理本地环境与部署环境之间的差异

---

## 后续优化方向

### P1：继续扩大评测集
- 更复杂的法律问答样本
- 更多自然语言改写
- 更多混淆题和复杂题
- 更大规模的回答级评测集

### P2：轻量 Agent
- 检索 Tool
- 法条摘要 Tool
- 最终回答 Tool
- 为复杂法律问题引入多步处理能力

### P3：自动化评测增强
- 继续扩展 Answer Eval
- 可选接入 RAGAS 等自动化指标体系

### P4：SFT / LoRA
- 在已有评测闭环基础上，再探索专业表达风格增强
- 避免“只学会律师腔，没有真正提升可靠性”

---

## 说明

当前所有评测结果均基于**自建测试集**完成，主要用于验证系统设计与优化方向。  
后续还需要在更大规模、更复杂、更接近真实用户输入的数据上继续验证泛化能力。

---
