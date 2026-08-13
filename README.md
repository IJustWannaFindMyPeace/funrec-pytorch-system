# FunRec PyTorch Recommendation System

基于 Datawhale [FunRec](https://github.com/datawhalechina/fun-rec) Web 推荐系统完成的 PyTorch 工程化迁移项目。

本项目已将原有 TensorFlow YouTubeDNN 召回模型和 DeepFM 排序模型迁移至 PyTorch，并保留离线特征处理、模型训练、模型评估、工件部署、在线推理、冷启动、多样性重排、FastAPI 后端和前端展示链路。

> 本项目用于学习和非商业用途。原项目版权及许可归原作者所有。

## 核心能力

- PyTorch YouTubeDNN 双塔召回
- PyTorch DeepFM 点击率排序
- CUDA GPU 训练与 CPU/CUDA 在线推理
- 全量 softmax 召回训练
- Recall、Hit Rate、NDCG 离线评估
- 全局热门召回基线对比
- 断点续训、最佳 checkpoint 与训练历史
- 模型、向量和词表工件部署
- YouTubeDNN 召回、DeepFM 排序和多样性重排在线链路
- Redis 用户画像、PostgreSQL 业务数据和 Elasticsearch 搜索
- FastAPI 后端与 Web 前端

## 系统流程

```text
MovieLens-1M / IMDb
        │
        ▼
特征预处理
        │
        ├── YouTubeDNN 召回训练
        │       ├── 用户塔模型
        │       ├── 物品向量
        │       └── 召回离线评估
        │
        └── DeepFM 排序训练
                ├── 最佳 checkpoint
                ├── 排序模型
                └── 训练历史
        │
        ▼
模型与词表部署
        │
        ▼
在线召回 → DeepFM 排序 → 多样性重排 → FastAPI
```

## 技术栈

- Python 3.11
- PyTorch 2.12
- FastAPI
- NumPy、Pandas、scikit-learn
- PostgreSQL
- Redis
- Elasticsearch
- Docker Compose
- Vue 前端
- pytest

## 项目结构

```text
funrec-pytorch-system/
├── backend/
│   ├── app/                         # FastAPI 应用
│   ├── modeling/
│   │   ├── youtubednn.py            # PyTorch YouTubeDNN
│   │   └── deepfm.py                # PyTorch DeepFM
│   ├── offline/
│   │   ├── evaluation/              # 召回评估与热门基线
│   │   ├── feature/                 # 召回、排序特征处理
│   │   ├── storage/                 # Redis 写入和本地模型部署
│   │   ├── training/                # 训练数据、trainer 和训练入口
│   │   └── pipeline.py              # 离线流水线入口
│   ├── online/
│   │   ├── recall/                  # 在线召回
│   │   ├── ranking/                 # 在线 DeepFM 排序
│   │   ├── reranking/               # 多样性重排
│   │   └── cold_start/              # 冷启动策略
│   ├── tests/
│   ├── Makefile
│   └── pyproject.toml
├── frontend/
├── docker-compose.yaml
├── .env.example
└── README.md
```

## 环境要求

- Python 3.11
- Git
- Docker Desktop
- 推荐使用 `uv`
- CUDA 训练可选；CPU 也可以运行
- 本项目实测训练环境：NVIDIA GeForce RTX 5070

## 数据准备

下载并解压 MovieLens-1M 数据：

[funrec-movielens-1m.zip](https://funrec-datasets.s3.eu-west-3.amazonaws.com/funrec-movielens-1m.zip)

复制环境变量模板：

### PowerShell

```powershell
Copy-Item .\.env.example .\.env
code .\.env
```

### Bash

```bash
cp .env.example .env
```

配置两个绝对路径：

```dotenv
FUNREC_RAW_DATA_PATH=C:\path\to\raw\data
FUNREC_PROCESSED_DATA_PATH=C:\path\to\processed\data
```

两个目录含义不同：

- `FUNREC_RAW_DATA_PATH`：解压后的原始数据目录
- `FUNREC_PROCESSED_DATA_PATH`：预处理数据、checkpoint 和部署工件目录

## 安装依赖

进入后端目录：

```powershell
Set-Location .\backend
```

使用 Python 3.11 同步运行依赖和测试依赖：

```powershell
uv sync --python 3.11 --extra dev
```

项目依赖以 `backend/pyproject.toml` 为准，推荐使用 `uv sync`。

## 启动基础服务

在项目根目录运行：

```powershell
docker compose up --build -d
```

检查容器：

```powershell
docker compose ps
```

查看日志：

```powershell
docker compose logs -f
```

停止服务：

```powershell
docker compose down
```

Docker Compose 会启动：

- PostgreSQL：`localhost:5432`
- Redis：`localhost:6379`
- Elasticsearch：`localhost:9200`
- FastAPI：`localhost:8000`
- 前端：`localhost:3000`

## 导入业务数据

进入后端目录：

```powershell
Set-Location .\backend
```

导入数据库并创建测试用户：

```powershell
make ingest-data-to-database
```

该命令实际执行：

```powershell
uv run scripts/ingest_data_to_database.py --reset --create-test-user
```

索引电影到 Elasticsearch：

```powershell
make index-movies-to-elasticsearch
```

该命令实际执行：

```powershell
uv run scripts/index_movies_elasticsearch.py
```

## 运行完整离线流水线

确保 PostgreSQL、Redis 和 Elasticsearch 已启动，然后在 `backend` 目录运行：

```powershell
make run-offline-pipeline
```

等价命令：

```powershell
uv run python -m offline.pipeline --steps all --flush-redis
```

完整流水线依次执行：

1. 召回特征预处理
2. 排序特征预处理
3. YouTubeDNN 召回训练
4. DeepFM 排序训练
5. 用户特征写入 Redis
6. 模型与工件部署

也可以单独执行步骤：

```powershell
uv run python -m offline.pipeline --steps retrieval_preprocess
uv run python -m offline.pipeline --steps ranking_preprocess
uv run python -m offline.pipeline --steps retrieval_training
uv run python -m offline.pipeline --steps ranking_training
uv run python -m offline.pipeline --steps ingest --flush-redis
uv run python -m offline.pipeline --steps deploy
```

多个步骤使用逗号分隔：

```powershell
uv run python -m offline.pipeline --steps retrieval_preprocess,retrieval_training,deploy
```

## 单独训练 YouTubeDNN

```powershell
Set-Location .\backend

python -m offline.training.train_retrieval `
    --epochs 3 `
    --device cuda `
    --num-workers 0
```

使用 CPU：

```powershell
python -m offline.training.train_retrieval `
    --epochs 3 `
    --device cpu `
    --num-workers 0
```

断点续训：

```powershell
python -m offline.training.train_retrieval `
    --epochs 5 `
    --device cuda `
    --num-workers 0 `
    --resume
```

## 评估 YouTubeDNN

```powershell
Set-Location .\backend
python -m offline.evaluation.evaluate_retrieval --device cuda
```

评估协议：

- 测试用户：6,040
- 每位用户一个目标物品
- 过滤用户历史物品
- 候选物品：3,883
- Embedding 维度：16
- 指标：Recall、Hit Rate、NDCG
- 截断位置：5 和 10

结果保存在：

```text
<FUNREC_PROCESSED_DATA_PATH>/web_project/retrieval_metrics.json
```

## 单独训练 DeepFM

```powershell
Set-Location .\backend

python -m offline.training.train_ranking `
    --epochs 3 `
    --device cuda `
    --num-workers 0 `
    --patience 2
```

使用 CPU：

```powershell
python -m offline.training.train_ranking `
    --epochs 3 `
    --device cpu `
    --num-workers 0 `
    --patience 2
```

断点续训：

```powershell
python -m offline.training.train_ranking `
    --epochs 5 `
    --device cuda `
    --num-workers 0 `
    --patience 2 `
    --resume
```

## 部署模型工件

部署全部模型：

```powershell
Set-Location .\backend
python -m offline.storage.local_deploy
```

仅部署召回模型：

```powershell
python -m offline.storage.local_deploy --recall-only
```

仅部署排序模型：

```powershell
python -m offline.storage.local_deploy --ranking-only
```

部署目录：

```text
<FUNREC_PROCESSED_DATA_PATH>/web_project/deployed_models/
├── recall/
│   ├── retrieval_user_model.pt
│   ├── item_embeddings.npy
│   ├── movie_ids.npy
│   └── vocab_dict.pkl
└── ranking/
    ├── ranking_model.pt
    ├── model_config.pkl
    ├── feature_dict.pkl
    └── vocab_dict.pkl
```

## 启动本地 API

在 `backend` 目录运行：

```powershell
make launch-api
```

等价命令：

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问：

- API：<http://localhost:8000>
- Swagger：<http://localhost:8000/docs>
- 前端：<http://localhost:3000>

## 健康检查

```powershell
curl.exe http://localhost:8000/health
curl.exe http://localhost:9200
docker exec funrec-postgres pg_isready -U funrec -d funrec_db
docker exec funrec-redis redis-cli ping
```

## 测试

进入后端目录并安装开发依赖：

```powershell
Set-Location .\backend
uv sync --python 3.11 --extra dev
```

运行全部测试：

```powershell
make test
```

等价命令：

```powershell
uv run pytest
```

运行详细测试：

```powershell
make test-verbose
```

语法和格式检查：

```powershell
python -m compileall app modeling offline online
git diff --check
```

## 真实实验结果

以下结果均来自本仓库迁移后的 PyTorch 代码、MovieLens-1M 处理数据和实际运行日志，不是估算值。

### YouTubeDNN 召回训练

训练环境：

- GPU：NVIDIA GeForce RTX 5070
- Epoch：3
- Embedding 维度：16
- 候选物品：3,883

| Epoch | Train Loss | Validation Loss |
| ---: | ---: | ---: |
| 1 | 7.312562 | 7.208624 |
| 2 | 6.770610 | 6.940392 |
| 3 | 6.490195 | 6.787967 |

### YouTubeDNN 召回评估

| Metric | YouTubeDNN | Popularity baseline | Relative lift |
| --- | ---: | ---: | ---: |
| Recall@5 | 0.055629 | 0.008940 | 6.2222× |
| Hit Rate@5 | 0.055629 | 0.008940 | 6.2222× |
| NDCG@5 | 0.033633 | 0.005721 | 5.8793× |
| Recall@10 | 0.100662 | 0.019536 | 5.1525× |
| Hit Rate@10 | 0.100662 | 0.019536 | 5.1525× |
| NDCG@10 | 0.048037 | 0.009085 | 5.2874× |

YouTubeDNN 在 6,040 位测试用户上取得：

- Hit@5：336
- Hit@10：608
- 完整 GPU 检索评估耗时约 0.50 秒

热门基线取得：

- Hit@5：54
- Hit@10：118

由于每位测试用户只有一个相关目标物品，本评估协议下 Recall@K 与 Hit Rate@K 数值相同。

### DeepFM 排序训练

训练环境：

- GPU：NVIDIA GeForce RTX 5070
- 训练样本：1,925,407
- 验证样本：449,828
- Batch size：128
- Epoch：3

| Epoch | Train Loss | Train AUC | Validation Loss | Validation AUC |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.405689 | 0.866792 | 0.437134 | 0.843122 |
| 2 | 0.359169 | 0.898718 | 0.430139 | 0.852847 |
| 3 | 0.340651 | 0.909774 | 0.440420 | 0.854670 |

最佳 checkpoint：Epoch 3，Validation AUC 为 `0.854670`。

### 在线推荐链路

使用真实部署的 PyTorch 模型进行端到端 API 烟雾测试：

- YouTubeDNN 召回候选：20
- DeepFM 精排结果：10
- `ranking_strategy`：`deepfm`
- 历史电影重合：0
- 排序分数保持降序
- 召回分数得到保留
- 海报 URL 正常生成
- HTTP 状态：200

### 测试结果

迁移完成后的后端测试结果：

```text
102 passed
```

覆盖范围包括：

- YouTubeDNN 模型与训练工具
- 召回训练、评估和热门基线
- 召回模型部署与在线推理
- DeepFM 模型、数据集和训练工具
- 排序训练、部署与在线推理
- Elasticsearch 无连接时的快速降级
- 推荐服务和召回策略

## 已知行为

- 没有可用的 DeepFM 工件时，排序服务会退化到召回分数排序。
- Redis 或 Elasticsearch 不可用时，相关辅助召回和搜索功能会快速降级。
- YouTubeDNN 和 DeepFM 在线推理设备分别可以通过 `RECALL_DEVICE`、`RANKING_DEVICE` 设置。
- Docker 环境通过 `MODEL_DEPLOY_DIR=/app/tmp/web_project/deployed_models` 加载模型。
- CUDA 结果取决于本机 PyTorch、驱动和显卡兼容性。

## 测试账户

数据导入命令创建的测试账户：

```text
Email: test@funrec.com
Password: test123456
```

该账户仅用于本地开发环境，请勿用于生产环境。

## 项目来源

原始项目：

- [Datawhale FunRec](https://github.com/datawhalechina/fun-rec)

本仓库重点记录 TensorFlow 到 PyTorch 的模型与工程链路迁移。
