# WebSim 推荐系统交互仿真平台

![What is WebSim](Method.png "WebSim Platform")

`WebSim` 是一个基于 Flask 的本地推荐系统仿真平台，提供两种交互页面：

- 网格页（`/`）：4 卡片展示，支持点击与翻页
- 滑动页（`/swipe`）：上下滑（含键盘上下键）浏览推荐

当前服务支持 3 个数据集和 7 个推荐模型，并可按会话记录曝光/点击统计，实时展示评分与热度。

> 路径说明：本文档默认使用 `/Users/chongzhang/WebSim`。若你本地目录仍为旧名 `/Users/chongzhang/Web_sim`，请将命令中的路径替换为旧目录即可。

## 当前能力（与代码一致）

- 数据集：
  - `ml1m`（MovieLens-1M）
  - `amazon_all_beauty`
  - `amazon_magazine_subscriptions`
- 推荐模型：
  - `sasrec`
  - `lightgcn`
  - `multvae`
  - `poprec`
  - `bprmf`
  - `gru4rec`
  - `bert4rec`
- 核心后端接口：
  - `GET /health`
  - `GET /api/init`
  - `POST /api/select`
  - `POST /api/next`
  - `POST /api/session/end`
  - `GET /poster/<dataset_key>/<item_id>`

## 目录结构（关键）

- `app.py`：Flask 入口与 API
- `recommender.py`：数据目录解析、模型引擎封装、卡片生成
- `dataset_utils.py`：训练数据加载（MovieLens/Amazon）
- `train_*.py`：7 种模型训练脚本
- `templates/` + `static/`：前端页面与脚本
- `scripts/`：启动、停止、批量训练与维护脚本
- `artifacts/`：模型权重与训练日志
- `docs/`：系统图与 UML 文档

## 环境安装

```bash
cd /Users/chongzhang/WebSim
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` 当前依赖：

- `flask==3.0.2`
- `torch>=2.6.0`
- `pandas>=2.2.0`
- `numpy>=1.26.0`

## 数据准备

默认会自动查找以下目录（也可用环境变量覆盖）：

- `../WebSim_Dataset/MM-ML-1M-main`
- `../WebSim_Dataset/Amazon_MM_2018/All_Beauty`
- `../WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions`

### MovieLens-1M 最低要求

- `ratings.dat`
- `movies.dat`
- 可选：`movies_details_clean.csv`、`posters/`

### Amazon_MM_2018 最低要求

- `raw/*.json.gz`（review 文件）
- `raw/meta_*.json.gz`（meta 文件，建议提供）
- 可选：`movies_details_clean.csv`、`posters/` 或 `images/`

## 快速启动

### 1) 直接启动 Flask（默认 19001，首页网格页）

```bash
cd /Users/chongzhang/WebSim
PORT=19001 python3 app.py
```

访问：

- 首页网格页：`http://127.0.0.1:19001/`
- 滑动页：`http://127.0.0.1:19001/swipe`
- 健康检查：`http://127.0.0.1:19001/health`

### 2) 推荐使用脚本启动滑动页（默认 19002）

```bash
cd /Users/chongzhang/WebSim
./scripts/run_swipe_page.sh
./scripts/stop_swipe_page.sh
```

默认地址：

- `http://127.0.0.1:19002/swipe`
- `http://127.0.0.1:19002/health`

可选环境变量：

```bash
HOST=127.0.0.1 PORT=19002 LOG_FILE=web.log OPEN_BROWSER=1 ./scripts/run_swipe_page.sh
PORT=19002 ./scripts/stop_swipe_page.sh
```

说明：

- `OPEN_BROWSER=0/false/no` 可禁用自动打开浏览器
- `LOG_FILE` 若为相对路径，则按项目根目录解析

## API 简要说明

### `GET /health`

返回服务状态、每个数据集可用模型列表与默认模型。

### `GET /api/init?dataset_key=...&model_name=...`

初始化会话并返回第一屏随机卡片（4 条）。

### `POST /api/select`

根据点击项更新历史并返回推荐第一页。请求体示例：

```json
{
  "movie_id": "296",
  "dataset_key": "ml1m",
  "model_name": "sasrec"
}
```

### `POST /api/next`

翻推荐下一页。请求体示例：

```json
{
  "dataset_key": "ml1m"
}
```

### `POST /api/session/end`

清理当前 session 统计与历史状态。

## 训练模型

### 单模型训练（示例）

```bash
cd /Users/chongzhang/WebSim
python3 train_sasrec.py   --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/sasrec_ml1m.pt   --epochs 10 --eval-ks 10,20
python3 train_lightgcn.py --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/lightgcn_ml1m.pt --epochs 30 --eval-ks 10,20
python3 train_multvae.py  --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/multvae_ml1m.pt  --epochs 30 --eval-ks 10,20
```

其它训练脚本：

- `train_poprec.py`
- `train_bprmf.py`
- `train_gru4rec.py`
- `train_bert4rec.py`

### Amazon_MM_2018 一键七模型训练

```bash
cd /Users/chongzhang/WebSim
./scripts/train_amazon_all_beauty_all.sh /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/All_Beauty
./scripts/train_amazon_magazine_subscriptions_all.sh /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions
```

或使用通用脚本：

```bash
./scripts/train_amazon_mm2018_all.sh <dataset_dir> <model_suffix>
```

例如：

```bash
./scripts/train_amazon_mm2018_all.sh /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/All_Beauty amazon_all_beauty
```

训练输出模型位于 `artifacts/*.pt`。

可通过环境变量覆盖各模型训练 epoch：

- `SASREC_EPOCHS`
- `LIGHTGCN_EPOCHS`
- `MULTVAE_EPOCHS`
- `BPRMF_EPOCHS`
- `GRU4REC_EPOCHS`
- `BERT4REC_EPOCHS`

## 环境变量（服务端）

### 数据目录

- `ML1M_DATASET_DIR`
- `AMAZON_ALL_BEAUTY_DATASET_DIR`
- `AMAZON_MAGAZINE_SUBSCRIPTIONS_DATASET_DIR`
- `DEFAULT_DATASET_KEY`（默认 `ml1m`）

### 服务

- `PORT`（默认 `19001`，脚本一般使用 `19002`）
- `FLASK_SECRET_KEY`
- `WEBSIM_INIT_RANDOM_SEED`（初始化随机采样种子，默认 `42`）

### 模型权重路径（示例）

- ML1M 支持：
  - `SASREC_ML1M_MODEL_PATH`（或 `SASREC_MODEL_PATH`）
  - `LIGHTGCN_ML1M_MODEL_PATH`（或 `LIGHTGCN_MODEL_PATH`）
  - `MULTVAE_ML1M_MODEL_PATH`（或 `MULTVAE_MODEL_PATH`）
  - `POPREC_ML1M_MODEL_PATH`（或 `POPREC_MODEL_PATH`）
  - `BPRMF_ML1M_MODEL_PATH`（或 `BPRMF_MODEL_PATH`）
  - `GRU4REC_ML1M_MODEL_PATH`（或 `GRU4REC_MODEL_PATH`）
  - `BERT4REC_ML1M_MODEL_PATH`（或 `BERT4REC_MODEL_PATH`）
- Amazon 数据集使用对应前缀变量，例如：
  - `SASREC_AMAZON_ALL_BEAUTY_MODEL_PATH`
  - `LIGHTGCN_AMAZON_MAGAZINE_SUBSCRIPTIONS_MODEL_PATH`

## scripts 清单

- `scripts/run_service.sh`：前台启动 Flask 服务（默认端口 `19002`）
- `scripts/run_swipe_page.sh`：后台启动服务 + 健康检查 + 可选自动开浏览器
- `scripts/stop_swipe_page.sh`：停止指定端口服务
- `scripts/quit_service.sh`：`stop_swipe_page.sh` 别名
- `scripts/train_amazon_mm2018_all.sh`：Amazon_MM_2018 七模型通用训练
- `scripts/train_amazon_all_beauty_all.sh`：All_Beauty 七模型训练入口
- `scripts/train_amazon_magazine_subscriptions_all.sh`：Magazine_Subscriptions 七模型训练入口
- `scripts/train_amazon_mi_all.sh`：旧 Amazon Musical Instruments 三模型训练脚本
- `scripts/train_amazon_cds_all.sh`：旧 Amazon CDs 三模型训练脚本（找不到 CDs 时会 fallback 到 MI）
- `scripts/monitor_train_progress.sh`：每 5 分钟输出训练进度日志
- `scripts/commit.sh`：`git add -A` + `commit` + `pull --rebase` + `push`

项目根目录也提供兼容入口：

- `run_service.sh`（转发到 `scripts/run_service.sh`）

## docs 文档

`docs/` 下提供了 UML 与系统图文档，包括：

- `websim_system_uml.pdf`
- `software_engineering_diagrams.md`
- `UML_web_sim.pdf`

## 常见问题

### 1) `/health` 里某数据集 `available_models` 为空

说明对应权重不存在或路径不正确。请检查 `artifacts/*.pt` 或模型路径环境变量。

### 2) 数据集路径找不到

优先确认默认目录结构是否存在；如果路径不同，使用数据目录环境变量显式指定。

### 3) 推荐结果看起来像随机

当选定模型不可用，后端会回退到随机样本，前端状态中会体现当前可用模型情况。

## 备注

- `task.yaml` 保留给上层批量仿真/调度流程读取，Web 服务本身不依赖该文件启动。
- `artifacts/` 中已有多批次权重与日志，可按文件名中的后缀区分数据集与实验批次。
