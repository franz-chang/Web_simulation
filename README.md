# Web_sim 推荐系统仿真平台

基于 Flask 的推荐系统交互仿真平台，支持两种页面形态：
- 网格页（Net page）：4 卡片展示 + 点击/翻页
- 滑动页（Swipe page）：上下划交互 + 实时状态反馈

## 当前进度（2026-04-14）

- 所有 Shell 脚本已统一迁移到 `scripts/` 目录。
- 启停与训练脚本均已适配新目录结构（从 `scripts/` 调用也能正确定位项目根目录）。
- `artifacts/` 当前可用模型类型共 **7 种**：`sasrec`、`lightgcn`、`multvae`、`bert4rec`、`bprmf`、`gru4rec`、`poprec`。

## 目录结构（关键）

- `app.py`: Web 服务入口
- `artifacts/`: 模型权重与训练日志
- `templates/` + `static/`: 页面模板与静态资源
- `scripts/`: 运行、训练、维护脚本

## 0. 数据集准备

请在 `Web_sim` 同级目录准备 `WebSim_Dataset`，常用路径示例：

- `../WebSim_Dataset/MM-ML-1M-main`
- `../WebSim_Dataset/Amazon_MM_2018/All_Beauty`
- `../WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions`

数据与预训练模型参考（历史链接）：
- 数据集：https://drive.google.com/drive/folders/1GvtEZcsLhcl3e6as6JOW0wEdmqSZo4CA?usp=sharing
- 预训练模型：https://drive.google.com/drive/folders/1Av0ly-myQlt2e-II9j4B0x5PBFqsTGW3?usp=sharing

## 1. 训练模型

### 1.1 MovieLens-1M（手动）

```bash
cd /Users/chongzhang/Web_sim
python3 train_sasrec.py   --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/sasrec_ml1m.pt   --epochs 10 --eval-ks 10,20
python3 train_lightgcn.py --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/lightgcn_ml1m.pt --epochs 30 --eval-ks 10,20
python3 train_multvae.py  --dataset-dir /Users/chongzhang/WebSim_Dataset/MM-ML-1M-main --output-model artifacts/multvae_ml1m.pt  --epochs 30 --eval-ks 10,20
```

### 1.2 Amazon_MM_2018（一键脚本）

```bash
cd /Users/chongzhang/Web_sim
./scripts/train_amazon_all_beauty_all.sh /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/All_Beauty
./scripts/train_amazon_magazine_subscriptions_all.sh /Users/chongzhang/WebSim_Dataset/Amazon_MM_2018/Magazine_Subscriptions
```

说明：
- 两个脚本默认路径分别为 `Amazon_MM_2018/All_Beauty`、`Amazon_MM_2018/Magazine_Subscriptions`。
- 可直接调用通用脚本：`scripts/train_amazon_mm2018_all.sh <dataset_dir> <model_suffix>`。
- 输出模型位于 `artifacts/`。

## 2. 启动网站

### 2.1 直接启动（Net page）

```bash
cd /Users/chongzhang/Web_sim
PORT=19001 python3 app.py
```

访问：
- 网格页：`http://127.0.0.1:19001/`
- 健康检查：`http://127.0.0.1:19001/health`

### 2.2 Swipe page（推荐）

```bash
cd /Users/chongzhang/Web_sim
./scripts/run_swipe_page.sh
./scripts/stop_swipe_page.sh
```

默认地址：
- Swipe 页：`http://127.0.0.1:19002/swipe`
- 健康检查：`http://127.0.0.1:19002/health`

可选环境变量：

```bash
HOST=127.0.0.1 PORT=19002 LOG_FILE=web.log OPEN_BROWSER=1 ./scripts/run_swipe_page.sh
PORT=19002 ./scripts/stop_swipe_page.sh
```

说明：
- `OPEN_BROWSER=0` 可禁止自动打开浏览器。
- `LOG_FILE` 可用绝对路径；若是相对路径，按项目根目录解析。

## 3. scripts 脚本清单

- `scripts/run_service.sh`: 前台启动服务（常用于被上层脚本托管）
- `scripts/run_swipe_page.sh`: 后台启动 + 健康检查 + 可选自动打开页面
- `scripts/stop_swipe_page.sh`: 停止指定端口服务
- `scripts/quit.sh`: `stop_swipe_page.sh` 的别名入口
- `scripts/train_amazon_all_beauty_all.sh`: 一键训练 Amazon All Beauty 七模型
- `scripts/train_amazon_magazine_subscriptions_all.sh`: 一键训练 Amazon Magazine Subscriptions 七模型
- `scripts/train_amazon_mm2018_all.sh`: Amazon_MM_2018 通用七模型训练脚本
- `scripts/monitor_train_progress.sh`: 周期性记录训练进度到 `artifacts/train_progress_5min.log`
- `scripts/commit.sh`: 自动 `add/commit/pull --rebase/push`

## 4. 当前 artifacts 模型概览

当前目录 `artifacts/` 中按 `.pt` 统计到以下 7 类模型：

- `sasrec`
- `lightgcn`
- `multvae`
- `bert4rec`
- `bprmf`
- `gru4rec`
- `poprec`

如需复查，可运行：

```bash
python3 - <<'PY'
from pathlib import Path
p=Path('/Users/chongzhang/Web_sim/artifacts')
ext={'.pt','.pth','.ckpt','.bin'}
files=[f for f in p.iterdir() if f.is_file() and f.suffix.lower() in ext]
types=sorted({f.stem.split('_')[0].split('-')[0].lower() for f in files})
print('types:', types)
print('count:', len(types))
PY
```
