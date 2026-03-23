# Recommender System simulation experiment platform

## 0. 数据集准备

请新建一个和 Web_simulation平级的目录： WebSim_Dataset. 所需的数据集在：https://drive.google.com/drive/folders/1GvtEZcsLhcl3e6as6JOW0wEdmqSZo4CA?usp=sharing.

预训练好的SASRec, Multi-VAE, LightGCN在：

## 1. 训练模型（MovieLens-1M）

```bash
cd /Web_sim
python3 train_sasrec.py --dataset-dir /WebSim_Dataset/MM-ML-1M-main --output-model artifacts/sasrec_ml1m.pt --epochs 10 --eval-ks 10,20
python3 train_lightgcn.py --dataset-dir /WebSim_Dataset/MM-ML-1M-main --output-model artifacts/lightgcn_ml1m.pt --epochs 30 --eval-ks 10,20
python3 train_multvae.py --dataset-dir /WebSim_Dataset/MM-ML-1M-main --output-model artifacts/multvae_ml1m.pt --epochs 30 --eval-ks 10,20
```

训练时会在每个 epoch 输出验证集指标（`HR@K`、`NDCG@K`），并按 `HR@10` 自动保存最佳 checkpoint。

## 2. 训练模型（Amazon Musical Instruments）

```bash
cd /Web_sim
python3 train_sasrec.py --dataset-dir /WebSim_Dataset/amazon_v2/Musical_Instruments --output-model artifacts/sasrec_amazon_mi.pt --epochs 10 --eval-ks 10,20
python3 train_lightgcn.py --dataset-dir /WebSim_Dataset/amazon_v2/Musical_Instruments --output-model artifacts/lightgcn_amazon_mi.pt --epochs 30 --eval-ks 10,20
python3 train_multvae.py --dataset-dir /WebSim_Dataset/amazon_v2/Musical_Instruments --output-model artifacts/multvae_amazon_mi.pt --epochs 30 --eval-ks 10,20
```

或使用一键脚本：

```bash
cd /Web_sim
./train_amazon_mi_all.sh /WebSim_Dataset/amazon_v2/Musical_Instruments
```

## 3. 启动网站

```bash
cd /Web_sim
PORT=19001 python3 app.py
```

打开 `http://127.0.0.1:19001` 即可访问。

## 功能逻辑

- 初始页：随机 4 个条目（1x4，一行四列）
- 可在下拉菜单中切换数据集：`MovieLens-1M`、`Amazon Musical Instruments`
- 可在下拉菜单中选择推理模型：`SASRec`、`LightGCN`、`Mult-VAE`
- 点击某个条目：将点击历史输入当前模型，返回推荐 1-4 位
- 点击“下一页”：返回 5-8 位、9-12 位...
