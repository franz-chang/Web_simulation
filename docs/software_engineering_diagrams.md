# Web_sim 软件工程图谱

下面这套图是按 `/Users/chongzhang/Web_sim` 当前代码结构绘制的，覆盖了你这个项目最常用的几类软件工程图。

## 1) 系统架构图（Component / Container）

```mermaid
flowchart LR
  User[用户 / Browser]

  subgraph Frontend[前端层]
    Grid[templates/index.html + static/app.js]
    Swipe[templates/swipe.html + static/swipe.js]
  end

  subgraph Backend[后端层 Flask app.py]
    API[/api/init /api/select /api/next /health /poster]
    Session[Flask Session]
    RecMgr[get_recommender()]
  end

  subgraph RecLayer[推荐层 recommender.py]
    MR[MovieRecommender]
    Catalog[MovieCatalog]
    Engines[SASRec/LightGCN/MultVAE/PopRec/BPRMF/GRU4Rec/BERT4Rec Engines]
  end

  subgraph DataAssets[数据与模型]
    Datasets[MovieLens-1M / Amazon MI]
    Artifacts[artifacts/*.pt]
    Posters[posters/ images/]
  end

  User --> Grid
  User --> Swipe
  Grid --> API
  Swipe --> API
  API --> Session
  API --> RecMgr
  RecMgr --> MR
  MR --> Engines
  MR --> Catalog
  Engines --> Artifacts
  Catalog --> Datasets
  Catalog --> Posters
  API --> Catalog
```

## 2) 模块依赖图（Module Dependency）

```mermaid
flowchart TD
  app[app.py]
  rec[recommender.py]
  du[dataset_utils.py]

  sas[sasrec.py]
  lgc[lightgcn.py]
  mvae[multvae.py]
  pop[poprec.py]
  bpr[bprmf.py]
  gru[gru4rec.py]
  bert[bert4rec.py]

  tsas[train_sasrec.py]
  tlgc[train_lightgcn.py]
  tmv[train_multvae.py]
  tpop[train_poprec.py]
  tbpr[train_bprmf.py]
  tgru[train_gru4rec.py]
  tbert[train_bert4rec.py]

  front[templates/*.html + static/*.js]

  app --> rec
  app --> front

  rec --> sas
  rec --> lgc
  rec --> mvae
  rec --> pop
  rec --> bpr
  rec --> gru
  rec --> bert

  tsas --> du
  tsas --> sas

  tlgc --> du
  tlgc --> lgc

  tmv --> du
  tmv --> mvae

  tpop --> du
  tpop --> pop

  tbpr --> du
  tbpr --> bpr

  tgru --> du
  tgru --> gru

  tbert --> du
  tbert --> bert
```

## 3) 核心接口时序图（Sequence: `/api/select`）

```mermaid
sequenceDiagram
    actor U as 用户
    participant FE as 前端(static/*.js)
    participant API as Flask /api/select
    participant S as Session
    participant R as MovieRecommender
    participant E as 选定Engine
    participant C as MovieCatalog

    U->>FE: 点击卡片/下滑触发选择
    FE->>API: POST(movie_id, model_name, dataset_key)
    API->>S: 读取history/model/dataset
    API->>R: recommend_ids(history + movie_id, model, topk=200)

    alt 模型可用且返回非空
        R->>E: recommend(history, topk)
        E-->>R: rec_ids
    else 模型不可用或返回空
        R-->>R: fallback random_movie_ids()
    end

    R->>C: has_movie() + movie_card()
    R-->>API: 去重后的推荐ID列表
    API->>S: 写回history/rec_ids/page_idx
    API-->>FE: cards/page/has_next/available_models
    FE-->>U: 渲染推荐结果
```

## 4) 训练流程活动图（Activity）

```mermaid
flowchart TD
  start([开始]) --> load[加载序列: load_user_sequences]
  load --> remap[重映射item + 划分train/valid/test]
  remap --> build[构建训练数据集或采样器]
  build --> loop{epoch <= epochs?}

  loop -->|是| train[训练1个epoch]
  train --> valid[验证集评估 HR@10/20, NDCG@10/20]
  valid --> best{HR@10提升?}
  best -->|是| keep[保存best_state_dict]
  best -->|否| cont[继续下一轮]
  keep --> cont
  cont --> loop

  loop -->|否| loadbest[加载最佳checkpoint]
  loadbest --> test[测试集评估]
  test --> save[保存artifact到artifacts/*.pt]
  save --> end([结束])
```

## 5) 会话状态图（State Machine）

```mermaid
stateDiagram-v2
    [*] --> Uninitialized

    Uninitialized --> RandomInitialized: GET /api/init
    RandomInitialized --> Recommending: POST /api/select

    Recommending --> Recommending: POST /api/select\n(追加history)
    Recommending --> Paging: POST /api/next\n(有下一页)

    Paging --> Paging: POST /api/next\n(继续翻页)
    Paging --> Exhausted: POST /api/next\n(没有更多)

    Exhausted --> RandomInitialized: GET /api/init\n(随机重置)
    Recommending --> RandomInitialized: 切换数据集/模型后 init
    Paging --> RandomInitialized: 切换数据集/模型后 init
```

## 6) 部署运行图（Deployment / Runtime）

```mermaid
flowchart LR
  Browser[浏览器\n127.0.0.1:19001/19002]

  subgraph Host[本机 /Users/chongzhang/Web_sim]
    Script[scripts/run_swipe_page.sh / scripts/stop_swipe_page.sh]
    FlaskProc[python3 app.py]
    Templates[templates/* + static/*]
    Artifacts[artifacts/*.pt]
    Dataset[../WebSim_Dataset/*]
    Log[web.log]
  end

  Browser --> FlaskProc
  Script --> FlaskProc
  FlaskProc --> Templates
  FlaskProc --> Artifacts
  FlaskProc --> Dataset
  FlaskProc --> Log
```

## 可继续扩展

如果你后面要写论文或技术报告，我可以在这套图基础上再补：
- UML 类图（精确到 `MovieRecommender` 与各 Engine 的属性/方法）
- 数据流图（DFD）
- C4 Model 的 Level 1/2/3 全套版本
