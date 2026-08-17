# DINOv3 × 端到端自动驾驶 — 学习笔记索引

> 总目标：以 DINOv3 自监督表征替换感知任务的伪标签，实现自监督数据上量的端到端（E2E）自动驾驶模型。
> 当前阶段：① 精读 DINOv3 论文与核心代码；② 建立端到端自动驾驶范式认知；③ 熟悉并配置 nuScenes 框架。

## 论文资料（已下载，位于 `../papers/`）

| 文件 | 内容 | arXiv |
| --- | --- | --- |
| `dino1_2104.14294.pdf` | DINO v1：Emerging Properties in Self-Supervised Vision Transformers | 2104.14294 |
| `dinov2_2304.07193.pdf` | DINOv2：Learning Robust Visual Features without Supervision | 2304.07193 |
| `dinov3_2508.10104.pdf` | DINOv3 技术报告（Meta AI，2025） | 2508.10104 |
| `e2e_ad_challenges_frontiers_2306.16927.pdf` | 端到端自动驾驶综述（OpenDriveLab，270+ 篇） | 2306.16927 |
| `e2e_ad_recent_advancements_survey_2307.04370.pdf` | E2E 自动驾驶深度学习综述（IIT Roorkee） | 2307.04370 |
| `nuscenes_1903.11027.pdf` | nuScenes 多模态数据集论文（CVPR 2020） | 1903.11027 |
| `uniad_planning_oriented_2212.10156.pdf` | UniAD：Planning-oriented Autonomous Driving（CVPR 2023） | 2212.10156 |

每份 PDF 旁都有同名 `.txt`（pdftotext -layout 提取），方便检索。

## 笔记清单

1. [01_DINOv3_paper_notes.md](./01_DINOv3_paper_notes.md) — DINOv3 论文精读：网络结构、损失设计、训练阶段、实验指标，以及论文 → 本地代码的映射。
2. [02_E2E_AD_survey_notes.md](./02_E2E_AD_survey_notes.md) — 端到端自动驾驶范式：定义、方法分类、评测体系、UniAD 案例、挑战与趋势。
3. [03_nuScenes_setup_notes.md](./03_nuScenes_setup_notes.md) — nuScenes 框架：数据集构成、数据格式、评测指标、devkit 配置步骤。
4. [05_nuscenes_family_and_mini.md](./05_nuscenes_family_and_mini.md) — nuScenes 家族子数据集（nuScenes/nuImages/nuPlan/lidarseg/panoptic/CanBus/地图）对比 + mini 版下载方式与实测数据结构。
5. [04_research_action_plan.md](./04_research_action_plan.md) — 结合总目标的研究路线图与下一步行动清单。

## 核心结论速览

- **DINOv3 = 数据上量 + 模型上量 + 解决"长训练密集特征退化"**。训练分两阶段：预训练（DINO+iBOT+KoLeo）→ Gram anchoring 精修（LRef / LHRef），再加高分辨率适应、7B 教师蒸馏、LiT 文本对齐三个 post-training 步骤。
- **对自动驾驶最有价值的是其密集特征**：冻结 backbone 线性探测即可在 ADE20k（55.9 mIoU）、Cityscapes（81.1 mIoU）、NYUv2（0.309 RMSE）、KITTI（2.346 RMSE）上超越全部对比方法；且支持任意分辨率推理（最高 4K+）。
- **端到端范式**：把感知/预测/规划做成一个可微系统，以规划为最终目标联合优化；主流学习方式是模仿学习（BC/IRL）与策略蒸馏，闭环评测（CARLA/nuPlan）与开环评测（nuScenes 轨迹指标）并重。
- **"用 SSL 表征替代伪标签"在综述里有明确先例**（PPGeo、ViDAR 等自监督预训练；以及 feature distillation 式的策略蒸馏），DINOv3 是这一方向当前最强的现成骨干，且其 Gram 机制对"训练中特征退化"有直接借鉴意义。
- **nuScenes 是开环 E2E 的标准数据集**：UniAD 等模型以 avg.L2（轨迹误差）和 avg.Col（碰撞率）为规划指标；devkit 已在本机 conda 环境 `nuscenes`（py3.12）配置完成，nuScenes/nuImages mini 已下载解压。

## 本机数据与工具（2026-08-09）

- 数据：`~/nuscenes/`（nuScenes mini 9GB + lidarseg）、`~/nuimages/`（nuImages mini）、`~/nuplan/`（mini splits/地图下载中）。
- 工具：`tools/nuscenes-download/`（li-xl 脚本 + 我们扩展的 `get_archive_urls.py`）、`tools/nuscenes-devkit/`（官方 devkit 源码，python-sdk 需加 PYTHONPATH）。
