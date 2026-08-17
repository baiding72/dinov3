# 研究路线图与下一步行动清单

## 1. 总目标回顾

> 用 DINOv3 自监督表征替换感知任务的伪标签，实现"自监督数据上量"的端到端自动驾驶模型。

即：**把对人工标注（3D 框/分割/深度）的依赖，替换为对自监督特征（DINOv3）的依赖**，从而让模型能从海量未标注驾驶视频中持续获益。

## 2. 阶段拆解（当前在阶段 1-2）

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | DINOv3 论文 + 代码精读 | 进行中（本笔记完成主干，建议按 §9 代码路径再走一遍） |
| 2 | 端到端范式 + nuScenes 框架 | 进行中（本笔记完成概念层；下一步动手配置环境） |
| 3 | 跑通基线：DINOv3 推理 + 一个 E2E 基线（如 UniAD 或 VAD） | 未开始 |
| 4 | 设计并验证"SSL 特征替代伪标签"实验 | 未开始 |
| 5 | 数据上量：引入未标注驾驶数据 + 蒸馏/一致性训练 | 未开始 |

## 3. 下一步行动清单（按顺序）

### 近期（1-2 周）

- [ ] 按 `01_DINOv3_paper_notes.md` §8 的代码路径把 DINOv3 训练代码过一遍（建议用 IDE 断点走 `forward_backward`）。
- [ ] 装好 conda + Python 3.11 环境，`pip install -e .`（仓库根目录，见 `conda.yaml`）；先跑通 `torch.hub.load("facebookresearch/dinov3", ...)` 加载预训练 backbone 的推理 demo。
- [ ] 下载 nuScenes **mini 版**（约 4GB），跑通 `03_nuScenes_setup_notes.md` §5 的最小示例。
- [ ] 精读 UniAD 代码（github.com/OpenDriveLab/UniAD），确认其数据管线与规划指标实现。

### 中期（3-6 周）

- [ ] 复现一个轻量 E2E 基线（建议从 VAD / UniAD 的开源实现起步），拿到 nuScenes 开环 avg.L2 / avg.Col 数字。
- [ ] 设计第一版实验：把基线感知监督（分割/深度/框）替换为 DINOv3 特征蒸馏损失（feature distillation），对比指标变化。
- [ ] 引入 Gram 式正则（或直接复用 DINOv3 Gram loss）防止长训练下密集特征退化。

### 远期

- [ ] 数据上量实验：收集/使用大规模未标注驾驶视频，用 DINOv3 冻结特征做自训练/伪标签，验证 scaling 曲线。
- [ ] 补闭环评测（CARLA Leaderboard / nuPlan / Bench2Drive），论证开环提升不是过拟合。

## 4. 候选实验设计（供讨论）

**实验 A — 特征蒸馏替代伪标签**：BEV 编码器（如 BEVFormer 主干）除规划损失外，增加 DINOv3 冻结特征的蒸馏项（如 patch/BEV 特征对齐或 Gram 对齐），观察 nuScenes 规划指标与标注需求量的关系。

**实验 B — 自训练（self-training）上量**：标注子集上训练教师 → 对未标注帧用 DINOv3 特征一致性过滤伪标签 → 学生迭代。DINOv3 的线性可分性可作为伪标签置信度信号。

**实验 C — 长训练稳定性**：在驾驶数据上继续 SSL 训练（DINOv3 目标 + Gram），观察密集特征退化问题是否在驾驶域复现，验证 Gram anchoring 的迁移性。

## 5. 关键文献清单（含后续可读）

- DINOv3（2508.10104）；DINOv2（2304.07193）；DINO（2104.14294）
- E2E 综述：2306.16927、2307.04370
- UniAD（2212.10156）；VAD（2303.12077）；ST-P3（2207.07601）；BEVFormer（2203.17270）
- 策略蒸馏：Learning by Cheating（1912.12294）、Roach（2108.08265）
- 自监督驾驶预训练：PPGeo（2301.01006）
- nuScenes（1903.11027）；nuPlan（2106.11810）

> 注：后续可补充 2024-2026 的 VLA（视觉-语言-动作）路线（如 DriveVLM、AD-MLLM）与 DiffE2E 等扩散规划工作，但按当前阶段优先级放在主线之后。
