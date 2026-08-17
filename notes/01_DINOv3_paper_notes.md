# DINOv3 论文精读笔记（arXiv:2508.10104）

## 1. 论文定位

DINOv3 是 Meta AI 的第三代 DINO 自监督视觉基础模型。核心贡献三件事：

1. **规模化**：把预训练数据从 DINOv2 的 LVD-142M 提升到 LVD-1689M（从约 17B 张 Instagram 图片池中筛选），模型从 ViT-g/1.1B 提升到 ViT-7B/6.7B。
2. **Gram anchoring**：提出 Gram 矩阵锚定损失，解决"长时间训练下密集特征退化"这一已知未解问题。
3. **训练后处理**：高分辨率适应（resolution scaling）、从 7B 教师蒸馏模型家族、以及 LiT 式文本对齐（dino.txt）。

核心论断：**SSL 模型不需要针对任务微调**，冻结的 DINOv3 特征在分割、深度、检测、跟踪、检索等任务上全面超越弱监督（CLIP/SigLIP/PE）和聚合式（AM-RADIO/PEspatial）模型。

## 2. 训练阶段总览

```
阶段 1  预训练 LPre = L_DINO + L_iBOT + 0.1·L_KoLeo         （约 1M iterations）
阶段 2  Gram anchoring 精修 LRef/LHRef                      （1M 之后继续训练）
阶段 3  Post-training：
         - 高分辨率适应（混合分辨率，10k iterations，仍带 Gram）
         - 7B → ViT-S/S+/B/L/H+ 蒸馏（1M iterations + 250k cosine cooldown）
         - 文本对齐（冻结视觉，训练文本塔 + 2 层视觉 adapter）
```

## 3. 数据准备（3.1）

- 原始池：约 17B 张 Instagram 公开图片（已过平台内容审核）。
- 三条数据线组成 LVD-1689M：
  1. **聚类筛选**：DINOv2 特征 + 5 层层次化 k-means（200M / 8M / 800k / 100k / 25k 簇），平衡采样 → 1,689M 张（保证视觉概念覆盖均衡）。
  2. **检索筛选**：以精选种子数据集检索相似图（覆盖下游任务相关概念）。
  3. **原始公开集**：ImageNet-1k / ImageNet-22k / Mapillary Street Sequences（优化特定能力）。
- 采样策略：10% 的 batch 用 ImageNet-1k 纯 batch（同质高质数据），其余用混合 batch。
- 消融结论（Table 1）：纯聚类、纯检索、纯原始数据都不如混合管线；混合管线在 IN1k kNN/Linear、ObjectNet、iNaturalist、Paris Retrieval 上全面最好。

## 4. 预训练设置（3.2）

### 学习目标

LPre = L_DINO（全局 CLS 对比）+ L_iBOT（掩码 patch 重建）+ 0.1·L_KoLeo（批内特征均匀化正则）

- DINO 与 iBOT 均使用 **Sinkhorn-Knopp 教师软化**（替代 DINOv2 的 centering）。
- 全局/局部 crop 的 backbone 输出使用**独立的 LayerNorm**（`untie_global_and_local_cls_norm`），提升 kNN 稳定性与密集任务表现。
- KoLeo 采用分布式实现，loss 在 16 样本小批次内计算。

### 教师模型对比（Table 2）

| 配置 | DINOv2 | DINOv3 |
| --- | --- | --- |
| Backbone | ViT-giant (1.1B) | ViT-7B (6.7B) |
| Blocks | 40 | 40 |
| Patch size | 14 | **16**（同分辨率下序列长度一致） |
| 位置编码 | 可学习 | **RoPE（box jittering）** |
| Registers | 4 | 4 |
| Embed dim | 1536 | 4096 |
| FFN | SwiGLU (4096) | SwiGLU (8192) |
| Attention heads / head dim | 24 / 64 | 32 / 128 |
| DINO head | 4096-4096-256, 128k prototypes | 8192-8192-512, **256k prototypes** |
| iBOT head | 4096-4096-256, 128k prototypes | 8192-8192-384, **96k prototypes** |

### 优化设置

- **去掉所有参数调度**：常数 LR / weight decay / teacher EMA momentum，只保留 LR 与 teacher temperature 的线性 warmup（好处：可以无限续训，超参更少）。
- AdamW，总 batch 4096，256 张 GPU。
- Multi-crop：2 个 global crops（256×256）+ 8 个 local crops（112×112），每 batch 共 3.7M tokens。
- RoPE 坐标归一化到 [−1,1] box，并做 **box jittering**（随机缩放 s∈[0.5,2]），提升对分辨率/尺度/宽高比的鲁棒性。

## 5. Gram Anchoring（第 4 章）— 论文核心方法

### 动机

- 长训练下全局指标持续上升，但密集任务（VOC/ADE20k 分割）在约 200k iterations 后**退化**（ViT-7B 甚至跌到早期水平以下）。
- 原因：CLS token 与 patch token 的余弦相似度随训练升高 → patch 特征局部性变差、相似图变噪（Figure 5/6）。这不是 DINOv2 提出的高范数 patch 异常值（有 register 后范数稳定），而是**特征相关性结构退化**。

### 目标函数

对 L2 归一化的 patch 特征，匹配学生与"Gram teacher"的 Gram 矩阵（patch 两两点积矩阵）：

```
L_Gram = || X_S·X_S^T − X_G·X_G^T ||_F^2
```

- 只在 global crops 上计算；只约束相似性结构，不直接拉特征本身（特征可以自由移动）。
- **Gram teacher**：取早期（如 200k iterations）EMA 教师的快照，每 10k iterations 更新为当前 EMA 教师。
- 1M iterations 之后才开启（效率考虑），但**晚期应用仍能修复已退化的密集特征**。

精修阶段总损失：

```
L_Ref = w_D·L_DINO + L_iBOT + w_DK·L_KoLeo + w_Gram·L_Gram
```

### 高分辨率 Gram（LHRef）

- 给 Gram teacher 输入 2 倍分辨率图像（256→512），输出特征图 2× bicubic 下采样，得到更平滑、局部一致性更好的 Gram 目标。
- 收益：在 LRef 基础上 ADE20k 再 +2 mIoU。
- 消融（Figure 9）：Gram teacher 取 100k/200k 差别不大；取 1M 会伤害性能（因为此时教师密集特征已经退化）；×2 分辨率优于 ×1。

### 效果

- 加 Gram 后前 10k iterations 密集任务立刻回升；iBOT loss 下降更快（说明 Gram 与 iBOT 作用相似），DINO loss 基本不受影响。
- 全局指标（ObjectNet 等）继续缓慢上升，说明 Gram 不牺牲全局能力。

## 6. Post-training（第 5 章）

### 6.1 高分辨率适应

- 混合分辨率采样：global crops ∈ {512, 768}，local crops ∈ {112, 168, 224, 336}，训练 10k iterations。
- **必须带 Gram anchoring**（用 7B 教师作 Gram teacher），否则密集任务显著退化。
- 效果：分类小幅提升、OOD（ObjectNet）高分辨率下提升、ADE20k 分割与 DAVIS 跟踪随分辨率显著改善；模型推理可外推到 4K+ 分辨率。

### 6.2 蒸馏模型家族

- 直接用固定 7B 教师蒸馏 ViT-S/S+/B/L/H+（同一阶段 1 目标，无 Gram）。
- 多学生并行蒸馏：所有学生共享一次教师推理（all-gather），按学生规模分配 GPU 使各学生迭代时间一致。
- 训练：1M iterations + 250k cosine LR cooldown，再高分辨率适应（不带 Gram）。
- 关键结论：ViT-H+（0.8B）性能逼近 7B 教师（Figure 16b）。

### 6.3 文本对齐（dino.txt）

- LiT 范式：冻结视觉骨干，训练文本编码器 + 2 层视觉 transformer。
- 关键技巧：**mean-pooled patch embeddings 与 CLS token 拼接**后再与文本匹配 → 全局+局部对齐，提升密集任务表现。

## 7. 评估协议与指标（第 6-7 章）

### 统一评测协议

- 所有模型输入分辨率适配为 1024 patch tokens（patch16 → 512×512；patch14 → 448×448），保证公平。
- 默认使用**冻结 backbone** + 轻量探测。

| 任务 | 数据集 | 指标 | 结果（DINOv3 7B） |
| --- | --- | --- | --- |
| 线性分割 | ADE20k / Cityscapes / VOC | mIoU | **55.9 / 81.1 / 86.6**（SOTA 对比全部第一） |
| 线性深度 | NYUv2 / KITTI | RMSE ↓ | **0.309 / 2.346**（超 DINOv2 0.278） |
| 3D 对应 | NAVI（几何）/ SPair（语义） | correspondence recall | **64.4 / 58.7** |
| 无监督物体发现 | VOC07/VOC12/COCO-20k + TokenCut | CorLoc | **66.1 / 69.5 / 55.1** |
| 视频分割跟踪 | DAVIS/YTVOS/MOSE | J&F | DAVIS-L **83.3**（+6.7 vs DINOv2） |
| 视频分类 | UCF101/SSv2/K400 | top-1（attentive probe） | 93.5 / 70.1 / 87.8 |
| 线性分类 | IN1k（域泛化：R/S/A/C/ObjectNet） | top-1 | 88.4，OOD 全面接近弱监督 |
| 细粒度分类 | Places205 / iNat18 / iNat21 / Fine-S | acc | iNat21 **89.8**（超 PEcore） |
| 实例检索 | Oxford-H / Paris-H / Met / AmsterTime | mAP / GAP | Met +10.8、AmsterTime +7.6 vs DINOv2 |

### 复杂系统（微调/加解码器）

| 任务 | 方案 | 指标 | 结果 |
| --- | --- | --- | --- |
| 检测 | 冻结 DINOv3 + Plain-DETR（100M 可训） | COCO mAP / COCO-O | **65.6 / 66.4**，首个"冻结骨干"达到 SOTA |
| 分割 | 冻结 + ViT-Adapter + Mask2Former（927M） | ADE20k mIoU | 62.6（TTA 63.0，追平 ONE-PEACE） |
| 深度 | 冻结 + Depth Anything v2 管线（DPT） | ARel ↓ / δ1 | NYU 4.3/98.0，全部数据集新 SOTA |
| 3D | VGGT 换用 DINOv3 ViT-L | Re10K/CO3Dv2 AUC@30 | 86.3 / 89.6（超 VGGT） |

### 模型家族（Table 14，代表性指标）

| 模型 | Params | IN-ReaL | IN-R | ObjectNet | ADE20k | NYU ↓ | DAVIS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S | 21M | 87.0 | 60.4 | 50.9 | 47.0 | 0.403 | 72.7 |
| ViT-S+ | 29M | 88.0 | 68.8 | 54.6 | 48.8 | 0.399 | 75.5 |
| ViT-B | 86M | 89.3 | 76.7 | 64.1 | 51.8 | 0.373 | 77.2 |
| ViT-L | 300M | 90.2 | 88.1 | 74.8 | 54.9 | 0.352 | 79.9 |
| ViT-H+ | 840M | 90.3 | 90.0 | 78.6 | 54.8 | 0.352 | 79.3 |

## 8. 论文 → 本地代码映射（仓库 /Users/baiding/Desktop/dinov3）

### 核心训练代码

| 论文概念 | 代码位置 | 说明 |
| --- | --- | --- |
| 训练入口 | `dinov3/train/train.py` | argparse + 配置驱动；`do_train` / `forward_backward` 循环 |
| 自监督元架构 | `dinov3/train/ssl_meta_arch.py` | `SSLMetaArch`：teacher/student 前向、`compute_losses`（DINO local/global + KoLeo + iBOT + Gram）、EMA 更新、Sinkhorn-Knopp |
| 多学生蒸馏 | `dinov3/train/multidist_meta_arch.py` | `MultiDistillationMetaArch`，对应 5.2 节多学生共享教师推理 |
| Gram 损失 | `dinov3/loss/gram_loss.py` | 即 L_Gram；MSE of Gram matrices；可选去除负相似度 |
| DINO 损失 | `dinov3/loss/dino_clstoken_loss.py` | 全局 CLS 对比 + Sinkhorn-Knopp 教师 |
| iBOT 损失 | `dinov3/loss/ibot_patch_loss.py` | 掩码 patch 对比 |
| KoLeo | `dinov3/loss/koleo_loss.py` | 批内均匀化正则（分布式版 `KoLeoLossDistributed`） |
| 骨干网络 | `dinov3/models/vision_transformer.py` | `DinoVisionTransformer`：patch embed、CLS + storage/register tokens、RoPE、`forward_features_list` |
| 位置编码 | `dinov3/layers/rope_position_encoding.py` | `RopePositionEmbedding`（box 归一化、jitter 等参数） |
| 注意力块 | `dinov3/layers/block.py` | `SelfAttentionBlock` / `CausalSelfAttentionBlock`，SwiGLU FFN 选项 |
| 数据增强/多 crop | `dinov3/data/transforms.py` + `augmentations.py` | DINO 风格 multi-crop |
| 掩码生成 | `dinov3/data/masking.py` | iBOT 掩码（比例 0.1–0.5） |
| 采样/数据混合 | `dinov3/data/samplers.py` + `loaders.py` | infinite sampler、CombinedDataLoader |
| FSDP/并行 | `dinov3/fsdp/`、`dinov3/distributed/` | 大规模分布式训练 |
| 提交/启动 | `dinov3/run/submit.py` | SLURM/submitit 封装 |

### 关键配置

| 配置 | 对应论文 |
| --- | --- |
| `configs/train/dinov3_vit7b16_pretrain.yaml` | 阶段 1：LVD 预训练 |
| `configs/train/dinov3_vit7b16_gram_anchor.yaml` | 阶段 2：Gram（`use_loss: true`、`update_frequency: 10000`、`it_first_update: 1010000`、`gram_teacher_crops_size: 512`） |
| `configs/train/dinov3_vit7b16_high_res_adapt.yaml` | 阶段 3a：高分辨率适应 |
| `configs/train/dinov3_vitl16_lvd1689m_distilled.yaml` | 5.2：ViT-L 蒸馏配方（`META_ARCHITECTURE: MultiDistillationMetaArch`） |
| `configs/train/vitl_im1k_lin834.yaml` | README 快速上手：ViT-L 在 ImageNet-22k 上 4 节点 32 GPU 跑 14h 达 kNN 82.0 / linear 83.5 |

### 启动命令（来自 README）

```bash
# 预训练（32 节点 / 256 GPU，SLURM）
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/train/train.py \
  --nodes 32 \
  --config-file dinov3/configs/train/dinov3_vit7b16_pretrain.yaml \
  --output-dir <OUTPUT_DIR> \
  train.dataset_path=<DATASET>:root=<DATA>:extra=<DATA>

# Gram anchoring（续训，需指定 gram.ckpt）
... --config-file dinov3/configs/train/dinov3_vit7b16_gram_anchor.yaml \
  gram.ckpt=<PATH/TO/GRAM_TEACHER>

# 高分辨率适应
... --config-file dinov3/configs/train/dinov3_vit7b16_high_res_adapt.yaml \
  gram.ckpt=<GRAM_TEACHER> student.resume_from_teacher_chkpt=<TEACHER>

# 多学生蒸馏
... --config-file dinov3/configs/train/multi_distillation_test.yaml --multi-distillation

# 评测（线性分类）
PYTHONPATH=${PWD} python -m dinov3.run.submit dinov3/eval/log_regression.py \
  model.config_file=<OUTPUT>/config.yaml model.pretrained_weights=<OUTPUT>/teacher_checkpoint.pth ...
```

### 值得精读的代码路径（建议顺序）

1. `dinov3/train/train.py::do_train` → 训练主循环、checkpoint 恢复、scheduler。
2. `dinov3/train/ssl_meta_arch.py::forward_backward` + `compute_losses` → 一行行对公式 LPre/LRef。
3. `dinov3/train/ssl_meta_arch.py::get_teacher_output` / `get_student_output` → 理解 teacher/student 的 crop 与 mask 流。
4. `dinov3/loss/gram_loss.py` → Gram 损失实现（注意 `remove_neg` 等变体）。
5. `dinov3/models/vision_transformer.py::prepare_tokens_with_masks` / `forward_features_list` → RoPE + register token 的序列组装。
6. `dinov3/data/transforms.py` + `configs` 里的 crop 尺寸 → 多 crop 策略。

## 9. 对"SSL 替换伪标签"的启发（个人理解）

- DINOv3 证明**冻结的 SSL 密集特征**可以作为下游系统的"免费监督信号"：线性分割/深度已接近甚至超过专门的监督模型，检测用 100M 冻结骨干解码器就到 SOTA。
- Gram anchoring 提供了一种**保持特征局部一致性**的正则手段——如果未来在驾驶数据上继续自监督训练，这个机制可以防止长训练导致密集特征退化。
- 高分辨率适应（RoPE 天然支持任意分辨率）对自动驾驶多相机/不同分辨率输入很友好。
- 但注意：DINOv3 是在**自然图像**（Instagram 为主）上预训练的；驾驶场景属于域偏移（虽然 Cityscapes 线性结果已证明较强的泛化），在 nuScenes 等驾驶数据上做适配/续训是自然下一步。

## 10. 模型超参速查（DINOv2 vs DINOv3，2026-08 从官方代码核对）

| 项 | DINOv2 ViT-L/14 | DINOv2 ViT-g/14 | DINOv3 ViT-L/16（蒸馏） | DINOv3 ViT-7B/16 |
| --- | --- | --- | --- | --- |
| embedding 维度 | 1024 | 1536 | 1024 | 4096 |
| 层数 | 24 | 40 | 24 | 40 |
| 注意力头数（每头维） | 16×64 | 24×64 | 16×64 | 32×128 |
| FFN 类型 | SwiGLU（swiglufused） | SwiGLU | MLP | SwiGLU（swiglu64） |
| FFN 实际宽度 | 2736 | 4096 | 4096 | 8192 |
| FFN 激活 | SiLU 门控 | SiLU 门控 | GELU | SiLU 门控 |
| 位置编码 | 可学习（可插值） | 可学习 | RoPE | RoPE |
| patch / registers | 14 / 0（reg 版 4） | 14 / 0 | 16 / 4 | 16 / 4 |
| LayerScale 初值 | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| qkv bias | true | true | true | false |

要点：DINOv2 发布版 ViT-L/g 用 SwiGLU（官方 vitl14/vitg14.yaml 的 `swiglufused`），宽度 = 2/3×名义 hidden 再对齐（L: 2736，g: 4096）；普通 MLP 用 GELU。DINOv3 7B 用 swiglu64（4096×3 的 2/3 = 8192）。位置编码是 v2→v3 最大架构变化：可学习绝对位置 → 2D RoPE（base=100、坐标归一化 [-1,1]、rescale_coords=2 即训练时 box jittering），支持任意分辨率推理。配置中 `rope`/`ropenew` 仅为历史标签，实际行为由 `pos_embed_rope_*` 决定。

代码依据：DINOv3 本地 `dinov3/models/vision_transformer.py`、`dinov3/layers/ffn_layers.py`、`dinov3/layers/rope_position_encoding.py` 与两个训练配置；DINOv2 官方仓库 `dinov2/models/vision_transformer.py`、`dinov2/configs/train/{vitl14,vitg14}.yaml`、`dinov2/layers/{swiglu_ffn,layer_scale}.py`。
