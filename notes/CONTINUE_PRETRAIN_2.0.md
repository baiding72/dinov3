# DINOv3 nuImages 续训 2.0（OpenCode 实现规格）

> 目的：v1 的 probe 三指标全线下滑，怀疑两个因素——
> (a) `freeze_last_layer_epochs=5` 让前 5k 步 DINO/iBOT head 冻结；
> (b) split 泄漏（train+val+test 一起哈希 95:5，官方 val/test 混进 SSL 训练池）。
> v2 = **受控对照实验**：只改这两个变量，其余训练超参与 v1 完全一致，用来判断下滑到底由谁引起。
> 仓库：facebookresearch/dinov3。只改数据集层与配置，**不动训练主循环**。

## 0. 背景：v1 基线（对照基准，以下全部保持不动）

- batch：`batch_size_per_gpu: 64`
- LR：`schedules.lr` start=0, peak=5e-5, end=5e-5（batch 64 单卡时 `4×sqrt(64/1024)=1`，实际 lr 就是 5e-5）
- warmup：1 epoch = 1000 步（`OFFICIAL_EPOCH_LENGTH: 1000`）
- 总步数：10 epoch = 10k 步（`optim.epochs: 10`）
- 其余（teacher_temp、momentum、crops、各 loss 权重、clip_grad、weight_decay 等）一律照旧
- v1 的 split：官方 train+val+test 全部 CAM_FRONT 关键帧（约 18.4k 张）哈希 95:5 → 训练池 17.4k，泄漏
- v1 的 freeze：`freeze_last_layer_epochs: 5`（前 5k 步 head 最后一层 lr=0）

v2 只做两处实验变量改动：

1. `schedules.lr.freeze_last_layer_epochs: 5 → 0`；
2. split：只从官方 `v1.0-train` 内部哈希 95:5；官方 val/test 完全不参与 SSL。

## 1. Split 方案（严格官方定义 + train 内部 95:5）

```
官方 v1.0-train：67,279 个 sample → CAM_FRONT 关键帧 13,187 张（有标注）
  ├── SSL 训练集 95%   ≈ 12,528 张   ← 确定性哈希（CAM_FRONT sample token md5），无标签，仅训练
  ├── SSL 监控集 5%    ≈ 659 张      ← 确定性哈希，仅看 loss 趋势，不进训练
官方 v1.0-val：16,445 个 sample → CAM_FRONT 关键帧 3,249 张（有标注）
                    ← 全程不参与 SSL；probe 评测集
官方 v1.0-test：9,752 个 sample → CAM_FRONT 关键帧 1,932 张
                    ← 已核实 object/surface 标注为空数组；不参与 SSL 与有监督 probe
```

必须满足的断言（实现后运行，输出全 0 才通过）：

```
SSL 训练集 sample ∩ 官方 val sample = ∅
SSL 训练集 sample ∩ 官方 test sample = ∅
SSL 监控集 sample ∩ 官方 val sample = ∅
SSL 监控集 sample ∩ 官方 test sample = ∅
```

### 哈希划分实现要点（确定性、可复现）

- **先筛、再哈希**：先从 v1.0-train 筛出 `channel==CAM_FRONT && is_key_frame` 的关键帧（13,187 张，
  对应 13,187 个 sample token；`key_camera_token` 是 6 相机轮换的，**不能**拿全量 67,279 个 sample 当 CAM_FRONT 数）；
  再对这 13,187 个 sample token 计算 `int(md5(token).hexdigest()[:8], 16) % 100`
- `< 95` → SSL 训练集；`>= 95` → SSL 监控集
- seed/比例写进配置与文档，重跑结果一致

probe 评测集固定为官方 `v1.0-val`（与 PROBE_PLAN.md 一致）；train5 仅作 SSL loss 监控，不进 probe。

## 2. 配置改动（v1 → v2 的完整 diff）

```yaml
# 1) 实验变量：去掉 freeze（train.py 中 last_layer_lr 不再被清零，head 第 1 步就学）
schedules:
  lr:
    freeze_last_layer_epochs: 0

# 2) 实验变量：严格 split
train:
  dataset_path: NuImages:root=<nuimages_root>:split=TRAIN:subset=train95:camera=CAM_FRONT

# 3) 存档频率（artifact-only，不改变任何训练动力学；v1 只有 7999/8999/9999、probe 无中间点）
checkpointing:
  period: 1000
  max_to_keep: 10
evaluation:
  eval_period_iterations: 1000
```

说明：

- 除上述 3 处外，yaml 与 v1 **逐字段一致**：batch 64、lr 5e-5、warmup 1 epoch、epochs 10、crops、loss 权重全部不动；
- 存档改动只影响"留哪些中间点"：`output_dir/ckpt/<iter>/` 全量断点（resume 用，受 max_to_keep 裁剪，
  10k 共 10 个点不触发清理）；`output_dir/eval/training_<iter>/teacher_checkpoint.pth`（probe 用，不受裁剪）；
- 若要求"零改动洁癖"，存档两条也可以删掉恢复 v1 原值，但 probe 曲线会退化到 3 个点，不推荐。

## 3. 数据集类改动（dinov3/data/datasets/nuimages.py）

- 支持官方 split：`split=TRAIN/VAL/TEST` 映射 `v1.0-train/v1.0-val/v1.0-test`；
- 新增 `subset` 参数：`train95` / `train5` / `None`（None = 全量该 split）；
  `train95`/`train5` 按 §1 哈希规则在 v1.0-train 内部划分，**只对 v1.0-train 生效**；
- `split=VAL/TEST` 时 `subset` 必须为 None（val/test 不得再切）；
- `camera` 参数本期固定 `CAM_FRONT`；按 `key_camera_token → channel == "CAM_FRONT"` 过滤，
  其他视角/sweeps 不加载；
- 其余读取逻辑（RLE mask、bbox）保持不变；
- dataset 字符串示例：
  - SSL 训练：`NuImages:root=...:split=TRAIN:subset=train95:camera=CAM_FRONT`
  - SSL 监控（可选）：`NuImages:root=...:split=TRAIN:subset=train5:camera=CAM_FRONT`
  - Probe 训练：`NuImages:root=...:split=TRAIN:camera=CAM_FRONT`
  - Probe 评测：`NuImages:root=...:split=VAL:camera=CAM_FRONT`

## 4. 验证步骤（全部通过才算完成）

1. **split 断言**：§1 的四条交集检查输出全 0；
2. **规模统计**：SSL train95 ≈ 12,528 / train5 ≈ 659 / val 3,249 / test 1,932（无标注），与文档一致；
3. **smoke test**：全新 `--output-dir`，启动日志 `Starting training from iteration 0` +
   `Loading pretrained weights from <wrapped>`；跑 20 步确认：
   - lr 从 0 线性爬升（warmup 1000 步）；
   - `last_layer_lr` 与 lr 一致（freeze=0 生效）；
   - 显存峰值 < 120GB（batch 64）；
4. **checkpoint 频率**：跑过 1k 步后确认 `output_dir/ckpt/999/` 与
   `output_dir/eval/training_999/teacher_checkpoint.pth` 出现；跑完 10k 后 `ckpt/` 下
   999/1999/.../9999 共 10 个目录齐全；
5. **对照评测**：用修复后的同一 probe 管线，重测官方 / v1 / v2 的中间点曲线，同图画对比
   （v2 与 v1 的唯一差异就是 freeze 与 split）。

## 5. 禁止事项

- 不改 `train.py` / `ssl_meta_arch.py` / 训练主循环；
- **不引入 v1 之外的任何超参改动**：batch/lr/epochs/warmup/crops/损失权重一律不许动；
- 不把官方 val/test 的任何 sample 放入 SSL 训练或监控（断言保护）；
- 不下载模型权重（由用户提供路径）；
- 不使用 v1 的 output_dir（必须新目录，防止 resume 污染）；
- 本期不加 GRAM anchoring（那是后续实验的变量，不是 v2 的）。

## 6. 交付物

1. 数据集类改动（文件/函数/行清单）+ yaml diff（v1→v2 应正好是 §2 的 3 处）；
2. §4 验证输出（断言、规模、smoke 日志、checkpoint 列表）；
3. v2 中间点 probe 曲线（与官方、v1 同图对比）。
