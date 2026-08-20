# DINOv3 nuImages 续训 2.0（OpenCode 实现规格）

> v2 = v1 的受控对照，只改两处实验变量：**去掉 freeze** + **split 严格化（防泄漏）**，
> 其余超参与 v1 完全一致。目的：判断 v1 的 probe 下滑由哪个因素引起
> （修好 bbox 口径后已确认：仅 drivable IoU 明显退化 0.97→0.78；
> obj top-1 与 CLS mAP 基本持平 ~0.69/~0.74，不作为 v2 的主判据）。
> 仓库：facebookresearch/dinov3。只改数据集层与配置，**不动训练主循环**。

## 0. v1 基线（除两处改动外全部保持不动）

- batch 64、lr peak 5e-5（batch 64 单卡实际 lr=5e-5）、warmup 1 epoch = 1000 步、
  总 10k 步、`OFFICIAL_EPOCH_LENGTH: 1000`
- 其余（teacher_temp、momentum、crops、loss 权重、clip_grad、weight_decay 等）一律照旧
- v1 的问题：`freeze_last_layer_epochs: 5`（前 5k 步 head 最后一层 lr=0）；
  split 泄漏（train+val+test 一起哈希 95:5，训练池≈17.4k 张，含官方 val/test）

## 1. 两处改动（v1 → v2 完整 diff）

```yaml
# 1) 去掉 freeze：head 最后一层从第 1 步就学
schedules:
  lr:
    freeze_last_layer_epochs: 0

# 2) 严格 split：只用官方 v1.0-train 内部 95:5，val/test 不碰
train:
  dataset_path: NuImages:root=<root>:split=TRAIN:subset=train95:camera=CAM_FRONT

# 3) 存档频率（不影响训练动力学，仅让 probe 有中间点；可还原，但曲线只剩 3 个点）
checkpointing:
  period: 1000
  max_to_keep: 10
evaluation:
  eval_period_iterations: 1000
```

## 2. Split 细节（防泄漏核心）

```
v1.0-train：67,279 个 sample → CAM_FRONT 关键帧 13,187 张
  ├─ SSL 训练集 95% ≈ 12,528 张   ← 哈希 sample token md5 % 100 < 95
  └─ SSL 监控集 5%  ≈ 659 张      ← 不进训练，仅看 loss 趋势
v1.0-val：CAM_FRONT 3,249 张       ← 全程不碰 SSL；probe 评测集
v1.0-test：CAM_FRONT 1,932 张      ← 已核实无标注；不碰
```

断言（实现后运行，输出全 0 才通过）：SSL 训练集/监控集 ∩ 官方 val/test sample = ∅（4 条）。

实现要点：

- **先筛、再哈希**：先筛 `channel==CAM_FRONT && is_key_frame`（13,187 张），再对这 13,187 个
  sample token 做确定性哈希 `int(md5(token).hexdigest()[:8], 16) % 100`；`key_camera_token`
  是 6 相机轮换的，**不能**拿全量 67,279 个 sample 当 CAM_FRONT 数；
- seed/比例写进配置，重跑结果一致。

## 3. 数据集类改动（dinov3/data/datasets/nuimages.py）

- 支持 `split=TRAIN/VAL/TEST`；新增 `subset=train95/train5/None`（只对 TRAIN 生效；
  VAL/TEST 必须 None）；
- `camera` 固定 `CAM_FRONT`（按 calibrated_sensor→sensor 精确解析 channel；sweeps/其他视角不加载）；
- 其余读取逻辑（RLE mask、bbox）保持不变。

## 4. 验证

1. split 断言全 0 + 规模统计（12,528 / 659 / 3,249 / 1,932）；
2. smoke：全新 output_dir、`Starting training from iteration 0`、lr 线性爬升、
   `last_layer_lr == lr`（freeze=0 生效）；
3. ckpt：1k 步后 `ckpt/999/` 与 `eval/training_999/teacher_checkpoint.pth` 出现，10k 后 10 个目录齐全；
4. 对照：同 probe 管线重测官方 / v1 / v2 中间点，同图对比。

## 5. 禁止事项

- 不改 `train.py` / `ssl_meta_arch.py`；不引入 v1 之外的任何超参改动；
- val/test 的 sample 不进 SSL（断言保护）；不用 v1 的 output_dir（防 resume 污染）；
- 本期不加 GRAM anchoring（后续实验：v2+GRAM，官方 checkpoint 作冻结 Gram teacher，
  `gram.use_loss=true, ema_teacher=false, rep_update=false, it_load_ema_teacher=-1` +
  `crops.gram_teacher_crops_size=256`；等 v2 裸版结果出来再开）。

## 6. 交付物

1. 数据集类改动（文件/函数/行清单）+ yaml diff（应恰好是 §1 的 3 处）；
2. §4 验证输出；
3. v2 中间点 probe 曲线（官方 / v1 / v2 同图）。
