# DINOv3 nuImages 续训 2.0（OpenCode 实现规格）

> 目的：修复 1.0 的 split 污染（哈希划分混入官方 val/test 导致评测集泄漏），按严格官方 split 重新续训。
> 仓库：facebookresearch/dinov3。只改数据集层与配置，**不动训练主循环**。

## 0. 背景与原则

- 1.0 教训：续训训练集必须与 probe 评测集（官方 val）**零交集**，否则 A/B 对比失效；
- 判断依据：frozen probe 指标 + 中间 checkpoint 曲线；SSL loss 只作诊断；
- **数据口径（铁律）**：本期全程只用 `CAM_FRONT` 关键帧（单目），sweeps 与其他 5 个视角不参与训练/probe；
- **关键事实（已本地核实，2026-08-17）**：官方 `v1.0-test` 的 `object_ann.json` /
  `surface_ann.json` 是**空数组**（9752 个 sample 但标注 0 条），无公开标注，
  因此 probe 评测使用**官方 val**（16445 样本、有完整标注）；test 不用于任何有监督指标。

## 1. Split 方案（严格官方定义 + train 内部 95:5）

```
官方 v1.0-train：67,279 个 sample（场景时间戳）→ CAM_FRONT 关键帧 13,187 张（有标注）
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
```

probe 替代方案（test 不可用后的落点）：
- **主评测集 = 官方 `v1.0-val`**（与 PROBE_PLAN.md 一致，train/val/test 交集断言保护）；
- **可选补充 = SSL 监控集 `train5`**（官方 train 的 5%，带标注、与训练集同分布，
  可当"域内 held-out"再画一条曲线）；是否启用由后续实验决定，默认不启用。

### 哈希划分实现要点（确定性、可复现）

- **先筛、再哈希**：先从 v1.0-train 筛出 `channel==CAM_FRONT && is_key_frame` 的关键帧（13,187 张，
  对应 13,187 个 sample token；`key_camera_token` 是 6 相机轮换的，**不能**拿全量 67,279 个 sample 当 CAM_FRONT 数）；
  再对这 13,187 个 sample token 计算 `int(md5(token).hexdigest()[:8], 16) % 100`
- `< 95` → SSL 训练集；`>= 95` → SSL 监控集
- seed/比例写进配置与文档，重跑结果一致

## 2. 配置改动（续训 2.0）

基于 7B 配置改造的 `dinov3_vitl16_nuimages_continue.yaml`，相对 1.0 的改动：

```yaml
schedules:
  lr:
    start: 0
    peak: 5.0e-05        # 实际 lr = peak × 4×sqrt(全局batch/1024)，跑 20 步看日志确认
    end: 5.0e-05
    warmup_epochs: 1      # 1000 步
    freeze_last_layer_epochs: 0   # ← 1.0 是 5（10k 里占 50%，浪费）；改为 0
  teacher_temp:
    warmup_epochs: 1      # 同步改小
train:
  batch_size_per_gpu: 128          # ← B200 180G 余量足；注意全局 batch = 128×卡数，LR 随其缩放
  dataset_path: NuImages:root=<nuimages_root>:split=TRAIN:subset=train95
  monitor_gradient_norm: true
  sharded_eval_checkpoint: false
optim:
  epochs: 10              # 10k 迭代
  clip_grad: 3.0
  layerwise_decay: 0.9
  adamw_beta2: 0.999
evaluation:
  eval_period_iterations: 1000     # 每 1k 步存 eval/training_<iter>/teacher_checkpoint.pth（probe 用）
checkpointing:
  period: 1000
  max_to_keep: 10                  # 保留最近 10 个全量 checkpoint
  keep_every: 99999999999999999    # 10k 共 10 个点，max_to_keep=10 已全保留，不需要额外副本
```

说明：

- 两类 checkpoint 路径不同，别混：
  - **全量断点（resume 用）**：`output_dir/ckpt/<iter>/`，受 `max_to_keep` 裁剪；
  - **评测断点（probe 用）**：`output_dir/eval/training_<iter>/teacher_checkpoint.pth`（EMA teacher 的 backbone，
    由 `train.py` 的 `do_test` 保存），**不受裁剪**，1k/2k/.../10k 共 10 个；
- 1.0 只看到 7999/8999/9999：那是全量断点目录被 `max_to_keep: 3` 清剩的；同时 1.0 的
  `eval_period_iterations` 用的是默认 12500（大于 10000），导致一个评测断点都没存，probe 无中间点可用；
  2.0 同时改 `max_to_keep=10` 和 `eval_period_iterations=1000`；
- 若以后把训练延长到 2 万步以上，再把 `keep_every` 设成 1000 作为长命副本保险（不会被清），10k 内用不上；
- batch 128 的显存估算：峰值约 50GB（180GB 卡余量充足）；若 OOM 回退 64；
- 续训监控（SSL loss 趋势）用 `subset=train5` 单独跑一个监控 job（可选），不进训练。

## 3. 数据集类改动（dinov3/data/datasets/nuimages.py 或现有 NuImages 类）

- 支持官方 split：`split=TRAIN/VAL/TEST` 映射 `v1.0-train/v1.0-val/v1.0-test`（若尚未支持）；
- 新增 `subset` 参数：`train95` / `train5` / `None`（None = 全量该 split）；
  `train95`/`train5` 按 §1 哈希规则在 v1.0-train 内部划分，**只对 v1.0-train 生效**；
- `split=VAL/TEST` 时 `subset` 必须为 None（val/test 不得再切）；
- `camera` 参数本期固定 `CAM_FRONT`；实现时按 `key_camera_token → channel == "CAM_FRONT"` 过滤，
  其他视角/sweeps 的数据不加载（数据集类里写死或配置默认，禁止混入）；
- 其余读取逻辑（key_camera_token → channel 过滤 CAM_FRONT、RLE mask、bbox）保持不变；
- dataset 字符串示例：
  - SSL 训练：`NuImages:root=...:split=TRAIN:subset=train95:camera=CAM_FRONT`
  - SSL 监控（可选）：`NuImages:root=...:split=TRAIN:subset=train5:camera=CAM_FRONT`
  - Probe 训练：`NuImages:root=...:split=TRAIN:camera=CAM_FRONT`
  - Probe 评测：`NuImages:root=...:split=VAL:camera=CAM_FRONT`

## 4. 验证步骤（OpenCode 执行，全部通过才算完成）

1. **split 断言**：运行 §1 的四个交集检查，输出必须全 0；
2. **test 标注检查**：本地已核实 `v1.0-test` 的 object/surface 标注为**空数组**（文件存在≠有标注，
   必须数记录数）；工作 PC 上只需复跑一次 `len(json.load(...)) == 0` 的统计确认，不阻塞其余工作；
   probe 评测维持官方 val；
3. **smoke test**：全新 `--output-dir`，确认启动日志 `Starting training from iteration 0` +
   `Loading pretrained weights from <wrapped>`；跑 20 步，确认：
   - lr 从 0 线性爬升（warmup 1000 步），非 0 且非跳变；
   - `last_layer_lr` 与 lr 一致（freeze=0 生效）；
   - 显存峰值 < 120GB；
4. **checkpoint 频率**：跑过 1k 步后确认 `output_dir/ckpt/999/` 全量断点与
   `output_dir/eval/training_999/teacher_checkpoint.pth` 评测断点都出现；跑完 10k 后确认
   `ckpt/` 下 999/1999/.../9999 共 10 个目录齐全（未被 `max_to_keep` 清掉）；
5. 记录 batch 128 下实际 lr（日志字段）与迭代时间，供超参复核。

## 5. 禁止事项

- 不改 `train.py` / `ssl_meta_arch.py` / 训练主循环；
- 不把官方 val/test 的任何 sample 放入 SSL 训练或监控（断言保护）；
- 不下载模型权重（由用户提供路径）；
- 不使用 1.0 的 output_dir（必须新目录，防止 resume 污染）。

## 6. 交付物

1. 数据集类改动（文件/函数/行清单）+ 配置 yaml；
2. §4 验证输出（断言、test 标注结论、smoke 日志、checkpoint 列表）；
3. 记录 SSL 训练集/监控集/val/test 各自 sample 数与交集数。
