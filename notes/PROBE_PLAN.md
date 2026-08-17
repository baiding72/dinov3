# nuImages 下游 Frozen-Probe 验证方案（OpenCode 实现规格）

> 用途：实现并验证 DINOv3 ViT-L/16 续训的冻结特征下游评测（A/B 对照 + 中间 checkpoint 曲线）。
> 仓库：facebookresearch/dinov3。本方案只涉及**评测侧新代码**，禁止改动任何训练相关文件。

## 0. 背景与目标

验证问题：**DINOv3 在 nuImages 上续训后，冻结特征是否比原始 DINOv3 更适合自驾视觉任务？**

- 对照 A：原始 DINOv3 ViT-L/16（官方权重）
- 对照 B：续训各中间 checkpoint（`eval/<iter>/teacher_checkpoint.pth`，如 5k、10k 步）
- 判定标准：frozen probe 指标随 SSL 步数的**方向与一致性**；SSL 训练 loss 不作为成功指标。
- 公平性铁律：A/B 用**同一脚本、同一 seed、同一分辨率、同一 probe 超参**，只换 backbone 权重；评测集固定 `nuImages v1.0-val`。

## 1. 指标与协议（预注册，写死，不事后改）

### 指标 1：drivable-surface segmentation（dense 指标，主）
- 冻结 backbone，输入 512×512，取最后一层 patch 特征（32×32×1024）
- 线性头（1×1 conv），二分类 drivable / 非 drivable
- 目标来自 `surface_ann.mask`（RLE 解码 → 二值图，原图 900×1600 → resize 到 512 与特征对齐）
- 训练集：v1.0-train 关键帧（channel==CAM_FRONT 且 is_key_frame）；评测：v1.0-val
- 指标：**drivable IoU**（主）+ F1/PR（辅助）

### 指标 2：object region classification（区域/全局指标）
- 冻结 backbone，512×512；对每个 `object_ann.bbox` 取 bbox 内 patch token **均值池化**作为物体特征
- 线性分类到 category（8 大类）
- 训练集：v1.0-train 的 object_ann；评测：v1.0-val
- 指标：**top-1 acc**（按 object 样本计）+ 可选 mAP

### 统一协议
```
输入分辨率: 512×512
特征层: backbone 最后一层 patch tokens（norm 后）
head: Linear（主指标）/ MLP-256（辅助，可选）
优化器: Adam, lr=1e-3, weight_decay=0
训练: 10 epochs, batch_size=64, seed=0
评测: v1.0-val, 单尺度
```

## 2. 数据层（字段已在本机 nuImages mini 实测确认）

### nuImages 目录结构
```
<root>/
├── samples/   # 关键帧图像（按 6 视角子目录）
├── sweeps/    # 中间帧图像（按 6 视角子目录，第二阶段时序用）
└── v1.0-{train,val,test,mini}/   # 元数据：split 由该目录决定
```

### 关键帧关联链
```
sample.json  → token, key_camera_token, log_token, timestamp
sample_data.json → filename("samples/CAM_FRONT/xxx.jpg"), is_key_frame,
                   calibrated_sensor_token, prev/next, sample_token
calibrated_sensor.json → sensor_token
sensor.json  → channel("CAM_FRONT" ...)
```
- 每个 sample 的关键帧 = `key_camera_token` 指向的 sample_data
- 单相机筛选：解析 key_camera_token 的 channel == "CAM_FRONT"

### 标注字段（已实测）
```json
// surface_ann：drivable 标注，RLE mask
{"category_token": "...", "mask": "{'size': [900,1600], 'counts': '<RLE>'}",
 "sample_data_token": "...", "token": "..."}

// object_ann：2D 框 + 实例 mask + 类别
{"attribute_tokens": "['...']", "bbox": "[725, 293, 909, 522]",
 "category_token": "...", "mask": "{'size': [900,1600], 'counts': '<RLE>'}",
 "sample_data_token": "...", "token": "..."}
```
- `bbox` 是**字符串**，需 `json.loads` 或 `eval` 转 `[x1,y1,x2,y2]`（原图坐标）
- `mask` 是 pycocotools RLE 格式（`maskUtils.decode`）
- `category.json` 提供 `name`（8 大类）
- 标注按 `sample_data_token` 与关键帧对齐

## 3. 代码结构（新建，位于 `dinov3/eval/nuimages/`）

```
dinov3/eval/nuimages/
├── __init__.py
├── config.py            # 协议常量（分辨率/超参/路径），A/B 共用
├── data.py              # NuImagesProbeDataset
├── extract_features.py  # 冻结 backbone 提取并缓存特征（每 checkpoint 一次）
├── drivable_probe.py    # 指标 1
└── object_probe.py      # 指标 2
```

### data.py 实现要点
- 依赖：nuscenes-devkit（**源码方式**，`PYTHONPATH=<devkit>/python-sdk`，PyPI 1.2.0 缺 nuimages 模块）+ pycocotools
- 构造参数：`root`（nuImages 根）、`split`（TRAIN/VAL，映射 v1.0-train/v1.0-val）、`camera="CAM_FRONT"`
- 读取 7 张表：sample / sample_data / calibrated_sensor / sensor / category / surface_ann / object_ann
- 提供：
  - `keyframe_paths()`：关键帧图像路径列表
  - `get_drivable_mask(keyframe_sd_token)`：RLE → 二值 mask（PIL，随后按协议 resize）
  - `get_objects(keyframe_sd_token)`：list of `{"bbox_xyxy", "category_name"}`
- 只读，不改 nuImages 原始数据

### extract_features.py 实现要点
- 加载冻结 backbone（eval 模式、`torch.no_grad()`），输出最后一层 patch 特征
- 每 checkpoint 一个缓存目录：`<output>/probe_features/<name>/`（name = raw / 5k / 10k）
- 每张图缓存：`patch_feats.pt` [32,32,1024]、`drivable_mask.pt` [32,32]（resize 后）、`objects.pt`（patch 网格坐标的 bbox + 类别）
- 支持多进程/多卡并行（B200 显存充裕，batch 可开大）

### drivable_probe.py / object_probe.py 实现要点
- 只读特征缓存，不碰原图/backbone
- 训练线性头（或 MLP-256），协议见 §1
- 输出：IoU/F1（指标 1）、top-1 acc（指标 2），打印到 stdout + 写 json
- 命令行参数：`--features-dir`、`--split VAL`、`--seed 0` 等；A/B 只换 `--features-dir`

## 4. 执行步骤

1. **数据层 + 可视化**：实现 `data.py`，输出 2–3 张"原图 + drivable mask 叠加"和"原图 + bbox 叠加"图，验证 RLE 解码与坐标对齐；统计 CAM_FRONT 关键帧的标注覆盖数（train/val 各多少带 surface/object 标注）
2. **特征提取**：`extract_features.py` 对 A（raw）+ B（5k、10k）各提取一次
3. **Probe 跑通**：先在 A 上跑两个指标，确认数值合理（drivable IoU 明显高于随机、object top-1 明显高于多数类基线）
4. **B 各中间点**：同一命令换 `--features-dir`
5. **产出**：结果表 + 曲线（IoU/top-1 vs SSL steps，0 步 = A）
6. **Gate 分析**：看两条指标的方向与中间点一致性

## 5. 结果表模板

```
Model                     Drivable IoU ↑    Object top-1 ↑
Original DINOv3 (0 step)      ?               ?
Adapted (5k steps)            ?               ?
Adapted (10k steps)           ?               ?
```

## 6. 已知的坑（实现时规避）

- RLE mask 是原图分辨率 900×1600：resize 到 512 时用与图像一致的插值，避免 mask 与特征错位
- bbox 是原图坐标：映射到 32×32 patch 网格时除以 16、clip 边界、空 bbox 丢弃
- SSL 训练与 probe 训练同源（都用 v1.0-train）：A/B 协议一致即可接受，报告注明 caveat
- 类别不平衡：drivable 占大头，必报 F1；object 按 8 大类聚合
- `teacher_checkpoint.pth` 加载：eval 路径自动去掉 `backbone.` 前缀（`init_model_from_checkpoint_for_evals`）
- 确认续训输出目录里有 5k/10k 的 `eval/<iter>/teacher_checkpoint.pth`

## 7. 禁止事项

- 不改 `train.py` / `ssl_meta_arch.py` / 任何训练配置
- 不下载模型权重（权重路径由用户提供）
- 不改动 nuImages 原始数据
- 不启动任何训练

## 8. 交付物

1. `dinov3/eval/nuimages/` 全部代码
2. 可视化验证图（原图 + mask / bbox 叠加）
3. 特征缓存（3 份：raw / 5k / 10k）
4. A/B 对比表 + 曲线
5. 简短分析（是否 Gate 通过）
