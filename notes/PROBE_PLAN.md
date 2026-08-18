# nuImages 下游 Frozen-Probe 验证方案（OpenCode 实现规格）

> 用途：实现并验证 DINOv3 ViT-L/16 续训的冻结特征下游评测（A/B 对照 + 中间 checkpoint 曲线）。
> 仓库：facebookresearch/dinov3。本方案只涉及**评测侧新代码**，禁止改动任何训练相关文件。

## 0. 背景与目标

验证问题：**DINOv3 在 nuImages 上续训后，冻结特征是否比原始 DINOv3 更适合自驾视觉任务？**

- 对照 A：原始 DINOv3 ViT-L/16（官方权重）
- 对照 B：续训各中间 checkpoint（`eval/<iter>/teacher_checkpoint.pth`，如 5k、10k 步）
- 判定标准：frozen probe 指标随 SSL 步数的**方向与一致性**；SSL 训练 loss 不作为成功指标。
- 公平性铁律：A/B 用**同一脚本、同一 seed、同一分辨率、同一 probe 超参**，只换 backbone 权重；评测集固定 `nuImages v1.0-val`。
- 备注（已核实）：官方 `v1.0-test` 的 object/surface 标注为空数组，无有监督评测价值，故全方案只用 train/val。
- 数据口径（铁律）：所有指标、特征提取只用 `CAM_FRONT` 关键帧（单目），sweeps 与其他视角不参与。
- 规模口径（已实测）：sample（场景时间戳）数 ≠ CAM_FRONT 图像数；`key_camera_token` 为 6 相机轮换。
  v1.0-train / val / test 的 CAM_FRONT 关键帧分别为 **13,187 / 3,249 / 1,932** 张（mini 为 8 张）。
- 执行顺序（铁律）：先完成 §2 指标修复与验证，再进入完整 A/B 流程；尺子可信之前不下任何实验结论。

## 1. 指标与协议（预注册，写死，不事后改）

### 指标 1：drivable-surface segmentation（dense 指标，主）
- 冻结 backbone，输入 512×512，取最后一层 patch 特征（32×32×1024）
- 线性头（1×1 conv），二分类 drivable / 非 drivable
- 目标来自 `surface_ann.mask`（RLE 解码 → 二值图，原图 900×1600 → resize 到 512 与特征对齐）
- **只取 `category == flat.driveable_surface`**，同一关键帧的多条记录 **OR 合并**成一张二值 drivable mask
  （注意 `surface_ann` 还包含 `vehicle.ego` 自车 mask，不能混入）
- 训练集：v1.0-train 关键帧（channel==CAM_FRONT 且 is_key_frame）；评测：v1.0-val
- 指标：**drivable IoU**（主）+ F1/PR（辅助）

### 指标 2：object region classification（区域/全局指标）
- 冻结 backbone，512×512；对每个 `object_ann.bbox` 取 bbox 内 patch token **均值池化**作为物体特征
- 线性分类到 category（**5 coarse**：vehicle / human / movable_object / static_object / animal，
  由 category.json 的 name 首段聚合；细类共 24 个，总类别 25 个，其中 flat.driveable_surface 只出现在
  surface_ann；object_ann 里罕见的 vehicle.ego 记录应过滤）
- 训练集：v1.0-train 的 object_ann；评测：v1.0-val
- 指标：**top-1 acc**（按 object 样本计）+ 可选 mAP

### 指标 3：image-level multi-label classification（CLS 全局指标，补过拟合盲区）
- 背景：DINO loss 作用在 CLS token 上，patch 级指标测不到 CLS 特征可能的退化；
  判断续训是否过拟合必须同时看全局特征，故增加本指标
- 冻结 backbone，512×512，取 **CLS token**（1024 维）
- 线性头 `Linear(1024→8) + sigmoid`，**BCE 多标签**
- 标签：该关键帧内出现的 object 类别集合（object_ann 的 category 聚合 → 8 维多标签向量）
- 训练集：v1.0-train 关键帧（channel==CAM_FRONT 且 is_key_frame，带 object_ann）；评测：v1.0-val
- 指标：**mAP**（per-class AP 平均）+ 可选 top-1（最高概率类是否在标签集合中）
- 与指标 1/2 的关系：patch 测 dense、CLS 测 global；三者任一在中间曲线上"先升后降"才是过拟合信号

### 统一协议
```
输入分辨率: 短边 resize 512（等比, BICUBIC）→ CenterCrop 512×512
            （对齐官方 make_classification_eval_transform 语义；原图 1600×900 → 910×512 → 裁 512×512）
特征层: backbone 最后一层 patch tokens（norm 后）
head: Linear（主指标）/ MLP-256（辅助，可选）
优化器: Adam, lr=1e-3, weight_decay=0
训练: 10 epochs, batch_size=64, seed=0
评测: v1.0-val, 单尺度
```

### 分辨率与几何变换协议（复现必需，写死）

| 环节 | 值 | 说明 |
| --- | --- | --- |
| 数据集原始分辨率 | 1600×900 | nuImages 固定，16:9 宽幅 |
| SSL 续训输入 | global 256×256 / local 112×112 | RandomResizedCrop（随机裁方形），官方配方 |
| Probe 输入 | 短边 resize 512（等比, BICUBIC）→ CenterCrop 512×512 | 非拉伸；对齐官方 eval transform 语义 |
| Probe 特征图 | 32×32 | 512/16 |
| mask / bbox 对齐 | 与图像同一变换 | 短边 512 + 中心裁剪，坐标相应变换，否则错位 |

注意：1600×900 是宽幅，方形 crop 会丢失左右视野（910×512 → 裁 512×512 丢左右各约 199px）；
该事实用于解释指标，A/B 必须使用完全相同的变换。

## 2. 指标修复与验证（P0，先做，不需要新训练）

背景：v1 三个指标全线下滑，但 **baseline 本身不可信**——官方 drivable IoU=0.0168 明显低于合理水平
（drivable 通常占画面 20~40%），怀疑实现 bug；object top-1=0.857 需对照多数类基线。
**先修尺子，再谈 A/B。** 本步只使用现有 checkpoint（官方权重 + v1 全量断点 7999/8999/9999），
不启动任何新训练。

### 2.1 指标 1（drivable IoU）修复清单

- **可视化三重校验**（输出 `<output>/probe_debug/`，每步抽查 10 张）：
  1. 原图 + 解码后 mask 叠加（1600×900 原始坐标，验证 RLE 解码）；
  2. 512 变换后图像 + mask 叠加（验证短边 resize + CenterCrop 与图像一致）；
  3. 32×32 特征网格 + 下采样 mask 叠加（画网格线，验证 mask 与特征网格对齐）。
- **RLE 解码**：`counts` 是 base64 字符串，必须先 `base64.b64decode` 再 `maskUtils.decode`；
  或直接用 devkit `from nuimages.utils.utils import mask_decode`（内部已封装）。
- **category 过滤**：只留 `category_name == "flat.driveable_surface"`；`vehicle.ego` 必须排除。
- **同帧多条记录**：同一关键帧（sample_data_token）的多条 drivable mask 用 **OR 合并**成一张二值图。
- **mask 下采样到 32×32**：与图像同一几何变换后，用面积平均（area-average）或与 patch 网格严格对齐的方式；
  禁止用会引入半像素偏移的 resize。
- **已知病灶（2026-08-18 实测，最高嫌疑）**：val 关键帧的 grid positive fraction 呈 0/1 双峰
  （56% 为 0、44% 为 1），而真实 drivable 占比应为连续分布 ~0.3–0.5（mini 实测 8 张 CAM_FRONT：
  min 0.332 / mean 0.390 / max 0.478）。整图全 0 或全 1 说明目标网格被**图像级标量阈值**填充了，
  典型错误写法：
  ```
  frac = mask_crop.mean()          # 整图平均 → 一个标量（≈0.45，贴着 0.5 两侧 → 56/44 双峰）
  grid = (frac > 0.5)              # 一个 0/1 标量
  grid = grid.broadcast_to(32, 32) # 广播成整张网格 → 全 0 或全 1
  ```
  等价错误：`adaptive_avg_pool2d(mask, (1, 1))` 或 `mask.mean(dim=(1,2,3))`。
  正确做法：**逐格**面积平均 `adaptive_avg_pool2d(mask[None, None], (32, 32))` → 32×32 占比图 → 逐格 `> 0.5`。
  该 bug 会把目标退化成"整图二分类"，线性头只能学图像级猜测，这正是官方 baseline=0.0274、续训=0 的直接原因。
- **第一面板自检（原图+原始 mask 叠加，2026-08-18 实测判据）**：
  数据事实（已实测）：surface_ann 只存在于关键帧；每个 CAM_FRONT 关键帧**恰好 1 条**
  `flat.driveable_surface` 记录（train 13184/13184，val 3249/3249）；val 全分辨率 drivable 占比
  min 0.079 / mean 0.303 / max 0.462，**不存在 0 或 1**。
  因此服务器 debug 第一张图若出现"整图全绿 / 分界线在天上"，问题必在 mask 生成（变换之前），
  按三条指纹定位：
  1. 打印该关键帧 `sample_data_token` 对应的 surface_ann 记录数 → 必须恰好 1；
  2. 打印该记录 `category_name` → 必须是 `flat.driveable_surface`；
  3. 打印解码 mask 全分辨率占比 → 必须落在 0.08~0.46；=1.0 → RLE 解码错；0.6~0.8 → 串相机/尺寸错
     （最常见：用 `sample_token` 而非 `sample_data_token` 关联，混入同一 sample 的其他相机 mask）。
  黄金标准：同一记录分别用 devkit `mask_decode` 与实现解码，逐像素对比 **IoU 必须 = 1**。
- **指标输出**：IoU + F1 + PR 一起报，不能只报 IoU。
- **修复后断言（全部通过才算修好）**：
  1. val 全部关键帧的 grid positive fraction 输出直方图：应为 0.2–0.6 的连续分布，**不允许出现 0.0 / 1.0**
     （除个别极端图外）；
  2. 全 val 均值落在 0.3–0.5（mini 实测 0.39）；
  3. 单图单元测试：用 mini 的 debug_0 / debug_1 两张图，网格占比必须等于 **0.375 / 0.456**
     （本机 `notes/debug_vis_nuimages.py` 的输出值），对不上就是变换或阈值仍有问题。
- **验收**：官方 baseline 必须显著高于 0.0168；若修复后仍 <0.1，继续排查：
  特征层选择（是否末层 norm 后）、checkpoint 加载（teacher vs student、`backbone.` 前缀）。

### 2.2 指标 2（object top-1）补基线

- 输出 train/val 的 5 coarse 分布表（附 24 fine 明细，尾类按 ≥100 实例过滤或单列）；
- 输出"全猜多数类"基线 top-1（与官方 0.857 对照，判断该数字是否有信息量）；
- 输出 **per-class accuracy + balanced accuracy**，禁止只报全局 top-1；
- bbox→patch 选择可视化：原图 + bbox 框 + 被选中 patch 网格标亮，抽查 10 张；
- 验收：能明确回答"官方 0.857 是否等于多数类基线"。

### 2.3 指标 3（CLS mAP）补细节

- 确认多热标签 = 该关键帧（channel==CAM_FRONT 且 is_key_frame）内 object_ann 全部类别聚合；
- 输出 **per-class AP 表**，不只看均值；
- 验收：与已测 v1 值 0.678 / 0.574 一致，即认为实现正确，无需大改。

### 2.4 修复后重测与决策闸门

- 用修复后**同一管线**重测：官方权重 + v1 的 7999/8999/9999
  （从全量断点提取 EMA teacher backbone，沿用现有可读 v1 断点的加载方式）；
- 填 §6 结果表模板，并输出**修复前后 baseline 对比**（旧 drivable IoU=0.0168 vs 新值）；
- 判定：
  - 三个指标仍全线下滑（指标 1 用修好的值）→ 结论成立：无锚 SSL 续训退化特征 →
    v2 上 GRAM anchoring + 步数压到 5k + 降有效 LR；
  - 下滑大幅缩水或消失 → v1 结论改写，v2 设计重排；
- 产出：每步可视化图 + 修复前后对比表。

## 3. 数据层（字段已在本机 nuImages mini 实测确认）

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
- **解码方式**：`counts` 是 base64 字符串，不能直接 `maskUtils.decode`，必须先用 `base64.b64decode` 再解；
  直接复用官方 devkit：`from nuimages.utils.utils import mask_decode`（内部已完成 b64decode + decode）
- `category.json` 提供 `name`（25 细类，首段聚合为 5 coarse）
- 标注按 `sample_data_token` 与关键帧对齐

## 4. 代码结构（新建，位于 `dinov3/eval/nuimages/`）

```
dinov3/eval/nuimages/
├── __init__.py
├── config.py            # 协议常量（分辨率/超参/路径），A/B 共用
├── data.py              # NuImagesProbeDataset
├── debug_vis.py         # 指标修复可视化（§2.1/2.2 的三重校验与 bbox→patch 抽查）
├── extract_features.py  # 冻结 backbone 提取并缓存特征（每 checkpoint 一次）
├── drivable_probe.py    # 指标 1
├── object_probe.py      # 指标 2
└── cls_probe.py         # 指标 3
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
- 每张图缓存：`patch_feats.pt` [32,32,1024]、`drivable_mask.pt` [32,32]（resize 后）、`objects.pt`（patch 网格坐标的 bbox + 类别）、`cls_feat.pt` [1024]
- 支持多进程/多卡并行（B200 显存充裕，batch 可开大）

### drivable_probe.py / object_probe.py / cls_probe.py 实现要点
- 只读特征缓存，不碰原图/backbone
- 训练线性头（或 MLP-256），协议见 §1
- 输出：IoU/F1（指标 1）、top-1 acc（指标 2）、mAP（指标 3），打印到 stdout + 写 json
- 命令行参数：`--features-dir`、`--split VAL`、`--seed 0` 等；A/B 只换 `--features-dir`

## 5. 执行步骤

> 入口：先完成 §2 指标修复与验证（尺子可信），再执行以下完整流程。

1. **数据层 + 可视化**：实现 `data.py`，输出 2–3 张"原图 + drivable mask 叠加"和"原图 + bbox 叠加"图，验证 RLE 解码与坐标对齐；统计 CAM_FRONT 关键帧的标注覆盖数（train/val 各多少带 surface/object 标注）
2. **特征提取**：`extract_features.py` 对 A（raw）+ B（5k、10k）各提取一次
3. **Probe 跑通**：先在 A 上跑两个指标，确认数值合理（drivable IoU 明显高于随机、object top-1 明显高于多数类基线）
4. **B 各中间点**：同一命令换 `--features-dir`
5. **产出**：结果表 + 曲线（IoU / top-1 / mAP vs SSL steps，0 步 = A）
6. **Gate 分析**：看两条指标的方向与中间点一致性

## 6. 结果表模板

```
Model                     Drivable IoU ↑    Object top-1 ↑    CLS mAP ↑
Original DINOv3 (0 step)      ?               ?                 ?
Adapted (5k steps)            ?               ?                 ?
Adapted (10k steps)           ?               ?                 ?
```

## 7. 已知的坑（实现时规避）

- RLE mask 是原图分辨率 900×1600：resize 到 512 时用与图像一致的插值，避免 mask 与特征错位
- bbox 是原图坐标：映射到 32×32 patch 网格时除以 16、clip 边界、空 bbox 丢弃
- SSL 训练与 probe 训练同源（都用 v1.0-train）：A/B 协议一致即可接受，报告注明 caveat
- 类别不平衡：drivable 占大头，必报 F1；object 按 5 coarse 聚合（vehicle≈49%），并报 per-class + balanced
- `teacher_checkpoint.pth` 加载：eval 路径自动去掉 `backbone.` 前缀（`init_model_from_checkpoint_for_evals`）
- 确认续训输出目录里有 5k/10k 的 `eval/<iter>/teacher_checkpoint.pth`

## 8. 禁止事项

- 不改 `train.py` / `ssl_meta_arch.py` / 任何训练配置
- 不下载模型权重（权重路径由用户提供）
- 不改动 nuImages 原始数据
- 不启动任何训练

## 9. 交付物

1. `dinov3/eval/nuimages/` 全部代码
2. 可视化验证图（原图 + mask / bbox 叠加）
3. 特征缓存（3 份：raw / 5k / 10k）
4. A/B 对比表 + 曲线
5. 简短分析（是否 Gate 通过）
