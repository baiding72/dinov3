# nuScenes 框架配置笔记

> 依据：nuScenes 论文（arXiv:1903.11027，CVPR 2020）+ 官方 devkit（github.com/nutonomy/nuscenes-devkit）。

## 1. 数据集事实

- 规模：1000 个 20s 场景（Boston Seaport/South Boston + Singapore One North/Holland Village/Queenstown），共 5.5h、约 242km；850 个 train/val + 150 个 test。
- 关键帧：40k 个标注关键帧（2Hz 标注，可插值到 10Hz）。
- 传感器：6×相机（1600×900, 12Hz）+ 1×32 线激光雷达（20Hz, ≤70m）+ 5×毫米波雷达（≤250m）+ GPS/IMU。
- 标注：23 类、1.4M 个 3D 框，含属性（行人姿态、车辆状态）；11 层语义地图（车道、路缘、人行道等）。
- 覆盖条件：昼夜、雨、多云、施工区；波士顿/新加坡左右舵混合。
- 授权：CC BY-NC-SA 4.0（仅非商业研究）。

## 2. 数据格式（database schema）

devkit 把数据组织为 SQLite 数据库（`nuScenes` 类）+ 磁盘文件：

- 顶层表：`scene`（场景）→ `sample`（关键帧）→ `sample_data`（某传感器在某关键帧的数据文件，含 token 树）。
- 标注表：`category`（类别）、`instance`（实例）、`sample_annotation`（3D 框）、`attribute`（属性）。
- 传感器表：`calibrated_sensor`、`ego_pose`、`log`、`map`、`lidarseg`/`panoptic`（扩展任务）。
- 时间轴：`sample` 之间通过 `prev`/`next` token 串联；`sample_data` 是 2Hz/12Hz/20Hz 各自的时间序列。

## 3. 任务与指标（E2E 相关）

| 任务 | 主要指标 |
| --- | --- |
| 3D 检测 | mAP + NDS（nuScenes Detection Score：mAP + ATE/ASE/AOE/AVE/AAE 加权） |
| 3D 跟踪 | AMOTA / AMOTP / IDS |
| 预测 | minADE / minFDE / miss rate |
| 地图 | IoU（车道/路缘） |
| 占用 | IoU / VPQ |
| **规划（E2E 开环）** | **avg.L2（m）↓、avg.Col（碰撞率 %）↓**，3s 规划视野（UniAD 协议） |

## 4. 配置步骤（本机实测环境）

当前环境：macOS（Apple Silicon，/opt/homebrew），系统 Python 3.14.4，**尚未安装 torch/nuscenes-devkit**。

建议用 conda 建独立环境（nuscenes-devkit 依赖 pyquaternion、numpy、tqdm 等，Python 3.14 兼容性风险高）：

```bash
# 1) 创建环境（Python 3.10 或 3.11 最稳）
conda create -n nuscenes python=3.11 -y
conda activate nuscenes

# 2) 安装 devkit
pip install nuscenes-devkit
# 如需摄像头检测等还可用 mmdet3d（按需，较重）

# 3) 下载数据（需在 https://www.nuscenes.org/nuimages 注册账号）
#   下载 nuScenes full dataset（trainval ~260GB / test ~80GB，或 mini ~4GB 先跑通）
#   目录结构约定：
#   $NUSCENES_ROOT/
#     maps/  samples/  sweeps/  v1.0-mini/  ...

# 4) 环境变量
export NUSCENES_ROOT=/path/to/nuscenes
```

## 5. 最小可运行示例

```python
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-mini", dataroot="/path/to/nuscenes", verbose=True)

# 场景 / 关键帧 / 数据文件
scene = nusc.scene[0]
first_sample_token = scene["first_sample_token"]
sample = nusc.get("sample", first_sample_token)

# 遍历一个关键帧的 6 相机 + lidar + radar 文件
for sensor in ["CAM_FRONT", "LIDAR_TOP", "RADAR_FRONT"]:
    data = nusc.get("sample_data", sample["data"][sensor])
    print(sensor, data["filename"])

# 读取 3D 标注（框 + 类别 + 属性）
for ann_token in sample["anns"]:
    ann = nusc.get("sample_annotation", ann_token)
    cat = nusc.get("category", ann["category_token"])["name"]
    print(cat, ann["translation"], ann["size"], ann["rotation"])

# 语义地图
from nuscenes.map_expansion.map_api import NuScenesMap
map_api = NuScenesMap(dataroot="/path/to/nuscenes", map_name="singapore-hollandvillage")
```

## 6. 规划数据集使用（E2E 开环协议）

- E2E 模型（UniAD 等）通常只用 6 相机 + ego 状态，输出未来 3s 轨迹，对比 GT 轨迹算 L2 与碰撞率。
- 关键实现细节：输入分辨率、时间窗口（历史帧数）、BEV 尺寸（如 200×200，0.5m/pixel）、是否用 HD map（UniAD 不用）。
- 数据量对比：nuScenes 只有 1000 场景，属于"小数据基准"；**用户的"自监督数据上量"目标意味着要引入未标注驾驶视频**（如大规模行车记录），这正是 DINOv3 式 SSL 发挥作用的地方。

## 7. 结合 DINOv3 的切入点

1. 用 DINOv3 冻结特征做 nuScenes 相机输入的 dense feature backbone（替代或增强 BEVFormer 的透视编码器）。
2. 用 DINOv3 特征为未标注帧生成"伪监督"（特征蒸馏/一致性），再用于 BEV 表征或规划特征预训练。
3. 注意评价闭环：nuScenes 开环指标提升 ≠ 真车安全，后续应补 CARLA/nuPlan 闭环验证。

