# nuScenes 家族：子数据集对比 + mini 版数据结构

> 更新于 2026-08-09，基于官方 devkit / 论文 / 实际下载的 mini 数据。

## 1. 家族总览

nuScenes 官网（nuscenes.org，Motional/前 nuTonomy）旗下有多个相关数据集，共用一套数据库 schema 和 devkit，但定位、传感器、标注和评测方式完全不同：

| 数据集 | 定位 | 传感器 | 规模（全量） | 标注/任务 | 典型评测 |
| --- | --- | --- | --- | --- | --- |
| **nuScenes** | 多模态 3D 感知基准 | 6 相机 + 1×32 线 LiDAR + 5 雷达 + GPS/IMU | 1000 场景×20s（5.5h）；40k 关键帧；1.4M 3D 框 | 3D 检测/跟踪/预测/地图 | mAP+NDS、AMOTA、minADE |
| **nuImages** | 2D 图像感知（纯视觉） | 6 相机（无 LiDAR/雷达） | 93k 关键帧（每帧含前后各 6 帧→约 1.2M 图像）；约 69 万标注 | 2D 检测、语义/实例/全景分割 | 2D mAP、mIoU |
| **nuPlan** | 闭环/开环规划基准 | 规划侧无传感器；v1.1 传感器集：8 相机+5 LiDAR+IMU/GPS（120h 子集） | 约 1300h 驾驶数据、15k+ logs、4 城市 | 场景日志、轨迹、地图；无逐对象标注（自动标注轨迹） | 闭环：计划得分（score）；开环：L2/碰撞率 |
| **nuScenes-lidarseg** | nuScenes 扩展：LiDAR 语义分割 | 同 nuScenes | 约 40k 关键帧逐点标签 | 32 类点级语义 | mIoU |
| **Panoptic nuScenes** | nuScenes 扩展：LiDAR 全景分割+跟踪 | 同 nuScenes | 约 40k 关键帧 | 8 类实例级全景 | PQ、SQ、RQ、PAT |
| **nuScenes-CanBus** | nuScenes 扩展：车辆 CAN 总线 | 车辆状态（转向/油门/刹车/挡位/轮速等） | 全 nuScenes | 低层车辆动力学 | — |
| **nuScenes-map-expansion** | nuScenes 扩展：语义地图 | 地图矢量/栅格 | 4 城市 11 层 | 车道、路缘、人行道、停止线、红绿灯等 | — |

> 注：nuScenes 与 nuImages 共用一个 devkit（nuscenes-devkit）；nuPlan 有独立 devkit（nuplan-devkit）。另有较新的 nuReality（传感器仿真平台）与 nuScenes-QA 等社区扩展，暂不展开。

## 2. 各数据集要点

### nuScenes（3D 感知主线）

- 波士顿（Seaport/South Boston）+ 新加坡（One North/Holland Village/Queenstown），昼夜/雨雪/施工场景，23 类、1.4M 个 3D 框、属性标注。
- 关键帧 2Hz 标注（40k 帧），可插值到 10Hz；每关键帧 6 相机+1 LiDAR+5 雷达同时刻数据。
- 数据组织：`samples/`（关键帧原始数据）、`sweeps/`（帧间数据）、`maps/`、`v1.0-{trainval,test,mini}/`（JSON 元数据表）。
- 任务与指标：检测（mAP + NDS，NDS 综合 mAP 与 5 个误差项）、跟踪（AMOTA/IDS）、预测（minADE/minFDE/MR）、地图（IoU）。
- **E2E 开环规划用法**：只用 6 相机 + ego 状态 → 输出未来 3s 轨迹 → 对比 GT 算 avg.L2（m）与 avg.Col（碰撞率 %），这是 UniAD/VAD 等模型的评测协议。

### nuImages（2D 视觉主线）

- 与 nuScenes 相同 6 相机布局、相同城市，但**没有 LiDAR/雷达**；93k 个关键帧，每帧带前后各 6 帧（0.5s 间隔，2Hz）形成 13 帧短视频片段。
- 标注类型：`object_ann`（2D 框 + 实例掩码，覆盖 8 大类 25 类）与 `surface_ann`（语义面：可行驶区域等）。
- 数据组织：`samples/`（关键帧）、`sweeps/`（非关键帧，无标注）、`v1.0-{train,val,test,mini}/`。**没有 scene 表**（按 log 组织）。
- 典型用途：2D 检测/分割、图像预训练、nuScenes 的 2D 辅助任务数据。

### nuPlan（规划主线）

- 约 1300h 人类驾驶数据、15k+ logs、4 城市（波士顿、匹兹堡、拉斯维加斯、新加坡），v1.1 起含 route plan、红绿灯状态、mission goal。
- **核心不是感知标注**：提供驾驶日志（`splits/*.db`，SQLite 含轨迹/状态/场景标签）+ 语义地图（`maps/*.gpkg`）+ 可选传感器数据（`sensor_blobs/`，8 相机 CAM_F0/B0/L0-2/R0-2 + 5 路 LiDAR 合并点云 MergedPointCloud）。
- 评测：闭环仿真（nuPlan Simulator，计划得分 score = 交通规则/进展/舒适度等综合）与开环（L2、碰撞率）。
- 数据组织（官方要求 ~/nuplan/dataset）：`maps/`、`nuplan-v1.1/splits/{mini,trainval,test}/`、`nuplan-v1.1/sensor_blobs/{mini_set,train_set,test_set}/`。
- 注意：**nuPlan 的"mini"传感器数据并不小**（单个相机 zip 约 48GB、单 LiDAR zip 约 69GB），全量数百 GB；理解结构阶段可只下 splits+地图，传感器按需单组下载。

## 3. mini 版对比与下载方式

| 数据集 | mini 文件 | 压缩大小 | 解压后 | 下载入口 |
| --- | --- | --- | --- | --- |
| nuScenes | `v1.0-mini.tgz` | 约 4.2GB | 约 9GB | `https://www.nuscenes.org/data/v1.0-mini.tgz`（免登录直链） |
| nuImages | `nuimages-v1.0-mini.tgz` | 约 118MB | 数百 MB | `https://www.nuscenes.org/data/nuimages-v1.0-mini.tgz`（免登录） |
| nuScenes-lidarseg | `nuScenes-lidarseg-mini-v1.0.tar.bz2` | 约 1.8MB | 小 | 同站 `/data/nuScenes-lidarseg-mini-v1.0.tar.bz2`（免登录） |
| nuPlan（logs+maps） | `nuplan-v1.1_mini.zip` + `nuplan-maps-v1.1.zip` | 约 7GB + 1GB | 约 13GB + 2GB | S3 公开桶（见下） |
| nuPlan（传感器） | `sensor_blobs/mini_set/nuplan-v1.1_mini_camera_{0..8}.zip`、`..._lidar_{0..8}.zip` | 单文件 48–69GB | 巨大 | 同上（按需） |

### 官方账号流程（li-xl/nuscenes-download 的原理）

1. 官网注册并同意条款（nuscenes.org/nuscenes → Download）。
2. 前端用 Cognito（client id `7fq5jvs5ffs1c50hd3toobb3b9`）以 USER_PASSWORD_AUTH 换 IdToken。
3. 调归档 API `https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/v1/archives/{version}/{filename}?region=asia|us&project={nuScenes|nuImages|nuPlan}` 拿 CloudFront 签名 URL。
4. 该 API 目前确认支持 nuScenes（v1.0）；nuPlan 不在其中（404），需直连 S3。

### nuPlan S3 公开桶（免登录）

- 桶：`motional-nuplan`（region：ap-northeast-1，即东京）
- 域名：`https://motional-nuplan.s3.ap-northeast-1.amazonaws.com`（注意用点式域名，横杠域名解析异常）
- mini 核心文件：
  - `public/nuplan-v1.1/nuplan-v1.1_mini.zip`
  - `public/nuplan-v1.1/nuplan-maps-v1.1.zip`
  - `public/nuplan-v1.1/sensor_blobs/mini_set/nuplan_mini_sensor.txt`（传感器文件清单）
- 直连速度不稳（本机实测 50–200KB/s），建议 aria2 多线程：`aria2c -x16 -s16 -k1M <url>`

## 4. 本机已配置的 mini 数据（2026-08-09）

```
~/nuscenes/                 # NuScenes(dataroot) 解压后 9.0GB
├── samples/                # 6 相机 + LiDAR + 5 雷达关键帧
├── sweeps/
├── maps/                   # 4 城市语义地图
├── lidarseg/v1.0-mini/     # 点级语义标签 + lidarseg.json
└── v1.0-mini/              # JSON 元数据表

~/nuimages/                 # NuImages(dataroot)
├── samples/                # 6 相机关键帧
├── sweeps/
└── v1.0-mini/

~/nuplan/                   # nuPlan（已完成）
├── dataset/data/cache/mini/    # 64 个日志 .db（已解压，15GB）
├── dataset/nuplan-maps-v1.0/   # 地图（已解压）：4 城市 map.gpkg + vegas 栅格 + nuplan-maps-v1.0.json
└── nuplan_mini_sensor.txt  # 传感器文件清单
```

nuPlan mini 实测补充：`nuplan_mini_sensor.txt` 列出 64 个有传感器数据的日志（分 8 组），传感器 zip 命名 `nuplan-v1.1_mini_camera_{0..8}.zip` / `nuplan-v1.1_mini_lidar_{0..8}.zip`，每个单文件 48–69GB。

nuPlan 日志库（SQLite `.db`）实测结构（单个 mini log，约 15s×10 场景）：

```
表：camera(8) / category(7) / ego_pose(18494) / image(14640) / lidar(1)
    lidar_box(72489) / lidar_pc(3660) / log(1) / scenario_tag(6847)
    scene(10) / track(311) / traffic_light_status(20300)
```

- `scene`：每 log 10 个场景（token、goal、时间戳序列）；`track`：agent 轨迹（类别、宽高长）；`lidar_box`：自动标注的 3D 框；`image`：8 相机 1920×1080 图像索引；`camera`：8 个相机（CAM_F0/B0/L0-2/R0-2，DesignCore D3CM-IMX390）。

## 5. 本机实际读取到的结构统计（devkit 实测）

### nuScenes v1.0-mini

```
场景 scene: 10；关键帧 sample: 404；样本数据 sample_data: 31,206
标注 sample_annotation: 18,538；类别 category: 32（有效 24）；实例 instance: 911
地图 map: 4（boston-seaport / singapore-onenorth / singapore-queenstown / singapore-hollandvillage）
传感器分布：camera 14,008（6 路）；lidar 3,935；radar 13,263（5 路）
```

### nuImages v1.0-mini

```
日志 log: 44；关键帧 sample: 50；sample_data: 650（关键帧 50 + 中间帧 600）
2D 目标标注 object_ann: 506；面标注 surface_ann: 58；类别 25；属性 12；传感器 6（仅相机）
```

## 6. devkit 使用要点

- 环境：`conda create -n nuscenes python=3.12` + `pip install nuscenes-devkit`（PyPI 1.2.0 支持 py3.9/3.12）。
- **坑**：PyPI 1.2.0 与 GitHub master 的 `nuscenes` 包都**不含 nuImages 模块**；nuImages 是独立顶层包 `nuimages`，位于 devkit 源码 `python-sdk/nuimages/`，需 `PYTHONPATH=python-sdk` 后 `from nuimages.nuimages import NuImages`。
- 用法：
  ```python
  from nuscenes.nuscenes import NuScenes
  from nuimages.nuimages import NuImages
  nusc = NuScenes(version="v1.0-mini", dataroot="~/nuscenes", verbose=False)
  nuim = NuImages(version="v1.0-mini", dataroot="~/nuimages", verbose=False)
  ```
- nuScenes 表：attribute/calibrated_sensor/category/ego_pose/instance/lidarseg/log/map/sample/sample_annotation/sample_data/scene/sensor/visibility。
- nuImages 表：attribute/calibrated_sensor/category/ego_pose/log/object_ann/sample/sample_data/sensor/surface_ann（**无 scene**）。

## 7. 磁盘与后续建议

- 当前 ~/nuscenes 解压 9GB + tar 4GB（tar 校验通过后可删，节省空间，随时可重新下载）。
- nuPlan mini splits 解压约 13GB + 地图 2GB，当前剩余空间约 49GB，够用但紧。
- nuPlan 传感器（相机/LiDAR）单个文件就 48–69GB，**建议暂不下载**；需要时先看 `nuplan_mini_sensor.txt` 清单，按场景/传感器挑选，或用 nuPlan 官方 devkit 的按需读取。
- 注意数据协议：nuScenes CC BY-NC-SA 4.0（非商业研究）；nuPlan 同样需官网账号同意条款后使用。
