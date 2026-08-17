# 下一阶段目标

优先完成**单帧DINOv3自驾域自监督续训的可行性验证**，目标不是“把DINOv3跑起来”，而是回答一个明确问题：

**DINOv3在自动驾驶图像上继续进行自监督训练后，冻结特征是否比原始DINOv3更适合自动驾驶视觉任务？**

1. **第一阶段优先使用nuImages，而不是nuPlan**
   
   nuImages有约9.3万张带2D标注的关键帧，并提供过去最多6帧、未来最多6帧的同相机图像，时间间隔约0.5s，总计约120万张图像；这意味着同一份数据既可以做第一阶段单帧SSL，也可以自然扩展到第二阶段时序SSL
   
   nuImages的关键帧提供前景目标**2D box、instance mask、类别和属性**，同时提供`diveable_surface`和ego vehicle的像素级mask；还包含时间戳、ego pose、速度等信息
   
   nuPlan留到后续E2E阶段更合适：它包含1282小时、4个城市的驾驶数据以及自动标注的object tracks、traffic light，并提供规划仿真与评测框架；传感器侧为8路10Hz、2000✖️1200相机

2. **训练前先完成实验设计，不允许直接让OpenCode开始写训练代码**
   
   必须先明确
   
   ```
   Dataset->split->input->model->loss->optimizer->training schedule->
   checkpoint->evaluation->success/failure criterion
   ```
   
   同时先写两个对照实验：
   
   **A：原始DINOv3；B：nuImages SSL continued-pretraining后的DINOv3**
   
   两者后续必须用完全相同的数据和probe评测

3. **第一版实验严格限制变量**
   
   只用**单相机、单帧图像**，建议先固定前视相机；从官方DINOv3 checkpoint初始化，不从零训练。自监督训练只使用train split图像，不使用任何标签。DINOv3官方训练目标本身包含DINO self- distillation、iBOT masked-image modeling、KoLeo regularization，并通过Gram anchoring保护dense feature；第一版应尽量沿用官方实现，不自行设计新的SSL loss

4. **第一阶段至少做两个frozen-backbone指标**
   
   - 第一项建议做**drivable-surface segmentation**：冻结DINOv3，只训练轻量segmentation probe，比较原始DINOv3和续训DINOv3的IoU，nuImages原生提供drivable-surface mask，因此这个实验最干净
   
   - 第二项建议做**目标表征能力**：利用nuImages的2D box/instance mask/category，用完全相同的lightweight probe比较object classification或2D detection指标
   
   **SSL training loss下降不能作为项目成功指标**，最终判断必须来自frozen downstream evaluation

5. **第一阶段的Gate要非常明确**
   
   最终至少得到下面这张表：
   
   ```
   Original DINOv3->Driveable IoU/Object metric
   nuImages-adapted DINOv3->Driveable IoU/Object metric
   ```
   并保存若干中间checkpoint，观察downstream指标随SSL training step的变化，而不是只评最后一个checkpoint。只有确认续训后的representation有稳定提升，才进入多帧阶段；如果没有提升，先分析SSL continuation、数据、augmentation和feature degradation

6. **开发方式需要调整**
   OpenCode可以用于生成代码、查实现和辅助修改，但不能形成“给任务->agent改代码->agent起训练->看结果”的黑盒工作方式，必须能够：
   ```
   打开工程->跟数据流->看Dataset输出->看tensor shape->看forward/loss->打断点->检查梯度->看训练日志->独立执行train/eval->解释结果
   ```
   VSCode、Pycharm或其他IDE均可，重点是必须建立**可观察、可调试、可复现**的实验流程，每次让OpenCode修改代码后，都要能够说明“改了什么、为什么改、预期改变哪个指标”，否则不能直接提交大规模训练
   **第一阶段交付物只要求四样东西：实验设计文档、可视化正确的数据集、可复现的训练/评测命令、Original DINOv3 vs Adapted DINOv3的frozen-probe对比结果**