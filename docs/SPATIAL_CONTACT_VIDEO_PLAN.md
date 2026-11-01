# 迈向真正手指接触点热图：现有证据与缺失门槛

用户要像结构受力分析一样用颜色显示手部触点。现有[并排回放视频](<HAND_CONTACT_VIDEO.md>)已按四根夹爪指连杆显示统一色标，但不声称画面上的精确触点位置。**绝不能根据指连杆名称、颜色条、视觉猜测或同 seed 的另一试次凭空制造逐像素受力图。**

## 已有真实数据：相机标定齐全、接触位置缺失

本机中继机已验证的开爪失败回合 HDF5 共有131帧，**每一帧**三路相机均含同源 `intrinsic_cv` (3×3)、`extrinsic_cv` (3×4)、`cam2world_gl` (4×4) 与RGB。CPU-only `src/project_contact_geometry.py`逐帧查形状、有限性、相机外参旋转及视轴投影：头相机不动，左右手腕相机外参平移分量最大变化分别约0.250米／0.265米，**不得把首帧标定沿用全片**。可复现聚合仅在Git忽略的 `data/reports/controlled_failure_camera_preflight.json`。

但现有 ForceProbe `src/force_probe.py` 的原始JSONL只记每根手指的非零接触点**数量、物体名和冲量大小和**，**不记每个触点的世界坐标**；现有HDF5也没有逐帧深度图。故此回合完全没有足够证据做像素级手指触点热图。`project_world_contact()`只是对**未来真实记录的世界触点**使用OpenCV世界到相机外参做像素/深度**视锥内预检**；没有实测触点就不能绘点，即便落在画幅内也不能据此断言它未被遮挡。

## 下一次**另行预登记**的独立仿真采集应满足

1. 只在 `scene.step()` 完成后可选地采集**非零接触冲量**的每个接触点世界坐标、具体指连杆、该点冲量大小/真实 `scene.get_timestep()`、物理步号；不可混另一回合，不可用肉眼估坐标。写入**独立、只作事后可视化的私有 sidecar**，不附加到 `get_obs()`／模型RPC、不更改动作或成功判据。现有 E009/E010 raw/obs/frames 的 schema 及成功门不得静默重解释。
2. 侧录逐物理步和同回合帧标记/相机标定 SHA 绑定，拒缺步／重试混流／不一致的 seed、host、task 或 episode。相机投影必须选对应拍照帧 `extrinsic_cv` 和 `intrinsic_cv`；深度≤0或图像外不画。未来点不得进入当前时刻颜色。
3. 如果希望称为“画面上可见的触点”，同一捕获还必须有深度图、物体/指头分割或其它可靠遮挡判据；否则叠加图须标明**三维触点的二维投影、遮挡不确定**。颜色继续共用已披露的接触冲量/步长代理色标；它不是标定真实夹持法向力、不得驱动硬件限力。
4. 独立新试仍须GPU空闲/租约和用户规定的Strata恢复时限、原RoboTwin只读。原试次成功和失败的视频不得重绘成拥有本次不存在的“真触点”；新的原始RGB、三维点、HDF5及视频继续只存本机Git忽略目录，GitHub只可能放脱敏方法与统计。

**当前代码准备：**`src/force_probe.py`新增明确的环境变量 `TACTILE_CONTACT_GEOMETRY_DIR=<同一trial绝对路径>/geometry`，必须为原`trial/force`的兄弟目录且只对**未来新回合**启用。在现有`scene.get_contacts()`同一次采样中只对非零手指接触另存 `physics_step`、`side`、`finger`、SAPIEN `point.position` 世界坐标和`|impulse|/dt`，没有世界坐标就让采集健康门失败；JSONL侧录和原raw/obs/frames分目录、默认关闭，`observation()`及模型RPC不返回任何坐标。CPU合成测试已核可选输出、零接触、错误目录与缺坐标拒绝；另用**显式 PhysX CPU-only** SAPIEN 3.0.0b1 盒接触场景实际验证100步中的97个接触物理步，共388个带真实世界坐标的点，z≈0.200m，对应同一仿真盒顶部（私有聚合在`data/reports/sapien_cpu_world_contact_smoke_v1.json`）；这不是机器人夹爪试次，也不是物理力标定。默认`sapien.Scene()`会初始化Vulkan渲染器，在隐藏GPU的headless CPU烟测曾崩溃；已把smoke和probe benchmark限定只建 PhysX CPU System，不涉及机器人真实环境。新增独立的 `src/validate_contact_geometry.py`：先重验本机成功候选或隔离负例的旧来源卡/HDF5，再要求私有几何sidecar的header与raw任务seed回合、时钟、basename匹配，**逐物理步／逐指**点数与每点载荷代理和都等于原raw；对应帧标记的同一物理步才可以投到逐帧相机K/E，只输出**视锥内候选计数**和SHA小卡（不含三维坐标／图像）。合成回合门禁已通过缺步、假坐标、错力、卡篡改和step0未来接触负控；旧seed100000失败试次**因根本无几何sidecar而按预期拒收**。这仅验证工具逻辑，尚无真实机器人侧录和可见性证据，不能称像素热图。

**当前状态：相机投影、私有可选几何采样及同源校验工具已过CPU单测；没有新回合真实空间触点，未制作像素级热图，也未训练触觉模型。**
