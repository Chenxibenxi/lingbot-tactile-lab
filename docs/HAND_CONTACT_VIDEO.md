# 受控空夹的手指接触代理着色视频

**私有本地视频：**`data/reports/controlled_open_grasp_hand_force_analysis.mp4`。7.04 秒、1600×920、25fps、H.264，无音轨；画面左侧为同种子成功专家，右侧是**两次抓取均保持张开的失败试次**，并非新一次仿真实验或实时力传感器。视频和独立的 SHA 清单及预览图都在 Git 忽略的 `data/reports/`，不会上传图片、视频或 HDF5。

四根指连杆 `fl_link7/8`、`fr_link7/8` 各自读取同一次运行的 SAPIEN 物理步接触数据，取对应 RGB 拍照步之前（含当前）的 **25 步 ≈100 ms 代理峰值**作手腕图像上的着色标签，另列拍照步瞬时值。共用的**对数色标**蓝0→青0.05→绿0.5→黄5→红≥50，不随角色或每一帧自动重新归一化，避免把失败试次的微小擦碰放大成红色承力。成功回合指连杆峰值约 41–47 代理单位；开爪失败峰值至多0.511。下方全程接触时间轴为**事后分析**，不能作为在线模型输入。

拍照和物理采样频率不同：每个25fps播放时刻，画面只取其**不晚于当前物理时间**的最新真实图像标记，颜色严格取**所显示拍照步**的同源历史，不使用未来力。25fps是播放速率，**不是**原始相机固定采样频率；两个回合开始一致、后续动作不同，不可把后续两列的帧号当同一动作阶段。

> **不可误读：**这里的数是 SAPIEN 接触冲量除以物理步长的 *load proxy*，并非经实物传感器标定的夹持法向力，更不能用作硬件限力。原始遥测未保存能投影进相机的三维触点坐标，因此颜色对应**指连杆标签**而非画面中的精确触点像素；从本视频不能推出触觉 VLA 已训练或闭环成功率。色阶只用于这组实验的直观比较。

本机从经过 SHA 验收的正负同 seed 数据重生成：

```bash
cd /mnt/relaydata/lingbot-tactile-lab
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD" /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -B src/render_hand_force_video.py --repo "$PWD" --out data/reports/another_local_hand_force_video.mp4 --fps 25
```

实现见 [`render_hand_force_video.py`](<../src/render_hand_force_video.py>)，来源与结果见[空夹试片结果](<CONTROLLED_OPEN_GRASP_RESULT.md>)。重生成需要同机原始HDF5及接触流；只读使用CPU，不申请GPU租约、不启动Strata。这条旧视频的HDF5虽有逐帧相机标定，却缺三维触点坐标和遮挡验证，详见[空间触点补采门槛](<SPATIAL_CONTACT_VIDEO_PLAN.md>)；不得把它改口称精确触点图。后续另一条独立 seed100007 新试次确实另录了三维点并做[遮挡未知的二维投影回放](<SPATIAL_OPEN_GRASP_RESULT.md>)，不回填本条旧试，也不宣称可见力地图。
