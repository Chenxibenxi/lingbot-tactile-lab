# LingBot-VLA 触觉输入研究路线（用户 2026-10-31 决策）

## 目标重新明确

**沿用并改进**现有 PhiZJU LingBot-VLA + RoboTwin 的数据、训练和评测管线，触觉最终必须作为输入进入训练和闭环推理；不是永久停留在只读日志。保留视觉-only 185800 checkpoint、原始脚本/成绩及可重复运行配置，供科学对照与随时回退；所有新开发在 opt-in 实验分支逐步接入，成熟后合并到通用管线。产出可向 UIUC Lourentzou 实验室展示的研究资产，而非只做一次单任务 demo。

## 第一程：同源示范数据（relay3090首个成功候选已采；多独立seed仍待完成）

1. **专家实时触碰的数据必须与其图像来自同一次 episode**。仅同一个 seed、同一个任务而另跑一次仍可能规划不同，不能相拼。本仓库 `src/make_expert_overlay.py` 在 expert 的 `together_move_to_pose` / `take_dense_action` 每物理步采 contact；`_take_picture` 成功写出 pkl 后同步记录真实 `(frame_index, physics_step, actual_seed, actual_qpos/qvel, command_state14)`，HDF5 视频沿用现成生成器。
2. 拒收未知时钟、探针失效、掉步、HDF5帧数/命令数组与 marker 不一致、任务不成功、状态不完整的产物；仅完成验证才原子写 `completion.json`。旧 HDF5 没时间戳，旧 LeRobot 50fps 是合成时间，不能用来当触觉标签。
3. **首个 relay `lift_pot` seed100048 同源专家回合已成功并hash验证**：1,514物理步、113帧、105个通过历史完整/正物理间隔门的候选训练对，见 `datasets/cards/relay_liftpot_s100048_expert1.json`；仍需至少5个独立seed才能做真正train/validation/test分割并只在train拟合归一化，不能对单seed模型声称泛化。随后加入小物抓取/接触操作、空夹与不同物体姿态/材质负例；失败示范与一次后处理故障的隔离轨迹仅供分析，不与合格同源专家训练对混用。记录碰撞冲量尖峰/采样 p95 和磁盘体积。既有 encoder 是 RGB，对应 SAPIEN scalar force 不是 GelSight 图像。

## 第二程：可复现数据集转化与模型输入

- 在实验版 `convert_robotwin_lerobot` 加训练集每帧 `observation.tactile.history` `[5,4,3]`、`observation.tactile.mask` `[5,4]`、age/dt/quality 和原始 episode UID/seed。严格使用同一次 frame marker 的物理步 T 对齐最近 `[T-24,T]`；缺帧留 mask；**视觉像素 HDF5 index 对齐该 marker**，首帧窗口无历史是 invalid，但不默默视作零力。
- 原动作依然是左右 6 关节+1 夹爪的 **14D**，旧模型 YAML 的 55 是统一 padding 维度，不是触觉空位。`state[t] → action[t]=state[t+1]` 保持旧标签规则；额外预测接触/代理力趋势的未来标签只作 auxiliary target，绝不拼进当前模型输入。
- 同一采样任务按原 seed/物体身份和 episode 分 train/val/test，不共享有重叠片段或同 seed 不同重跑；影像频率由原始物理步决定，而非盲称视频50Hz。每例 manifest 保存 RoboTwin/采集器版本、urdf/robot/config哈希、dt、norm stats 及 split，实验可以在另一台机器独立复核。

## 第三程：真正接入 LingBot-VLA 训练及推理

- 第一次验证：冻结现有视觉/语言主干及 action head 主权重，训练小型时序触觉 encoder（60标量 + 20 mask + age）到 **零初始化 gated adapter**，接进当前策略的状态条件或动作头；门控为 0 时要求输出与视觉版数值一致，传感器缺失自动退视觉-only；模型不再只看图像。若状态token/网络API原本只支持14D，单独传模态字段而不滥用 padding。
- 第二次验证：在融合有效后，解冻部分 LoRA/action head，加入短期 contact-onset / load-trend 辅助 loss 防模型忽略触觉；按训练集拟合数值归一化、稳定性和物理步延迟，在验证集中查反事实（打乱时间/替换另一物体触觉）后触觉贡献是否消失。
- 第三次：服务器与仿真客户端各加可开关的 `observation.tactile` 传输/校验；不能仅在 `TASK_ENV.get_obs` 放键却不传给 `model.infer`。更长远再尝试 GelSight-like 模拟与真实传感器数据，逐步接近 VTAM 的视觉—触觉联合动态建模；当前 SAPIEN scalar force 仍不等于 GelSight。

## 第四程：受控夹爪反馈和科学评测

- 在仿真中增加单独低层 250Hz 安全/夹爪 residual 控制（不让慢速 VLA 直接做高频安全回路）；触觉模型与低层反馈要独立开关，做 2×2：视觉/触觉模型 × 无/有反馈。
- 与原 185800 视觉版以及**同样新数据训练的视觉-only 对照**逐任务、同 seed、同物体条件比较；指标不止 success，还包括接触首帧误差、漏抓/掉落/夹碎代理、相机遮挡鲁棒性、部署延迟和磁盘/GPU 成本。加入触觉时序打乱负对照，优先验证脆物/遮挡/滑移类任务，随机背景和材质分别分层。大样本成功率统计才可解释。
- 实机之前须独立传感器标定、力/速度上限、掉线回退和机械急停；SAPIEN 穿模尖峰和论文从 GelSight 光流导出的“虚拟力”都不是可直接移植的实机夹持阈值。

## 对外研究资产包

每阶段都留：可运行的隔离补丁与单测、同源时钟映射和数据字典、失败/成功视频（脱敏）、train/val/test manifest、基线与触觉消融的图表、真实显存/延迟、不能复现的反例及其原因。论文参考 [VTAM](https://arxiv.org/abs/2603.23481)；明确列出我们与论文的传感器/机械臂/控制频率差异，不借其 90% 作为本模型预期成绩。
