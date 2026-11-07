# 下一批专家数据：逐物理步14维驱动目标侧录契约（尚未真实采集）

[动作chunk时钟审计](<PHYSICAL_CHUNK_READINESS_ZH.md>)发现既有本机成功专家只有已存相机帧的14D驱动目标，原LingBot 50步训练／验证均无充分可证的统一物理时钟窗口。**不能给旧12条轨迹补造每步命令。**因此只为**今后的全新独立relay专家回合**增加`TACTILE_EXPERT_COMMANDS_DIR=<trial>/commands`可选侧录；原RoboTwin/_base_task.py不改，现有ForceProbe样本及完成门在未开启此开关时字节/语义不变。

- 原隔离专家overlay已在**两类受测专家运动循环**的每次`scene.step()`后调用`ForceProbe.sample`（不宣称覆盖其他未插桩的仿真步骤）；opt-in后它在同一个`physics_step=1..N`写两臂`robot.get_left_arm_jointState()`与`get_right_arm_jointState()`各7维拼14D，输出`<trial>/commands/<rawstem>.commands.jsonl`（仅本机Git忽略）。这是**执行该仿真步后的drive-target／规范化夹爪命令**，不是机械关节实测qpos、力传感器、policy infer请求或已训练模型动作。初始step0的相机图片没有新侧录，应保留缺失，不能补填。
- 新[sidecar验证器](<../src/expert_command_timeline.py>)要求raw/frames/commands同一任务／seed／episode／时间步／严格连续物理步、每一正物理步保存帧对应命令±1e−5一致、所有14D有限且夹爪[0,1]、commands与原force为同trial路径。新专家成功门及`completion.json`另外绑定commands SHA，原HDF5/相机/力流的四SHA依然必须经过[原校验器](<../src/validate_expert_data.py>)；侧录缺失、步数不等、哈希或帧内容错即拒，新旧数据不会混洗。
- 只在原`src/run_expert_relay.sh --run`的新trial执行前外部显式提供**绝对路径**`TACTILE_EXPERT_COMMANDS_DIR=/mnt/relaydata/lingbot-tactile-lab/data/paired/<new-unique-trial>/commands`，且`FORCE_PROBE_EXPERT_DIR=<same-trial>/force`、`TACTILE_SOURCE_MACHINE_ROLE=relay_single_rtx3090`。命令未设置时不产生新sidecar。实际GPU采集必须先独立租约、检查Strata/进程、固定种子和最多一次试次；此文档/CPU合成测试**没有新GPU专家样本，也没有改变现有1361对候选或任何训练批准**。

**仍须证明的下一步**：检查每步drive-target与真实control API执行顺序、相机锚点和未来动作chunk的实际控制频率；按预登记新物理种子采得真实完整侧录、过成功/HDF5/commands全部SHA门后，还须分别量化**专家逐步drive-target更新周期**与**评测`take_action()`一次预测命令实际推进的物理步数**，再选择可持续覆盖的H和训练专用动作归一化；不能凭LeRobot旧50fps推定一次policy action等于5或15个physics ticks。24GB单3090适配器反传/服务与原H50视觉基座兼容仍未实测；不得因“每步有一条命令”就声称现有视觉VLA可以直接部署触觉。
