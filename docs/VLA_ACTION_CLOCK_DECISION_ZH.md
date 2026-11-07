# 下一道科学决策：原VLA 50动作 chunk 的执行时钟，不能从专家逐物理步目标猜出

已确认[真实V2视觉/零触觉完整去噪恒等](<REAL_VLA_VISUAL_ZERO_NUMERIC_RESULT_ZH.md>)，但这只检验**未训练**触觉支路，不能跨越监督时钟门。原VLA `chunk_size=50`按LeRobot打包的50fps帧索引取未来50个`action`，不是每20ms真的发一次命令；旧本机规则专家12回合在严格同回合物理帧时钟审计下，[提锅H50训练／验证0／0、双瓶训练7且验证0](<PHYSICAL_CHUNK_READINESS_ZH.md>)。新一条relay同seed规则专家成功技术复演虽侧录1,582个连续物理步14D *post-step drive-target*，但只是[命令侧录验收](<EXPERT_COMMAND_KNOWN_SEED_SMOKE_RESULT_ZH.md>)，不等于50个实际高层动作。

源码执行语义不同：原评测客户端在每次`model.infer`后逐条调用[`TASK_ENV.take_action(act)`](</mnt/relaydata/lingbot/RoboTwin/script/eval_policy_client_lingbotvla.py.orig#L385-L403>)；原`take_action`用双臂关节轨迹的TOPP规划，每条动作持续的受测物理步数可变（本机视觉自回放真实100个执行间隔**17–181步，中位86.5步**）。但规则专家的[`Base_Task.move`](</mnt/relaydata/lingbot/RoboTwin/envs/_base_task.py#L884-L968>)按左右`Action`编排并走`together_move_to_pose`/`take_dense_action`，**不是**将专家的每15个保存帧drive-target或每5个受测物理步转成等价的一次`take_action`。因此“每5步抽一条×50”等只读覆盖数不是训练标签，旧H50候选样本不能被批准或用于真实触觉VLA拟合。

两个可审查选项：
1. **保持原视觉评测控制语义（建议）**：在隔离专家管线上增加高层**与`take_action`调用等价**的14D命令边界，并侧录调用前相机／状态／过去触觉、该次命令与`(before_physics_step,after_physics_step]`真实跨度；每个anchor仅取同回合后续50次真正控制边界的命令，哈希绑定成功门和HDF5。必须先证明规则专家的规划/动作确能改成同等控制动作而不改变成功条件；若不能，拒绝该路线，不借旧数据拼帧。需额外录多个独立物理seed供train/validation，固定heldout、不偷读测试。
2. **改成明确的新时钟任务**：把模型实验分支标为新的H16/每15个物理步保存帧监督，并强制推理客户端也按该节奏执行动作、单独拟动作归一化及重做视觉-only对照；这不再是原`take_action`/H50评测可直接比较的同一任务，不能宣称恢复或改进原154800基线。

**现状两项都未获负责人选择／数据批准。** 仓库`approved_paired_episodes=[]`，候选六份触觉norm `training_approval=false`；即使装载真VLA并得到零adapter数值恒等，也不应启动训练`forward/backward/optimizer`或开放`tactile_v1` RPC能力。资源方面Strata已恢复systemd active+enabled/loaded=true，未来仅在它空闲、持本人租约且脚本按协议恢复后才能有界再试；私有GitHub新提交仍待逐路径审核。
