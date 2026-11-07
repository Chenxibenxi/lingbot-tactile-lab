# H50 原控制语义：旧专家逐步目标仅可提“未来受控回放路标”，仍是0个训练动作

用户已选择保持原`Base_Task.take_action` H50 控制语义。[同回合动作门](<H50_POLICY_CONTROLLER_CONTRACT_ZH.md>)要求真实高层调用的连续14D输入和每次可变的(before,after]物理区间；不能把技术回合规则专家的高频drive-target误记为这些调用。本轮只在CPU上对[原同seed成功技术复演](<EXPERT_COMMAND_KNOWN_SEED_SMOKE_RESULT_ZH.md>)执行其原raw/frames/command三流同回合校验（1,582步，专家6规划阶段，`take_action`调用**0**）：[只读路标生成器](<../src/technical_take_action_waypoints.py>)以**预先固定25受测物理步**抽取候选14D qpos末端，并包含最后一步，得到**64个路标提案**。仅[脱敏可行性卡](<../datasets/relay_known_seed_h50_waypoint_feasibility_v1.json>)保存64这个数量和原流SHA；私有14D命令未入Git。3项[合成负控](<../tests/test_technical_take_action_waypoints.py>)拒缺步/乱序/错误阶段/非有限命令/越界夹爪/随意改步长，输出持续标`actual_take_action_calls=0`、`training_approval=false`、`known_seed_replay_is_independent_seed=false`。

**64不是现成H50样本，更不是下一次真实执行也会得到64次。** `take_action`内部TOPP变时长并每步检查成功，控制路径从原规划器的dense set_arm_joints/together移动切换到高层qpos，可能重规划变慢、夹持失败或提前`eval_success`停掉；选25ticks只是**提案的原始路标间距**，不为部署设25tick动作频率。把旧回合路标输入新回合来做**已知seed技术pilot**只能测试控制可行性，**不得**把旧force、HDF5或RGB贴到新回合，也不得把同seed复演重复计独立专家seed。

若继续试，须先在独立、未覆盖原RoboTwin的overlay把路标当**预定目标**，由新回合的原`Base_Task.take_action(qpos)`逐条实际执行，每次在**同一新回合**记录调用前真实相机/状态及过去触觉、调用14D输入与before/after物理tick，来源SHA和成功门在完成后复核；若控制失败、无法得到≥50条连续有效实际调用或同回合相机/触觉缺失，应标负结果并停。只有机械pilot过关，再规划多个**新实际seed**的训练/验证和冻测试集、train-only动作/触觉归一化与负责人明确批准。当前H50可训练窗口仍为**0份已批准**，`models/`无真触觉adapter权重，绝不因真实VLA零残差前向成功或这64个路标就启动训练／RPC／增益声明。Strata本轮未停、GPU未用。
