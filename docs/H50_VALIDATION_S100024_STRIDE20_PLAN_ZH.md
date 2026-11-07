# 用户批准唯一seed100024的H50独立验证折试采（GPU启动前预登记）

已有本机[新seed100023在线专家59次真动作/8个H50窗口](<H50_ONLINE_EXPERT_STRIDE20_S100023_RESULT_ZH.md>)经原相机HDF5、成功门、触觉过去25物理步、私有动作sidecar八SHA重验，但八窗口仅来自**同一物理seed**，只可作未批准的train意图，不能作验证。用户**另行明确批准一次**新独立seed**100024**（本repo data无旧试次）作**validation-only**，使用事先固定的在线规则`stride=20`（规划路径索引，不是物理步），物理上限**10,000**（单条TOPP不可拆，越界动作结束即停）、墙钟**900秒**、最多128条真`Base_Task.take_action(qpos)`，独立新目录`data/paired/relay_pickdiverse_s100024_online_h50_stride20_validation_expert0/`。**不根据此验证回合结果调步长/选择模型超参、绝不挪入train或同seed换ID重试**。旧100022短回合、100023训练意图回合、原规则旧dense数据均保持不变。

`src/patch_online_expert_validation_pilot.py`从原seed100022收集器和预登记器的**固定SHA**生成仅在Git忽略`work/online_expert_stride20_s100024_validation_v1/`的隔离副本；只替换seed、独立ID、`fold_intent=validation_candidate_only_not_assigned`、独占GPU租约owner和显式stride20调用，原RoboTwin Base_Task、双瓶任务及规划器三SHA仍逐件复核。预飞必须确认CPU全套负控通过、trial/lock/status均不存在、工作台GPU后端由用户亲自停止、3090唯一卡实测无任何compute且free≥16GiB、本人独占TTL1500租约无其他waiter，外层timeout900；不得杀他人的进程或在Strata仍占卡时抢GPU。仅在用户本批**明确说已经停止**后才能run一次；成功或失败皆先审原目标/是否≥50条、同回合raw/obs/frames/HDF5/completion/私有高层动作SHA；若成功且有可用anchor也只是**未批准validation候选**，不自动登记训练、fit norm/adapter或根据它选新动作设计。结束按用户偏好在持租约条件下恢复工作台Strata一次，再release；如有瞬时别的GPU计算进程，拒恢复后待独立占卡清零、重新lease再做一次恢复，不误报成功。

即使这条val成功，训练折仍只有**一个物理seed**的8个相关窗口，旧双瓶逐帧归一化无H50训练批准；[真实3090 LingBot-VLA小adapter训练计划](<H50_REAL_GPU_TRAINING_PLAN_ZH.md>)要求扩充独立train来源、冻结新fold/train-only统计逐件审核、真实完整VLA BF16 forward/backward显存门和同数据视觉/触觉/乱触觉对照。用户此次批准**只覆盖这一个validation采集**，绝不自动开始GPU训练、修改旧权重、推GitHub或发布触觉收益。

**事后更新**：本次真实val seed100024完成80条动作却任务目标false，故没有验证专家HDF5/完成卡，见[结果及Strata自动恢复纠错](<H50_VALIDATION_S100024_RESULT_ZH.md>)。上文“结束恢复一次”只是执行前旧约定；用户最新明确要求以后**不再自动拉起**，以后全部以[新GPU收尾规则](<GPU_NO_AUTORESTORE_POLICY_ZH.md>)为准。
