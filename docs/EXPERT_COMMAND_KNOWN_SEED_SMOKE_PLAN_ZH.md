# 逐物理步命令侧录：已知成功seed的一次技术复演（运行前登记）

前一批预定的新独立seed提锅100017右臂规划失败、瓶100021物体初始化不稳，均已[独立登记失败](<COMMAND_TIMELINE_TWO_SEED_RESULT_ZH.md>)，不能换种子补作成功率。为检验新命令sidecar能否与**完整成功门/HDF5**一起通过，明确只允许一次**技术复演**：原先在本机中继机已成功的双瓶实际物理seed **100013**，新trial `data/paired/relay_pickdiverse_s100013_command_smoke_expert0/`；这个物理seed已在旧双瓶v2候选训练折，**不是新的独立种子、不加入旧1361对索引或训练/验证/测试、不据其估计专家成功率**。即使复演失败也原样记录，不再换种子。

执行前确认原RoboTwin `_base_task.py` SHA `448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101`、本机唯一RTX3090、无compute进程、Strata仍按用户窗口停机、离北京时间2026-11-02 12:00恢复时刻大于900秒；取得自己owner的interactive GPU lease TTL900，一次`timeout 660s`执行原 relay 专家runner，环境仅本trial私有`TACTILE_EXPERT_COMMANDS_DIR=<trial>/commands`，任何退出都释放自己租约且不触碰他人GPU/原基线。成功只在规划+官方目标、原同回合四源+HDF5 SHA、命令逐步流+正步号保存帧命令核对、双哈希成功门和completion均通过后**列为诊断通过**；不向正式成功数据registry注册，不更改公开候选split/norm `training_approval=false`。

后续另核真实视觉策略100个已执行`take_action`前后物理步跨度 min17/median86.5/max181，故即便成功复演得到每4ms命令，仍不能声称旧chunk50的动作控制间隔固定、或直接从这一个同seed回合推出触觉VLA训练有效；应先规定控制时钟、取得更多不同seed接触恢复/失败可学事件，再申请真实模型单步梯度/显存门禁。
