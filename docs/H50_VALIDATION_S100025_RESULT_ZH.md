# seed100025验证试采：双瓶初始状态不稳定，未进入专家动作，验证H50仍为0

用户单独批准[一条全新validation-only seed100025/stride20/cap10000/900秒试采](<H50_VALIDATION_S100025_PLAN_ZH.md>)，并**亲自**停止工作台GPU后端。实际运行前核本机唯一RTX3090/free23721MiB、无计算进程，工作台`running=false`、Strata systemd unit`inactive`、独占GPU租约free、本Git HEAD clean、trial/lock/status不存在，固定原三RoboTwin源码SHA和生成的两个私有runner SHA。只运行**一次**新ID，没有重播前一失败seed100024，也没有触碰旧训练折。

在原RoboTwin `robot.setup_demo(seed=100025)`初始化时，`_base_task._init_task_env_`直接抛出`UnStableError: Objects is unstable in seed(100025), unstable objects: 001_bottle`；发生于在线专家`execute_online_expert`之前。试次只留下私有`preregistration.json`与`failure.json`（记录异常类型/seed/`physics_steps:null`）；**没有执行任何一次`take_action`，没有可核的物理步数、raw/obs/frames、在线动作、专家HDF5或完成卡**。不能把“未知物理步数”偷换成0；不能从这个环境初始化异常推断真实VLA视觉/触觉效果。该seed已尝试，禁止同seed改路径重试，**验证H50候选0**。前一[seed100024](<H50_VALIDATION_S100024_RESULT_ZH.md>)有80真动作但任务目标false，故两条都不能补验证折；seed100023只有同一物理种子的8个未批准train意图窗。训练/归一化/同数据对照均未获数据门禁通过。

GPU作业退出码1，私有wrapper按最新[不自动恢复Strata规则](<GPU_NO_AUTORESTORE_POLICY_ZH.md>)只报告状态并释放本作业GPU租约；事后复核lease **free**、Strata `inactive/MainPID=0`、无GPU计算进程，**没有自动拉起Strata，也不擅自拉起**。两私有数据文件和两个生成代码文件的SHA记录于[脱敏阴性卡](<../datasets/relay_online_validation_s100025_unstable_v1.json>)；只有脱敏报告入Git。今后任何新seed都必须重新由用户审批，不能把此前的一条批准当成无限次可选种子重试。
