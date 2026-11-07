# 两个预先冻结的中继机逐步命令侧录试次（运行前登记）

用途只为验证[逐物理步驱动目标侧录](<EXPERT_COMMAND_TIMELINE_CONTRACT_ZH.md>)真实覆盖与相机／接触同一时间轴，而不是在原专家成功回合事后补命令、增加已批准训练数据或虚报模型有效。**在采集前固定且每seed仅尝试一次**：`lift_pot`物理seed100017、输出`data/paired/relay_liftpot_s100017_commandpilot_expert0/`；`pick_diverse_bottles`物理seed100021、输出`data/paired/relay_pickdiverse_s100021_commandpilot_expert0/`。不按成功结果替换种子，失败／转换中断保留单独诊断，成功也仅为未分折候选；旧提锅3/1/1、双瓶v2 5/1/1和私有v1/v2索引原样冻结。

[固定两次执行器](<../src/run_command_timeline_pilot.sh>)的`--dry-run`不创建数据或GPU工作；`--run`每次前独立查本机单3090、原RoboTwin base_task SHA、GPU租约与所有真实compute进程、Strata未加载以及距用户定于北京时间2026-11-02 12:00恢复≥15分钟。每次申请interactive租约TTL900、硬超时660秒，退出时仅释放**本执行者**租约；不杀别人的进程，不提前恢复或延后用户的Strata。标记`TACTILE_EXPERT_COMMANDS_DIR=<trial>/commands`仅在本次专家子进程有效；runner常规同源成功/HDF5/力流校验之外新命令sidecar逐步与帧一致且SHA绑定成功门／completion，才有候选。原数据、源仓、权重及Git之外的私有视频/HDF5不复制/上传。

事后只比较同回合实际保存帧锚点后250物理步内能否从**每步已记录**drive-target取固定间隔的50个动作，必须另核原部署`take_action()`每个预测命令到底推进多少物理步。即使形成50×14D目标，也**不是原LingBot-VLA实际训练、策略请求记录或获批模型能力**；传感器仍是SAPIEN冲量代理而非GelSight／硬件力上限。
