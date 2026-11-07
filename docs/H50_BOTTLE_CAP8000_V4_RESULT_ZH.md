# 8,000步技术上限单次验证：同seed控制在第56次动作/6,319步达到任务目标，但不构成训练专家数据

按用户[单次8,000物理步技术试验预登记](<H50_BOTTLE_CAP8000_V4_PLAN_ZH.md>)，用户先在工作台亲自停止GPU后端；代理独立复核唯一中继机RTX3090无compute、free≈23.9GiB、租约free，原RoboTwin源码SHA`448f7152...`和v3阴性诊断私有raw/obs/frames/calls/geometry/result**六SHA**仍保持，v4路径尚未存在。持交互租约TTL1500，以原seed100013/原**64个**SHA冻结的qpos路标，通过真正`Base_Task.take_action(qpos)`产生**新回合**物理/图像/瓶位：完整可观测控制器在第**56**次动作、物理步**6,319**时早期到达原任务`check_success`，`technical_result.goal_success_reported=true`、`goal_check_after_run=true`，从未逼近8,000的硬上限；64提议中后**8个并未执行**。新结果前54个已执行动作和各步后置双瓶记录逐条与旧6,000步失败v3**完全相同**，可定位超原上限后两步带来的同seed任务门转变；这是同一固定seed的技术重复，**不是第2个独立成功率样本**。

逐项任务目标：第55条/步6,201左瓶已入xy、高度门，右瓶y仍误差+0.13064m；第56条/步6,319左瓶x误差约+0.000049m、y约+0.072996m、z≈0.998464m，右瓶x约+0.005529m、y约**+0.099968663m**、z≈1.009482m，均刚满足原`|Δx|<0.1m`、`|Δy|<0.1m`且`z>0.89m`。尤其右瓶y只比严格0.1m门内缩约**0.0000313m（0.0313毫米）**：一次同seed贴边成功**不说明鲁棒性、独立seed成功率、夹力安全或改进后的VLA策略**，不能通过事后改阈值抬高结论。

[脱敏成功技术卡](<../datasets/relay_known_seed_h50_cap8000_technical_positive_v1.json>)逐SHA绑定新回合真实raw/obs/frames/calls/瓶功能点/result六个私有源；不把14D命令、RGB、世界坐标流或HDF5写入Git。这次`status=mechanical_pilot_not_expert_dataset`，**没有专家HDF5/completion、获批H50窗口、独立专家seed、训练审批、adapter反传或真实RPC**；正结果仅证明原qpos提议可以在允许更多物理时间时经过真实控制器达到已知seed的任务门，**绝不可据此启动VLA训练或宣称触觉带来提升**。下一阶段若要训练，仍需独立成功专家回合/同回合action+RGB+触觉严格对齐、train/val/test种子隔离、显式数据和norm审核，并另行取得用户授权。

试验子进程退出后、仍持自身GPU租约且无残留实验计算时，按用户“批次末再拉起”的要求仅一次调用原**工作台**`:8077/api/strata/start`，经`:8080/health loaded=true`才释放租约；`strata.service` unit仍`enabled/inactive`（因为当前模型归工作台管理），不能将unit inactive误报为GPU未占用。收尾复核已确认原工作台报告`running=true`、Strata health `loaded=true`、本人lease显示free；未来若再占卡仍须重新核实时状态，不凭此结果卡外推。
