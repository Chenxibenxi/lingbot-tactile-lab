# 触觉效用pilot v1：2026-11-06预登记

用户明确要求继续验证真实提升，并允许GPU训练、本机仿真及后续cwmj资产替换。本批使用固定原RoboTwin场景，保留原比赛源码/权重；任何新checkpoint只写私有models。Strata不自动启动。

## 数据与新的末尾H50契约

训练固定seed100023，validation固定100026/100027，均为已成功完成、同源真实take_action/相机/物理时钟的stride20专家。保留旧全部候选卡原样；本次用户授权记录只用于本pilot的新manifest，旧registry不自动推广。旧test不读，stride10回合不混入。单训练seed和两个验证seed仅支持可行性结论。

当前观测只含当前三摄RGB、14D状态及过去25物理步触觉。模型输出仍为50×55，其中14维为真实机器人动作；回合末尾只有K条真实动作时，loss只在前K×14上归约。其余位置使用冻结视觉模型仅依据当前观测、固定噪声、10步去噪生成的completion，detach并且完全排除监督loss。不得称这些补全为专家标签。填入未知位置的任意/NaN哨兵必须先被移除，不能影响实际forward。真实终止长度只控制loss，不作为模型观测或attention mask；推理使用完整50步、不传K或专家目标。这是冻结教师尾部插补假设，模型可能受补全分布影响；所有对照共享teacher，且最终评价必须使用纯当前观测sample_actions、不能用teacher-forcing loss冒充收益。

## 固定训练及评价

F=原冻结视觉基线；A=同容量无力adapter（mask/age保留，全部触觉历史特征为0）；B=正确过去触觉adapter；C=过去5个bin反序、mask同步反序adapter。每组相同初始化、train顺序、噪声/time、AdamW1e-4/weight_decay0.01、batch1、clip1、120优化步；基座全冻，adapter FP32、主模型BF16。仅train正接触bins拟合保零log1p median/IQR（floor0.1、clip4），不读validation拟合统计。该归一化新批准仅限本pilot，不改变旧候选。

验证每seed按过去接触分层，各均匀选择4个contact和4个noncontact anchor；指标：10步真实sample_actions的归一化动作MAE（前min(10,K)及全有效K）、按seed/接触层报告。相同seed内重叠窗口不算独立样本。训练后的B另做输入置零、过去反序干预；若C/B信号实际相同要报告，不能冒称负对照有效。

GPU门：原六片SHA严格复核、零adapter输出恒等、填充哨兵不变、无base梯度、有限loss/grad、峰≤22GiB、每组新adapter artifact及SHA。固定120步最后checkpoint评价，不看validation选步数或调超参。首批预算单GPU≤50分钟（超时退出），每60秒报告进度。未来闭环比较相同重规划10步和请求预算，成功/失败与规划初始化异常分开；本pilot离线通过也不自动声明成功率提升。

离线支持阈值预先定为：B在两个validation seed的接触层均比A/C低至少5% MAE，且置零/反序干预使B恶化；任何不满足均报告“尚未证明触觉提升”，不得靠换seed/挑指标逆转。即便满足，小样本仍须独立闭环复验。

## 独立闭环预登记（首次跑仿真前冻结）

新seed100030、100031，每seed各A（训练后的无力adapter）/B（正确触觉adapter）一次。无论离线结果是否有利，都保留这4次诊断结果，不重试筛成功。原`pick_diverse_bottles/demo_clean`、原400动作成功/终止规则，10步去噪、每10次真实take_action重规划、不做时间集成、相同请求编号噪声。单次1200秒/60000物理步的资源截断独立标记，不能当完整失败。setup不稳定单列init failure。新seed不做专家先筛；记录初始瓶ID/坐标来检验配对。原test专家资料仍不读取。

当前采用进程内真实model.sample_actions接线（非RPC服务上线）。输入/执行HDF5、物理步与infer marker由现有采集器记录，独立validator复核；B首个有效接触请求还做同噪声zero-input影子预测，证明训练adapter实际影响动作。影子输出不执行。全部rollout仅诊断，不作为专家监督回流。2个seed样本量不足以证明总体提升，至多提示方向；若没有改善如实报告。

### 初始化拒绝后的追加预登记

100031在A/B两组均于setup_demo抛出同一瓶不稳定异常，0动作，不计为策略失败。本批只追加下一个连续seed100032，A/B各一次，仍用同checkpoint/配置/预算；该决定仅由共同初始化拒绝触发，不因已有策略失败而替换场景。保留100030两组满400动作失败及100031两组拒绝记录；100032无论如何不再追加下一seed。


## 2026-11-06 infrastructure invalidation and v2 repeat

Cross-process identical-image/state, invalid-tactile initial actions differed by
0.0100854. Audit found 3 nonpersistent rotary buffers and 108 MoE counters
were not initialized by meta/to_empty + state_dict streaming (36 caches are None).
All v1 loaded-model training/effect and policy comparisons are invalidated for
scientific interpretation. Preserve their artifacts; do not select a winning arm.
Restore audited buffers from the installed rotary constructors and zero counters,
fail on unknown buffers, and require independent-process fixed-input equality.
Repeat the SAME data, initialization, 120 steps, three arms and evaluation under
v2 paths. Repeat seed 100030 zero/true once for corrected infrastructure; seeds
100031/32 already failed initialization in both arms and remain excluded. This
is an infrastructure correction, not outcome-selected tuning or reseeding.

Further audit: restoring buffers alone did not restore determinism. The optional
eval-only robby_moe_forward kernel produced max absolute action differences
0.03125 on repeat inference in one process. In this research process only,
disable that optional function and use the existing group_gemm path used in
training. No upstream source edit. This restored within-process exact equality
in run d. Require a second process before fitting. v2 was terminated during
loading, before any teacher/optimizer steps; preserve protocol as an aborted
infrastructure attempt. The corrected fit/closed-loop run is named v3.
A SHA-bound prerequisite in both runners now rejects a missing/failed gate.
