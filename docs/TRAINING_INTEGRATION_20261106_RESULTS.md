# 2026-11-06 真实训练接入实验结果

本轮用户授权继续训练集成与实验，已实际执行1次真实VLA反传技术门和4次独立规则专家试采。没有optimizer更新、没有触觉收益结论、没有改动原154800/185800权重或比赛源码。Strata始终未被本轮启动，收尾GPU无compute、租约free、服务inactive。

## 真实训练通路已验证

`src/probe_h50_training_backward.py` 使用原生 FeatureTransform、robotwin动作映射和归一化，把同回合50×14真实take_action标签映射到50×55并mask掉41个非真实维度。154800六分片逐SHA重新校验、strict装载1,624键；基座冻结，零初始化触觉state residual共65,664参数。

固定训练候选seed100023首个因果anchor（frame2/物理步89），训练模式原生flow-matching forward得到loss=1.0092004537582397；视觉与零adapter loss完全一致。真实backward后输出投影梯度L2=0.2569900155067444，有限；GRU梯度为0（零输出投影下第一步的预期现象），基座无梯度。CUDA peak allocated=14,701,534,720 bytes（13.69GiB），reserved=14,877,196,288 bytes。adapter参数/缓冲值未改变，optimizer_steps=0。报告中的forward_backward_seconds不含载入、预处理和前置恒等检查；不能当训练吞吐。

输入无接触信号：本结果验证可反传接线及显存，不证明触觉改善，也不批准训练数据。固定log1p/identity-scale只用于无参数更新的诊断，没有伪造train-only统计批准。

## 真实采集与独立源验证

| seed | 预登记用途/协议 | 结果 | 实际动作/物理步 | 有效完整H50窗口 | 其中接触输入窗口 |
|---|---|---|---|---|---|
|100023|既有train候选/stride20|历史成功，本轮重验|59/6808|8|0|
|100026|新validation/stride20|成功并完成同源HDF5|56/6528|5|0|
|100027|新validation/stride20|成功并完成同源HDF5|55/6217|5|0|
|100028|新train/stride10|setup_demo瓶子不稳定，拒收|未执行专家动作|0|0|
|100029|新validation/stride10|成功并完成同源HDF5|100/8092|50|0|

新成功回合独立复核raw/obs/frames/HDF5/actions/completion/success gate/pilot result八份SHA、真实动作物理时钟、原场景源码SHA、生成采集器源码及fold。未将validation改成train，未重跑失败seed，也未读取旧test图像/标签。stride10是输入覆盖诊断后预登记的新采集协议，不能用其与其他seed的差异宣称stride收益。

关键发现：seed100023/100026/100027/100029第一张有效接触观测分别位于动作索引40/31/35/67；从该观测到回合终止分别仅剩19/25/20/33条真实动作。完整H50窗口规则系统性排除了全部接触阶段。更密stride10没有解决。当前共68个完整H50候选窗口仍然全无过去正接触；继续拟合无法检验触觉贡献。

## 当前停止优化的原因与下一个实验

不是GPU、权限或真实模型反传问题，而是标签契约让接触阶段无法进入训练。本轮维持optimizer_steps=0，不将候选登记为正式获批训练集，也不生成虚假的adapter checkpoint。

建议下一实验保留模型H50输出，将真实回合尾部可用动作作为变长目标并显式mask终止后的位置，单独命名新数据/训练协议。原`LingbotVlaV2Policy.forward`接受`action_is_pad`却不直接用它计算loss；真正归约依赖`joint_mask`。不能只给`action_is_pad`就声称padding已被屏蔽。还须分析无效suffix tokens在flow-noise和attention中对有效动作的影响，避免训练/推理不一致或泄漏终止长度；不能只乘loss mask便跳过该检查。

验收必须包括：未来/无效目标不进入当前触觉输入；padding后的target/noise变化不污染有效监督的预期语义；真实动作维度与时长分母正确；无效位置零损失/梯度；seed隔离；正接触归一化仅从train拟合；匹配容量的视觉条件adapter、正确触觉、仅过去乱序触觉对照。此新协议尚未实现和验证，本报告不将它描述为可正式训练。

## 可复核产物

- 协议：`docs/TRAINING_INTEGRATION_20261106.md`
- 实测：`data/reports/h50_backward_20261106_v1.json` 及同名log
- 来源审核：`data/reports/h50_audit_{100026,100027,100029}_20261106.json`
- 接触时机：`data/reports/h50_contact_horizon_diagnosis_20261106.json`
- 采集器：`src/prepare_validation_batch_20261106.py`、`src/prepare_contact_h50_batch.py`
- 统一源审核/候选读取：`src/audit_h50_batch_20261106.py`
- 接触覆盖门：`src/h50_contact_readiness.py`

所有原始RGB/动作/权重仍在私有本地，未上传GitHub。本轮源码新增未提交，可审查后单独提交；实验报告/代码不能冒充完成了优化训练。
