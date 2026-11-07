# 真正接入 LingBot-VLA 前的动作 chunk 时钟审计：不能把50帧说成1秒

用户授权开始真实触觉VLA接线后，第一道门应先核验动作监督能否对应模型的原始chunk。原 LingBot-VLA [配置](</mnt/relaydata/lingbot/runs/relay_p1/lingbotvla_cli.yaml>)使用`chunk_size=50`；[原数据读取](</mnt/relaydata/lingbot/lingbot-vla-v2/lingbotvla/data/vla_data/base_dataset.py#L232-L240>)按数据集`fps`生成`0..49`的索引偏移。旧LeRobot `fps=50`是**转换封装时标**，不能证明原仿真中每帧间隔20ms；[时序契约](<TIMELINE_DESIGN.md>)已禁止据此伪造物理时间。

本轮在CPU上只重验本机同回合成功专家的训练／验证折（不使用测试折来挑训练时长）：对保存帧输入`t`只考虑**同一个HDF5回合**未来`t+1..t+H`的已保存14D **drive-target命令**、配套每帧真实`physics_step`；要求当前25步触觉窗口有效、每个未来帧步号严格递增，并单独统计所有相邻帧差均为**15个物理步**的均匀窗口。原[源校验器](<../src/validate_expert_data.py>)逐帧核对HDF5`joint_action/vector`和保存帧命令及SHA；但物理接触raw每步只记录接触/phase，**没有每步完整14D命令**，所以现有记录不能无中生有地恢复250Hz执行动作或真正策略请求。

| 任务及折 | H=16正向窗口／其中全15步 | H=50正向窗口／其中全15步 |
|---|---:|---:|
| 提锅 train 3实际seed | 93／**77** | **0／0** |
| 提锅 validation 1seed | 32／**26** | **0／0** |
| 双瓶 v2 train 5seed | 192／**164** | 8／**7**，仅新增seed100015 |
| 双瓶 v2 validation 1seed | 44／**38** | **0／0** |

上述计数已用同源成功门/HDF5/物理帧SHA经[可复验CPU审计代码](<../src/physical_action_chunks.py>)重算，[脱敏审计小卡](<../datasets/relay_physical_action_chunk_audit_candidate_v1.json>)绑定私有逐行索引SHA；原始14D未来保存帧向量只在Git忽略的`data/reports/relay_physical_action_chunks_candidate_v1.jsonl`，绝不推送。

H=16且每间隔15步的候选跨度`16×15×约4ms≈0.96秒`；H=50若每间隔15步则约**3秒**而不是1秒。这是物理时钟定义下的**保存帧目标序列**，还不是原VLA真实控制频率/真实执行动作标签。**原H=50在两个任务都没有可用的训练+验证连续支持，禁止直接把已有1361个一帧对喂入50-step模型并称之为获批VLA训练。**若选择H=16，必须显式支持新chunk长度、只从train折重拟动作归一化和相机/触觉流转训练batch、测试旧ckpt完整严格加载与新长度的forward/sample_actions，以及在隔离环境验证实际控制步与输出序列一致；现有单3090未有真实梯度/显存证据。

**下一步按顺序**：只读CPU建立SHA绑定的chunk候选索引和重复步/跨回合负控→在隔离模型测试视觉关闭时完全恒等与真实批数据透传（不能仅AST假体）→如需真正每物理步指令，新增同回合`set_arm_joints/set_gripper`侧录与`scene.step`编号、实际模型infer请求标记，重新采样后再门禁→获明确训练数据/拆分/归一化批准且GPU lease空闲时才做单步adapter反传/显存烟测。现有公开registry `approved_paired_episodes=[]`、candidate norm approval=false保持原样，负例/自回放/LXC字节绝不充专家；不修改原LingBot/RoboTwin文件、不提前启动用户停机的Strata。
