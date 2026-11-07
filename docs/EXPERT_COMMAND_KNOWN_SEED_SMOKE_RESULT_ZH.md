# 已知成功seed技术复演：新侧录与成功专家HDF5同回合验收通过，但不是新独立数据

按[试前固定计划](<EXPERT_COMMAND_KNOWN_SEED_SMOKE_PLAN_ZH.md>)只复演本机旧双瓶专家成功物理seed100013一次；规则专家规划和目标检查通过，单张RTX3090独立租约获取／退出释放，Strata按用户停机窗口仍inactive+disabled，原RoboTwin `_base_task.py` SHA不变。此次新回合计**1,582个受测专家物理步，120个相机保存帧**，`dt≈4ms`；每步左右各7维drive-target与原force流连续相同编号，119个正步号保存帧目标与HDF5/sidecar相等，step0首帧明确无instrumented命令。原专家成功门/HDF5/raw/obs/frames及额外command sidecar SHA在新的completion里全部复验。来源哈希仅在[脱敏技术烟测卡](<../datasets/expansion_diagnostics/relay_known_seed_command_smoke_v1.json>)公开；逐步命令、RGB、视频/HDF5都在Git忽略`data/paired/relay_pickdiverse_s100013_command_smoke_expert0/`。

**不许拼回旧same-seed回合：**此前同一个实际seed100013已登记旧专家卡是1,609步／121帧，与本次1,582步／120帧轨迹不同。虽然seed相同，每次专家规划与执行并不恒等；因此新的命令不能贴到旧HDF5、不能冒称新独立训练物理seed，也没有注册成功专家卡、改旧v1/v2切分、改变1361对私有训练候选索引或转`training_approval=true`。前两次新seed失败仍单独如实登记，技术复演不算它们的替代成功样本。

**还不能直接训原LingBot-VLA：**本侧录证明未来新回合“保存帧+每受测物理步drive-target”的数据链可行，但原视觉自回放100个真实`take_action`的前后步差为17–181、中位86.5个仿真步；原模型H50从封装50fps按帧偏移，不能未经验证就从每4ms侧录固定抽样凑50步。作为**只读覆盖算术、绝非选定训练频率**：若任意假设每5物理步采1条，50动作尚有99个保存帧锚点；假设每50步采1条则零锚点，而真实策略每动作推进17–181步变化很大。不同假设的结论相反，必须先测真实控制语义，不能把“99”宣传为现成H50样本。下一门先严格定义模型每一预测命令相对相机的实际作用间隔与动作chunk训练目标，配置再生成train-only归一化和时间掩码；必须留出更多**不同实际seed**验证和困难接触转变，再在批准/显存/租约门下做真正VLA adapter前向与反向。仓库现有代理MLP负结果不能代替这个测试。
