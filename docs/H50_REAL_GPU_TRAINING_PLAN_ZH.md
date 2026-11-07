> 2026-11-06 最终更新：修复加载缓冲区和推理一致性后，v3 已完成三组×120步真实训练及 seed100030 成对仿真。独立进程输出完全一致；离线未通过预登记收益标准，闭环两组均失败，尚未证明触觉提升。v1 效果解释作废，v2 加载阶段中止。结果见 `docs/TACTILE_EFFECT_20261106_RESULTS_ZH.md`。

# 真实LingBot-VLA触觉小适配器训练计划：**用3090，不用CPU代理训练；当前数据门未开**

> 2026-11-06后续更新：已按用户明确授权完成真实GPU训练（3组×120优化步），并开展进程内触觉闭环对照；原“没有optimizer/没有真实触觉权重”的叙述为历史状态。离线收益尚未达到预登记阈值，不能宣称成功率提升。见[本轮协议](TACTILE_EFFECT_PILOT_V1.md)与私有`data/reports/tactile_effect_pilot_v1/`。

> 本文保留历史门禁记录。2026-11-06的用户授权、真实反传与新增验证数据见[最新实测](TRAINING_INTEGRATION_20261106_RESULTS.md)；旧“无forward/backward/无验证来源”的描述已过时，接触覆盖不足仍阻止正式优化。资源收尾遵循当前AGENTS.md，禁止自动恢复Strata。

## 0. 现在到底能不能训练？

不能。新[只读就绪审计脚本](<../src/plan_h50_gpu_training.py>)逐SHA核现有训练来源元数据并核[新seed100022真实负结果](<H50_ONLINE_EXPERT_S100022_RESULT_ZH.md>)的私有动作源：任务虽然成功，只有**46条真正高层`take_action`**，达不到原VLA的50动作H50。旧双瓶split中的5个训练seed＋1个验证seed虽然有RGB/触觉**逐帧候选**，旧规则专家走`move/take_dense_action`而不是同控制单位的50条高层动作，**不得给它们换个名字当H50**。本机新[seed100023真实stride20来源审计](<H50_ONLINE_EXPERT_STRIDE20_S100023_RESULT_ZH.md>)已在**唯一1条物理seed**上找到**8个相互相关的50动作因果候选窗口**并验原HDF5/动作/触觉SHA；这是未分配的train意图，**不是8条独立样本**。独立[seed100024 validation-only真实试采](<H50_VALIDATION_S100024_RESULT_ZH.md>)虽有80次调用/9559步，却任务goal false且无专家HDF5/completion；六份私有失败证据SHA也已由训练就绪审计复核，**不能**把80条无成功回合标签当验证。[seed100025仅一次独立验证](<H50_VALIDATION_S100025_RESULT_ZH.md>)更在setup_demo因瓶初始不稳定直接失败、没有真动作；其私有两文件和两个runner四SHA亦由审计绑定，不重跑、不造H50。当前**获批**H50训练窗口仍**0**、验证窗口**0**（验证候选也为0），`approved_paired_episodes=[]`、双瓶split为`candidate_not_approved`、双瓶接触归一化`training_approval=false`。现有真实154800 BF16权重/零输出适配器已验证一次TRAIN图像`sample_actions`位级相等、3090推理峰≈11.84GiB，但**从未跑真实训练forward/backward/optimizer**，内存峰不能从推理峰臆算。上述事实中的任何一条为假，审计会拒绝并要求重新审阅；报告`gpu_training_permitted_now=false`只是**状态报告，不是以后可用于授权的训练启动器**。

用户特别明确：真正训练**可以且应该用本机唯一RTX3090**，不拿CPU小模型/代理试训冒充真实LingBot-VLA增益。本计划的CPU工作仅限校验路径、SHA、时钟与元数据，不更新任何模型参数；未经数据审批不占GPU“先训一下”。

## 1. 数据先过哪些门

1. **样本语义**：每条候选来自一个全新中继机RoboTwin规则专家回合，规划只看该回合当前场景，逐条真正调用原`Base_Task.take_action(qpos)`。每个训练anchor需要与本回合三摄RGB/14D当前关节状态对应，动作前过去25个物理步触觉压成5×4×3历史＋5×4mask及age；窗口最新力步≤照片物理步。标签是此anchor之后的**连续50次实际调用**及各自正时长`(before,after]`，不是50个物理步或保存帧。未来接触只能当未来监督，不能输入当下；短于50条/任务失败/原专家逐步drive-target/策略自回放一律拒绝。
2. **来源验真**：只允许本机单3090、三原源码SHA/隔离overlay、原相机像素HDF5＋原始raw/obs/frames＋逐调用14D/区间/完成卡全部同回合SHA互绑，且原`check_success`、规划与动作追踪审阅通过。`src/policy_highlevel_chunks.py`仅检查50条*格式*，上游必须独立验HDF5/completion/控制来源；`register_dataset`可登记候选却不自动批准。失败试次保留负证据，不换ID/同seed重试刷成功。
3. **物理seed切分**：重规划产生的在线专家H50与旧逐帧数据分开建新manifest，不复用已有val100012/test100009作在线H50评价；同物理seed的所有重试永不跨train/val/test。**最小技术检查**应先有不同seed的train与validation有效H50各≥1并预留不碰的test seed；严肃同数据对照建议在决定拟合前至少覆盖多独立train seed及不同validation seed、每组有足够有效anchor。样本太少时先报告“不足以训练/估泛化”，不靠重复同seed窗口制造独立样本。新的seed及预算均须负责人逐批批准；[stride20最初是CPU设计假设](<H50_ONLINE_STRIDE20_CPU_DESIGN_ZH.md>)；它在另一个新seed100023已得到8个H50候选锚，但跨seed对比不能分离stride效应和摆放差异，也不能自动用旧seed100022或100023重播。
4. **归一化审核**：仅用这些新*train H50*样本拟触觉统计（当前旧`relay_pickdiverse_contact_norm_candidate_v3.json`来自逐帧来源，不可直接借到新控制语义），保留缺失mask/零力，正力对数及裁剪都需审核；validation/test不能拟合统计。冻结train/val/test来源卡、H50证据、norm的SHA清单后，请负责人明确批准`approved_paired_episodes`、split和train-only norm；现在一个也没有。

[真实候选样本读取器](<../src/h50_online_samples.py>)已在**不训练任何模型、不占GPU**的前提下从seed100023原生HDF5实际解出当前三路240×320×3 RGB、当前14D状态及仅过去的5×4×3触觉/mask，并与之后50×14真实高层动作及50段可变物理区间**分成输入/目标两栏**；测试逐SHA排除HDF5换包与未来动作混入输入。八个anchor都来自同seed，只证明接线能读、不能当已获批训练批次，也不能因CPU读像素测试说“用CPU训过模型”。

## 2. 获得审批后如何在真实GPU训练（**计划，不是已执行**）

- **原模型**：外部154800原权重六分片逐SHA重验，仅在独立生成的V2模型副本上用BF16严格装载旧1,624个key；**冻结整个视觉/语言/原动作基座**，仅挂零输出、约65,664参数的触觉状态token残差adapter。训练时在原`LingbotVlaV2Policy.forward`真正计算50动作的flow-matching loss，触觉只改变state token；不是预计算/替换VLA输出，更不能用旧逐帧next-frame标签当H50。视觉开关关闭时要求同输入位级恒等；所有loss/grad统计来自同批已验train样本。
- **先测显存再拟合**：用户在工作台停止占GPU后端；代理核3090无其他compute、无他人租约/waiter，拿自身独占TTL可续租，单次已批准train样本做**真实VLA完整训练forward＋backward**的显存门，记录CUDA allocated/peak、adapter和基座各参数grad归属、有限loss/grad；这一步仍不得optimizer.step。既有推理峰11.84GiB**不说明反传能装入24GiB**。一旦OOM/NaN/非adapter出现梯度/缺source，退出并归还GPU，不偷偷换成CPU代理或缩成假模型宣称成功。只有负责人另审显存结果后才定步骤预算。
- **正式有界小训练候选**：如果上门通过、数据/normalization已签，再定待审核的初始预算（例：batch=1、梯度累积4、AdamW仅adapter、学习率1e-4、weight_decay0.01、最多120个优化step、warmup10、梯度范数≤1；这些**是提案非已批准超参**）。模型完整真实forward，保留原动作norm、三摄/语言预处理与原采样/flow loss；BF16数值异常即停，验证集只选模型、不更新梯度，测试折最后一次才开。仅私有目录保存有SHA和训练源manifest的adapter safetensors，不存基座拷贝、不推大权重/图像到Git。

## 3. 怎么判断“触觉有没有用”而非凑一个好看数字

在**同一个**冻结训练/验证seed与同一图像、动作标签、采样噪声、训练步数/随机数/优化器预算上比较：A原冻结视觉模型（能说明原基线）；B真实历史触觉adapter；C容量相同但**仅在train折**打乱/错配触觉的adapter（控制“多了参数/优化器就变好”的假象）。报告未分层与接触/无接触、遮挡等预登记子集的动作误差/flow loss和bootstrap按**seed**重采的宽区间；验证不足不报提升。若还要任务闭环成功率，须后续严格可逆server/client tactile_v1能力门和互不泄露的独立seed A/B、同预算调用。即使B略优也不能将冲量/dt代理冒称真实夹爪力，且必须对照C检验。公开旧基线的视觉-only checkpoint绝不覆盖；真实RPC能力只在训练adapter产物SHA/训练源/归一化/上线审批全部通过后开启。

## 4. 停机、资源与人工批准清单

- 数据门：`train/validation`均有**审核通过**的在线高层专家H50与完整同源RGB/触觉/真14D、未碰test；新train-only统计审签；旧短46回合/100013技术试片被排除。
- 模型门：旧权重与补丁SHA匹配，基座全冻且仅adapter可训，完整真实VLA GPU forward/backward内存峰和梯度实证可承受24GiB；推理两步零adapter恒等不冒充反传。
- 对照门：A/B/C同样本、超参、步骤与噪声，未来接触不回流输入，重复同seed不计独立样本，部署RPC默认关。
- 资源门：每批先获GPU租约、核真实计算进程/工作台后端已由用户停止，超时/等待者/显存压力/NaN/异常时安全停；用户不希望每个小试都重启Strata，按有界批次只在收尾恢复一次并释放本人租约。
- 负责人签字：缺**任意一项**只继续做CPU门禁/方案与合法采集的人工设计，**绝不执行训练**。用户已准许GPU作为将来真实训练设备，**不是**批准把当前零H50样本拿去训练。
