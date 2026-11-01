# LingBot-VLA 触觉输入实验接入点（设计与 CPU 模块验证）

> 目前落地仓库内可选 adapter、因果数据接口及CPU门禁，**尚未**把训练好触觉权重接进上游真实模型/服务器，也没有在中继机训练触觉VLA。单3090必须逐任务租约且遵守用户规定的Strata停机/恢复时限，不抢其他任务；每一层都须 opt-in 且视觉版输出可还原。

## 已核查的上游模型接口

- 原 LingBot-VLA V2 `/mnt/relaydata/lingbot/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:472` 用 `state_proj = nn.Linear(max_state_dim, proj_width)` 投影状态；其基类 `modeling_lingbot_vla.py:1133–1163` 的 `embed_suffix` 构成 state token + noisy action token。
- V2 训练 `forward` 在 `:767–812` 调用 `embed_suffix`；推理 `sample_actions` 在 `:920–1002` 每次去噪迭代通过 `predict_velocity` (`:1004–1021`) 重算 state token。顶层 `LingbotVlaV2Policy.forward` 在 `:1236–1288` 转发批数据，`sample_actions` `:1318–1319` 转发推理。仅改训练数据集**不会**使推理真使用触觉。
- 保持控制输出 14D（现有机器人 6+1 ×2），模型配置的 `max_state_dim=55` 是 padding，不能把触觉塞在未命名的41维里蒙混。第一版的融合点是投影后的**状态 token**：`state_emb = state_proj(state) + tactile_delta`。前缀 Qwen3-VL 和动作维度暂时不改。

## 已写并测试的 adapter

`src/tactile_adapter.py` 接收 `[B,5,4,3]` 历史特征、`[B,5,4]` 有效覆盖、`age_ms[B]`、`window_valid[B]`，经非负载荷 `log1p`、掩码和 5-bin GRU 输出 `proj_width` 残差。输出层严格**零初始化**，初始触觉开启时与视觉基线 state embedding 数值一致；显式关闭、缺传感器、无效窗口都走视觉原值。输出层第一步获得梯度，后续 GRU 可学习。`set_normalization()`现要求 `training_approval is True`，**缺字段也拒绝**，而不再只拒显式 false；目前本机两任务的候选配置均不得被adapter当已批准训练参数安装。非数值、错误形状、掩码越界立即报错；传感器数值单位是仿真冲量/dt代理而非标定夹力。

## 本轮新增在线与离线共享时序/影子通道（未改原模型）

`src/window_math.py` 统一离线 `align_temporal.py` 和在线 `src/live_tactile_window.py` 的 25 物理步/五分箱计算；已有单测逐特征、mask、age 比较同一输入的离线与在线结果一致，漏步无效、过去锚点/未来数据拒绝。`src/force_probe.py` 仅在 `TACTILE_LIVE_WINDOW=1` 缓存最近125步；`src/shadow_forward.py` 用伪状态token把物理接触→RPC payload→零初始化adapter串通做**CPU影子测试**，能演示数据路径但不会调用真实 LingBot 推理。专家sidecar的原始力header `clock_origin` 与已保存帧clock一致性已加进 `validate_expert_data.py`，历史不同时钟的LXC数据不能进relay候选集。

## 独立客户端补丁预制（未部署）

`src/patch_policy_client.py` 对真正源文件 `RoboTwin/script/eval_policy_client_lingbotvla.py.orig` 用精确锚点生成独立 `work/patches/*.orig`，原源码哈希不变；只在**真实** `model.infer(formatted_observation)` 前尝试 `src/force_probe.py` 的 `.policy.jsonl` 请求时间标记并附加实时因果触觉。`TACTILE_POLICY_INFER=1` 时在初始 WebSocket 连接后要求 `get_server_metadata().capabilities` 明示 `tactile_v1`；旧服务端 metadata `{}` 会提前拒绝，不能静默丢弃键却声称触觉推理。关闭时视觉版原流程不变。该实验源仅经 AST/CPU单测，**还不能运行**：旧 launcher 通过软链回写资产/conda、服务端也尚未发布真实触觉模型能力。 `.policy.jsonl` 是客户端调用尝试的 marker，不代表已收到服务端应答或动作成功。

## 服务端预处理桥（CPU契约，未部署）

`src/server_tactile_bridge.py` 测试了旧 `feature_transform.apply` 会丢未知触觉键时的安全做法：先解码、只把视觉字段交旧transform，然后把通过schema验证的触觉历史/掩码/age/valid加回；同批次旧视觉客户端显式获得invalid/zero-mask字段，不允许首个batch样本字段不同。修复了占位 SHA 的假阳性：**仅有 `model_wired=True` + 任意64位SHA不再能声明 `capabilities=['tactile_v1']`**。现在需另外出示仓库私有`models/`里真实SHA匹配且非零输出的 `.safetensors`、独立训练记录（来源单3090、真实优化步>0、已批准部署）及哈希匹配的已批准 provenance/train-only split/norm；`training_approval`缺失、false或真实`approved_paired_episodes=[]`一律拒绝。`src/adapter_artifact_gate.py`使用无pickle的`safetensors`按键/尺寸检查非零输出层、GRU结构及归一化缓冲**数值与获批训练折配置一致**，合成临时已批准fixture仅用来测代码正路径，不代表用户当前数据已获批。新门禁只是声明材料的完整性检查，**不能代替负责人批准或证明部署、触觉模型提升**；当前没有真实训练权重／模型forward接线／服务进程，服务器仍禁止宣称触觉能力。

## 可逆 FlowMatchingV2 原型（未安装/未训练）

`src/patch_model_v2.py` 只对本机已知 V2 源源码作严格锚点的**独立副本**补丁，启用 `config.tactile_enabled` 时初始化零输出 adapter，从顶层训练 `forward(**kwargs)` 向 `FlowMatchingV2.forward` 显式传递 history/mask/age/valid，并在 `sample_actions` 中单次编码残差，`predict_velocity` 每步只注入**state token**，不污染 action token；关闭时不加 adapter 新参数，旧行为保留。`tests/test_patch_model_v2.py` 抽取源 AST，在纯CPU假模型上检查零初始化视觉恒等、受训非零输出仅改state token、无adapter拒绝触觉，以及原源码哈希不变。在用户允许的Strata停机窗口内，经独立interactive GPU租约，已把此补丁生成到本仓库Git忽略`work/model_patch_preview.py`，使用本机RTX3090和LingBot实际依赖**成功导入真实补丁模块及FlowMatchingV2类**；源modeling文件SHA`9b3c9a2a…`始终不变。重要依赖边界：CPU仅AST测试可跑，但真实模块import过程中上游Triton group_gemm预调配置会主动调用`torch.cuda.get_device_capability()`；隐藏GPU直接import会`RuntimeError: No CUDA GPUs are available`，不能拿CPU import失败误判补丁有错，也不能在未拿租约时偷启GPU。此次**未实例化真实V2、未加载旧checkpoint、未训练或服务端透传**；不得据此发布tactile_v1能力。 基线权重应先以 `tactile_enabled=False` 按原流程完整载入，再由隔离训练代码显式附加新adapter并载入独立检查点；直接在加载旧权重前设为True可能因新参数缺权重/空初始化而失败，不能借 `strict=False` 掩盖。

## 同源策略自回放（隔离客户端实测，但未批准训练）

`src/policy_rollout_capture.py` 提供独立 HDF5 writer：在真正 `model.infer` 处保存该请求实际使用的三相机 uint8 RGB、14D 当前指令状态和仅过去物理步的触觉窗口；**每次实际** `TASK_ENV.take_action` 之后记录施加的14D命令与前后物理步，避免把预测chunk未执行的后半截当标签。相机 request 可恰好等于上一动作结束物理步，时钟不允许倒流；最初没有接触时mask必须保持invalid。`complete` 仅指本地文件有请求与实际动作，`target_semantics=executed_self_rollout_not_expert_or_training_approved`：输出来自本视觉模型的动作，**不是专家纠错目标**，即使回合成功也不能直接声明训练触觉改进。已在**隔离复制**的RoboTwin客户端上通过 `src/patch_policy_capture_client.py` 严格锚点接线；154800视觉-only/BF16、relay单3090/历史正例seed100048一集成功，自回放记录2次真实infer、100个实际施加动作、三路原始RGB HDF5和因果触觉，`src/validate_policy_rollout.py` 同源四流加HDF5校验通过，并在 `datasets/relay_policy_self_rollout_154800_s100048_v1.json` 登记字节长度与SHA256。该卡状态仍**self_rollout_not_expert_or_training_approved**，只有被选择的一条种子，不代表触觉模型成功率或获得专家纠错标签；下一步需更多独立种子/失败反事实、对接训练目标与真正模型条件化，再决定辅助接触任务或受控行为克隆对照。

## 尚须逐层接线（后续变更范围）

1. **数据**：`src/paired_samples.py` 已从唯一relay3090五条经SHA绑定成功门的同源专家 HDF5+侧录筛得 **518** 个可用因果帧对（5个独立seed，全部候选、未授权训练）。`src/split_and_norm.py` + `src/plan_relay_training.py` 按 `(task,实际seed)` 全轨迹分组让同seed重试不跨分割，建立**仅试点**3train/1val/1test seed split并只用train309帧对拟 log1p 特征 median/IQR；所有四指全部3特征的中位数0、IQR跌到0.1地板（接触正值仅约22–23%完整bin），说明零主导归一化可能夸大稀疏接触响应，正式训练前需评审接触感知归一化和幅度限制，不能把技术最低5 seed误认为足够科学统计功效。新增 `src/contact_normalization.py` v2 候选只用训练折正接触bin拟合、无力原值保持精确0、归一化力幅限制±4；真实训练折仍有8/8力通道IQR触底0.1、约4.76%正值会被夹，不把这当性能胜利；config仍 `training_approval=false`，adapter拒绝装载待评审候选。`src/paired_ablation.py` 在CPU真实解码五seed全部518对三路RGB并构造逐对同RGB/state/target的视觉-only、真实触觉、因果历史内部重排视图；只有123对含任一非零历史力，另外395对重排不改变力值，故总样本动作损失绝不能独当触觉增益，后续应预注册接触窗口分层指标。首批帧历史不足与零物理时长重复帧未被冒充有效触觉。以后以相同新数据训练的视觉-only对照须同步建立。
2. **训练 API**：在此仓库中生成 LingBot 模型的受控小补丁或独立实验 subclass，使训练 batch 显式包含 `tactile_history`, `coverage_mask`, `age_ms`, `window_valid`；从顶层 `forward` 到 `embed_suffix` 逐层透传。不改变旧 API 参数默认值。保存 adapter 的独立 checkpoint + 视觉基座 commit/hash/训练 split/归一化参数。初轮只训 adapter，记录 action loss + 接触辅助任务验证指标。
3. **推理 API**：`src/tactile_rpc.py` 已给出客户端/服务端的 CPU 可测字段契约（仅 `observation.tactile.history/mask/age_ms/valid`，绝不发送未来力或特权对象名；旧消息完全保持原样，缺失/陈旧传感器显式降级）。现有 RoboTwin 客户端 `script/deploy/websocket_client_policy.py:50–57` 的实际 msgpack_numpy packer 经过回环单测。**下一步才是**扩展客户端 `formatted_observation` 与服务端实际调用：`deploy/lingbot_vla_v2_policy.py:405–415` 的 `_prepare_model_input` 调用 `feature_transform.apply`，未知触觉键可能被丢弃，须在 transform 前解析/在后面显式加回；`:439–459` 批处理只遍历首个样本键，batch内必须给每条观测统一补可见的 invalid/zero-mask 缺省字段；`:75–86, 95–143` 才把字段送模型的 `sample_actions`。`sample_actions` 显式接收触觉并进入每次 denoise 的 state token。可预先算一次 `tactile_delta` 传 `predict_velocity`，避免重复GRU；确保多请求/前缀缓存不串数据。
4. **评测**：分别测 (A) 相同新数据训练的视觉-only、(B) 正确触觉、(C) 时序打乱触觉、(D) 正确触觉+独立受控反馈。共享原始 seed/任务/权重和渲染设置；报告成功率、接触 onset 检测、尖峰/滑落候选、延迟 p95。模型触觉 missing/fault 应确定性降级到视觉，不回写改变原比赛成绩。
5. **工程边界**：旧评测 launcher 会通过 overlay 软链修改 assets 与 conda `.pth`，必须先隔离可写资源，不能直接在原环境强行起完整评测。低层夹爪限力回路另行实现，不由 VLA 慢速 infer 替代。

## 可复验的 CPU 契约

```bash
cd /mnt/relaydata/lingbot-tactile-lab
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" \
  /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -B \
  -m unittest -v tests/test_paired_samples.py tests/test_tactile_adapter.py
```

此次模块尚不能承诺接触丰富任务成功率提升，更不能将 VTAM 的 GelSight 图像虚拟力当成我们已有的真实触觉传感器。只有在同源配对训练数据、模型端显式条件化与同种子消融均通过后才能称“触觉接入”。
