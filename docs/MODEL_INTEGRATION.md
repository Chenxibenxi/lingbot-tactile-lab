# LingBot-VLA 触觉输入实验接入点（设计与 CPU 模块验证）

> 目前只落地仓库中的独立 `src/tactile_adapter.py` 及 CPU 测试，**尚未**改写上游模型训练/服务器/评测客户端，也没有在中继机训练。3090 被其他任务占满，不能抢用。下一阶段每一层都必须 opt-in 且视觉版输出可还原。

## 已核查的上游模型接口

- 原 LingBot-VLA V2 `/mnt/relaydata/lingbot/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:472` 用 `state_proj = nn.Linear(max_state_dim, proj_width)` 投影状态；其基类 `modeling_lingbot_vla.py:1133–1163` 的 `embed_suffix` 构成 state token + noisy action token。
- V2 训练 `forward` 在 `:767–812` 调用 `embed_suffix`；推理 `sample_actions` 在 `:920–1002` 每次去噪迭代通过 `predict_velocity` (`:1004–1021`) 重算 state token。顶层 `LingbotVlaV2Policy.forward` 在 `:1236–1288` 转发批数据，`sample_actions` `:1318–1319` 转发推理。仅改训练数据集**不会**使推理真使用触觉。
- 保持控制输出 14D（现有机器人 6+1 ×2），模型配置的 `max_state_dim=55` 是 padding，不能把触觉塞在未命名的41维里蒙混。第一版的融合点是投影后的**状态 token**：`state_emb = state_proj(state) + tactile_delta`。前缀 Qwen3-VL 和动作维度暂时不改。

## 已写并测试的 adapter

`src/tactile_adapter.py` 接收 `[B,5,4,3]` 历史特征、`[B,5,4]` 有效覆盖、`age_ms[B]`、`window_valid[B]`，经非负载荷 `log1p`、掩码和 5-bin GRU 输出 `proj_width` 残差。输出层严格**零初始化**，初始触觉开启时与视觉基线 state embedding 数值一致；显式关闭、缺传感器、无效窗口都走视觉原值。输出层第一步获得梯度，后续 GRU 可学习。非数值、错误形状、掩码越界立即报错；传感器数值单位是仿真冲量/dt代理而非标定夹力。

## 本轮新增在线与离线共享时序/影子通道（未改原模型）

`src/window_math.py` 统一离线 `align_temporal.py` 和在线 `src/live_tactile_window.py` 的 25 物理步/五分箱计算；已有单测逐特征、mask、age 比较同一输入的离线与在线结果一致，漏步无效、过去锚点/未来数据拒绝。`src/force_probe.py` 仅在 `TACTILE_LIVE_WINDOW=1` 缓存最近125步；`src/shadow_forward.py` 用伪状态token把物理接触→RPC payload→零初始化adapter串通做**CPU影子测试**，能演示数据路径但不会调用真实 LingBot 推理。专家sidecar的原始力header `clock_origin` 与已保存帧clock一致性已加进 `validate_expert_data.py`，历史不同时钟的LXC数据不能进relay候选集。

## 独立客户端补丁预制（未部署）

`src/patch_policy_client.py` 对真正源文件 `RoboTwin/script/eval_policy_client_lingbotvla.py.orig` 用精确锚点生成独立 `work/patches/*.orig`，原源码哈希不变；只在**真实** `model.infer(formatted_observation)` 前尝试 `src/force_probe.py` 的 `.policy.jsonl` 请求时间标记并附加实时因果触觉。`TACTILE_POLICY_INFER=1` 时在初始 WebSocket 连接后要求 `get_server_metadata().capabilities` 明示 `tactile_v1`；旧服务端 metadata `{}` 会提前拒绝，不能静默丢弃键却声称触觉推理。关闭时视觉版原流程不变。该实验源仅经 AST/CPU单测，**还不能运行**：旧 launcher 通过软链回写资产/conda、服务端也尚未发布真实触觉模型能力。 `.policy.jsonl` 是客户端调用尝试的 marker，不代表已收到服务端应答或动作成功。

## 服务端预处理桥（CPU契约，未部署）

`src/server_tactile_bridge.py` 测试了旧 `feature_transform.apply` 会丢未知触觉键时的安全做法：先解码、只把视觉字段交旧transform，然后把通过schema验证的触觉历史/掩码/age/valid加回；同批次旧视觉客户端显式获得invalid/zero-mask字段，不允许首个batch样本字段不同。只有明确 `model_wired=True` 且提供adapter artifact SHA256的实验服务才能申请发布 `capabilities=['tactile_v1']`。这只是CPU拟合transform与batch的契约，**目前没有真实训练权重/模型forward接线/服务进程，禁止将占位SHA视作训练证明。**

## 尚须逐层接线（后续变更范围）

1. **数据**：`src/paired_samples.py` 已能从中继机本地、SHA 校验通过的同源 HDF5+侧录读取输入（当下新样本数=0）。`src/split_and_norm.py` 已实现按 `(task,实际seed)` 全轨迹分组并让同seed重试不跨分割（默认每任务≥5独立seed），以及**只读train trial**拟合 log1p 特征的 median/IQR；目前无新数据所以没有可发布 split 或 norm 参数。首批帧历史不足与零物理时长的重复帧不能暗中充当有效触觉。一个以相同新数据训练的视觉-only 对照必须同步建立。
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
