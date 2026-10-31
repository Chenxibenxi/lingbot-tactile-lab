# LingBot-VLA 触觉输入实验接入点（设计与 CPU 模块验证）

> 目前只落地仓库中的独立 `src/tactile_adapter.py` 及 CPU 测试，**尚未**改写上游模型训练/服务器/评测客户端，也没有在中继机训练。3090 被其他任务占满，不能抢用。下一阶段每一层都必须 opt-in 且视觉版输出可还原。

## 已核查的上游模型接口

- 原 LingBot-VLA V2 `/mnt/relaydata/lingbot/lingbot-vla-v2/lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py:472` 用 `state_proj = nn.Linear(max_state_dim, proj_width)` 投影状态；其基类 `modeling_lingbot_vla.py:1133–1163` 的 `embed_suffix` 构成 state token + noisy action token。
- V2 训练 `forward` 在 `:767–812` 调用 `embed_suffix`；推理 `sample_actions` 在 `:920–1002` 每次去噪迭代通过 `predict_velocity` (`:1004–1021`) 重算 state token。顶层 `LingbotVlaV2Policy.forward` 在 `:1236–1288` 转发批数据，`sample_actions` `:1318–1319` 转发推理。仅改训练数据集**不会**使推理真使用触觉。
- 保持控制输出 14D（现有机器人 6+1 ×2），模型配置的 `max_state_dim=55` 是 padding，不能把触觉塞在未命名的41维里蒙混。第一版的融合点是投影后的**状态 token**：`state_emb = state_proj(state) + tactile_delta`。前缀 Qwen3-VL 和动作维度暂时不改。

## 已写并测试的 adapter

`src/tactile_adapter.py` 接收 `[B,5,4,3]` 历史特征、`[B,5,4]` 有效覆盖、`age_ms[B]`、`window_valid[B]`，经非负载荷 `log1p`、掩码和 5-bin GRU 输出 `proj_width` 残差。输出层严格**零初始化**，初始触觉开启时与视觉基线 state embedding 数值一致；显式关闭、缺传感器、无效窗口都走视觉原值。输出层第一步获得梯度，后续 GRU 可学习。非数值、错误形状、掩码越界立即报错；传感器数值单位是仿真冲量/dt代理而非标定夹力。

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
