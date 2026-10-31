# E010 · 视觉/夹爪触觉代理的时间轴与数据契约（草案 v1）

> 目标：先使 RoboTwin 的相机、实际机器人状态、每指接触和动作拥有同一个**可审计的物理时钟**；随后在本独立仓库迭代 LingBot-VLA 的训练与评测，让触觉真正进入模型。原 185800 视觉版留作不可变对照；不是永久禁止改进训练或推理。**后续正式新数据仅由本机中继机单张 RTX3090 产生；历史 LXC 仅工程验证。**

## 0. 核查结论：旧示范没有物理时间戳

真实例子：`/mnt/relaydata/lingbot/raw/lift_pot/aloha-agilex_clean_50/data/episode0.hdf5` 有 114 张相机图和 114 条 `joint_action`；**无** scene physics step / 相机采集时间 / seed 的 HDF5 属性或数据集。`seed.txt` 第一集 seed=0，但第五集 seed=6；`episode_idx` 不能充当 seed。已生成的 LeRobot `lift_pot-train45` 对应第 0 集有 113 行，`timestamp=0,0.02,...,2.24` 来自 converter 强制 `fps=50`；预览视频又按 30fps 写出。**两个 FPS 都只是封装播放/训练采样时标，不能反推物理相机帧间隔。**

原 Demo 的 `_take_picture()` 在不同运动段会在段前、末尾和按 `save_freq=15` 采帧（含 `i=0`），不均匀且可近重复。历史 HDF5 + 新跑的触觉 JSONL 即使同任务同 seed，也**不是**同一轨迹；专家规划和物理执行可不确定。不得按 `frame_index * 0.02` 或相邻 seed 的重放强行拼接旧训练集。新触觉训练数据必须同一模拟 episode **重采**，或找到附带可信物理步索引的原始证据后再验。

## 1. 明确时钟与速率（绝不混用）

| 层 | 目前可验证的时钟 | 当前产物 | 禁止假设 |
|---|---|---|---|
| PhysX 接触 | `scene.step()` 后 `physics_step` 整数 + `scene.get_timestep()`（本场景约 0.004s） | 每一步左右指接触冲量和 `load_proxy_n` | SAPIEN 存了真实 GelSight 压力图，或 1 步 = 1 次模型决策 |
| `get_obs` 三相机 | 同 episode 中获取 RGB 后**旁路记** `obs_index → physics_step`；若无观测源则无绑定 | `.obs.jsonl` 不存像素，`rgb_shapes` 只核查完整性 | `get_obs` 每次都供策略；相机一定按固定 30/50Hz 触发 |
| 运动指令 | `take_action_cnt` 是本回合**单次 14 维动作**序号；一次动作在内部拓扑规划可包含多次 physics step | 原 E009 raw `action_index` | 14 维动作一次恰好对应一相机帧或一个固定物理时长 |
| LeRobot 视频/训练 | `fps=50`、chunk offsets `0..49` 是旧版**样本索引** | 原转换 `state[t]`, `action[t]=state[t+1]`、视频 3 路 | 50 个帧索引等于真实 1 秒、可与 250Hz 的 4ms 直接相乘 |
| VTAM 论文（外部） | [方法论文 §4.1](https://arxiv.org/html/2603.23481v1#S4.SS1) 采集/动作执行 30Hz，[§4.3](https://arxiv.org/html/2603.23481v1#S4.SS3) 推断 1Hz | GelSight 图像，非本项目的冲量 | 30Hz 是其模型推断；论文公布了完整硬件同步/插值方案（没有） |

历史 E009 策略 smoke 的 `physics_step=0` 表示**第一步被监控的 policy 动作以前**；现有专家版本则明确把 clock 原点设在 `check_stable` 结束、开始专家运动之前，初始化/布景物理步不算示范时间。专家版在两种运动路径每步采样，并在 `_take_picture` 成功落 pkl 后记录 `FRAME_IDX→physics_step`；`.obs.jsonl` 的 `capture_monotonic_ns` 只供同进程诊断延迟，**训练对齐以 physics_step 为准**。不同 clock origin 的运行不可直接拼接。

## 2. 已实现的四份同源数据（单 episode）

```
<episode_uid>.jsonl           # 原始每步夹爪接触；actual_seed/episode/dt/clock_origin
<episode_uid>.obs.jsonl       # 每次 get_obs 相机捕获的物理步；不是推断事件
<episode_uid>.frames.jsonl    # 成功保存图像的 FRAME_IDX→obs_index→physics_step + 实际qpos/qvel
<data>/episodeN.hdf5          # 同一次运行原有 RGB/14D command-state
completion.json               # 通过帧数/数值/SHA256 校验后才写的完成证明
paired_index.jsonl            # 派生的因果窗口 + HDF5 图像引用 + 下一帧动作标签
```

原始每步数据按四根指头 `fl_link7,fl_link8,fr_link7,fr_link8` 记录 `points`、对手物体、冲量 N·s 与 `load_proxy_n=Σ||J_point||/dt`。只读 SAPIEN 接触点，过滤机器人内部 link 碰撞和 0 冲量 near-contact。含对象名只用于诊断，**不进入初版策略特征**（物体名可能是特权标签）。载荷代理即使单位是 N，也**不是**经过指面法向校准的夹持法向力；不推导真实滑移、压力或力限。

`get_obs` 旁路 marker 含 `obs_index, physics_step, sim_time_s, action_index, valid, rgb_shapes, capture_monotonic_ns`。主键必须是文件中唯一 episode UID + obs_index，不能用跨任务复用的 seed 或模型 14D action 代替。单次 smoke 的相机 marker 步 `[0,50]`，其中步 0 没有 E009 采样，必须标 `valid=false`。

## 3. 实施的第一个因果窗口（不等于训练集）

以每个 marker `T` 为锚点，仅允许读取**同 episode** `s ∈ [T-24,T]` 的 25 个物理步样本。以 dt≈4ms 计，这覆盖最近 **25 次采样/100ms 物理时间格**（最早到最晚采样中心相距 96ms）；分成五个连续 5-step 子窗，每窗约 20ms。每侧两指共四指；每指每子窗三个原单位特征：

```
mean_load_proxy_n, peak_load_proxy_n, contact_fraction
shape = [5 time bins, 4 finger links, 3 features] = 60 scalars
mask  = [5, 4] 每子窗每指观测覆盖比例
```

- 特征仅在**已收到**的样本上计算；丢步输出 0 仅作存储占位，mask<1，**不做零力填补也不跨缺口插值**。
- `quality.window_valid` 要求所有 25 个连续物理步都有**四指**记录且 marker `valid=true`；首帧和完整窗口不足均 invalid。标注 `coverage, bins_coverage, last_source_step, force_age_ms, per_finger_complete`。`last_source_step ≤ T` 是无未来信息的审计证据。
- `high_impact_warning` 默认每指子窗 >500N 时标记**仅供人工审查**，不是未经校准的训练过滤或实机安全阈值。绝不自动 clip 原始值；模型输入的稳健归一化/阈值只能从训练划分拟合。
- 该输出只是 `get_obs` 的**候选观测**，不含图像像素、图片引用、真实机器人 `qpos/qvel`、下一动作标签，尚**不能直接训练**。正式决策锚点需在实验 client `model.infer` **发出前**打 `policy_request` marker；`get_obs` 中间为了录像/判断成功产生的采帧不能盲当模型推断帧。
- 针对时间一致性，帧间视觉按 `physics_step` 选择不晚于锚点的最近一帧并记 age；数值本体位置可在两条**过去**状态间按真实时间插值，**不允许**使用当前决策后的未来状态或图像。对于力/碰撞冲量不插值。

## 4. 真正可训练数据的下阶段采集契约（暂未修改演示采集器）

```
run_manifest.json:
  schema, simulator_commit, task_config_hash, timestep_s, clock_origin, save_freq,
  robot_urdf_hash, collector_version, seed_manifest_hash, modalities, source_split
frame_index.jsonl  一次 _take_picture 对应一条:
  task, source_episode_id, episode_uid, actual_seed, original_frame_idx,
  capture_physics_step, capture_sim_time_s, image_ref/{head,left,right},
  commanded_state14, actual_qpos/qvel(可用则另栏), valid_flags
physics.jsonl 每一 scene.step 一个:
  episode_uid, physics_step, dt_s, action_index/expert_motion_phase,
  left/right finger contacts, impulse Ns, optional actual joints
policy_request.jsonl 每次真正 model.infer 前一个:
  episode_uid, observation_frame_idx, request_physics_step,
  history_anchor_step, model_id, action_chunk_id
```

**专家重采工具已写好**（`src/make_expert_overlay.py` + `src/record_expert_demo.py`）：`take_dense_action` 和 `together_move_to_pose` 的所有**示范阶段**物理步采样，`_take_picture` 成功保存后记 FRAME_IDX，跳过之前的初始化/稳定阶段并明确 clock origin。此前在 LXC 上完成过一个同源试片（113训练行），只作历史工具验收；本仓库的新数据**必须在中继机3090重新采集**。所有 paired 数据在**同一实际运行 episode**取得实际 seed、episode 与图像索引；不能只复用策略的 `take_action` 插桩。重新生成 LeRobot 时输入新增独立 `observation.tactile_history` 及 mask / dt / age，动作仍为 14D；模型配置中的 `action_dim/max_action_dim=55` 是统一 padding，**不是**已有 41 维触觉位。

新数据建议先做 2 个接触任务 × 每任务几条 paired 演示 + 静态/运动遮挡负样本，保留整段 raw 与视频，实测 contact onset 持续长度、峰值、采集延迟 p50/p95。根据短触碰时长扫描历史窗口 40/100/200/500ms；不要先把 VTAM 的 30Hz/1Hz 硬套到现有 250Hz + 不规则相机帧。

## 5. 切分、监督及门禁

1. **不碰旧数据**：现有 2,250 clean 演示只作为 vision-only 预训练；不能把新力时序按 `t/50` 回填旧 HDF5。新 paired 集必须携带相机物理步+图像引用与时间来源；转换前核对 `raw 114帧 → LeRobot 113行` 的最后帧只是下一动作标签。
2. **防泄漏**：策略输入仅 `(≤当前 policy_request step)` 的相机、力、状态；未来接触/力可用作**训练辅助监督的 label**，永远不得放到当前输入。现有 185800 主训练配置 `use_future_image=false`；其他 teacher 配置可能为 true，触觉对比实验必须核验且避免未来图像泄漏。
3. **按轨迹切分**：按 `(task,original_episode_id, actual_seed, object/texture)` 固定 train/val/test，保存 split manifest；重编号后的 LeRobot `episode_index` 不可反查原 seed，须显式映射。
4. **验证指标**：同 seed 视觉-only vs 记录-only 动作/成功不变；物理步计数和 frame_idx 单调；无时间逆序/缺失悄悄补 0；隔离输入输出和命名信息；性能 p50/p95 与数据体积；训练时真实触觉胜过时序打乱触觉才算有效。

## 本仓库复验

```bash
cd /mnt/relaydata/lingbot-tactile-lab
PYTHONPATH="$PWD/src" /mnt/relaydata/lingbot/conda/envs/lingbot-eval310/bin/python -B \
  -m unittest discover -s tests -p 'test_*.py' -v
# 确认 GPU 空闲后，才按 README 的单机采集门禁产生 data/paired/<trial>。
```

早期 LXC 外部单动作工程样本曾得到步0无历史、步50完整 `[5,4,3]` 窗口；**不在本仓库数据集中**。真实中继机3090的新数据须自行通过 `completion.json`、HDF5 与帧索引严格验证后才可成为训练候选。
