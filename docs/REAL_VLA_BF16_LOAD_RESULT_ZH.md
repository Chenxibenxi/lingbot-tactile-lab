# 真 LingBot-VLA V2 权重：本机3090严格逐片装载＋内层零触觉适配器成功，**没有前向或训练**

2026-11-02 用户明确批准在Strata空闲时占用GPU。按[试前锁定的资源/恢复协议](<REAL_VLA_BF16_STREAM_PROBE_PLAN_ZH.md>)在唯一中继机RTX3090上，先确认Strata `busy=false/queued=0/requests=0/in_flight=0`、独占交互lease，原子`/v1/unload`成功后临停空闲Strata服务，在已冻结SHA的隔离补丁真实`LingbotVlaV2Policy`实例上：meta设备预检原**1,624键/形状**，BF16 `to_empty(cuda:0)`，按六原分片逐片再算**整个文件**SHA256与[原分片指纹](<../datasets/relay_154800_shard_sha_v1.json>)一致，再从原FP32 safetensors逐键同步复制到真实GPU模型。全键复制成功后**冻结整个视觉基座**，只在内层`FlowMatchingV2`挂一个输出初始化为零的触觉残差模块（65,664参数），保持训练和服务能力门默认关闭。上游LingBot源SHA仍`9b3c9a2a…`、六源分片原地只读，不复制权重到研究仓。此次脚本输出[脱敏装载证据卡](<../datasets/relay_v2_real_bf16_strict_load_v1.json>)：`keys_loaded=1624`，原视觉BF16参数分配前空壳12,539,666,432字节，挂零adapter后PyTorch CUDA allocated/peak均**12,539,800,064字节≈11.68GiB**，低于预登记18GiB上限；这些是**仅参数装载阶段**，不代表完整前向/反向峰值。

失败与恢复均须披露：第一次尝试六片值copy后因顶层Policy与内部FlowMatching挂载层级混淆而exit1；修复内层挂载并加CPU回归后，第二次已挂零adapter，但末端错误地把`model.tactile_adapter.*`算成“未冻结旧基座”而exit1；都经EXIT钩子恢复Strata、释放lease。第三次修正该前缀判断并通过**178/178 CPU测试**才取得上面的完整`exit0`证据；完成后原`strata.service`重新由systemd托管为`active+enabled`、`:8080/health loaded=true`，lease已释放、仅Strata占GPU，原RoboTwin和LingBot源码均未改。每次失败的GPU模型均随进程退出丢弃，不保留半加载权重。

**科学边界：**虽然这次是真154800检查点在真实VLA类上的严格GPU值装载和零adapter接线，不是代理MLP，但**没有调用模型完整`forward`、`sample_actions`、真实RPC、反向、optimizer或训练**；“零残差数学/CPU桩恒等”不等于真VLA图像/动作输入的数值对照。原H50物理时钟训练/验证窗口不足，[数据批准门](<TRAINING_REVIEW_GATE_ZH.md>)仍`approved_paired_episodes=[]`、六份候选归一化`training_approval=false`，不得拿这次装载宣称触觉增益或发布`tactile_v1`。下一步必须先在安全租约下测**同一批真实视觉输入**的禁用/零adapter数值恒等和单批前向/去噪峰值；涉及训练须先确定同回合专家动作chunk实际时钟、完备分折/归一化并由负责人明确审批。其他用户任务或Strata正忙时不抢GPU。

**后续独立阶段已完成本报告所列“下一步”中的推理恒等检查：**见[同一真实训练帧视觉／零触觉`sample_actions`数值结果](<REAL_VLA_VISUAL_ZERO_NUMERIC_RESULT_ZH.md>)；本页描述的首次参数装载job本身仍未跑任何前向，两个阶段的证据和GPU峰值不可混称。
