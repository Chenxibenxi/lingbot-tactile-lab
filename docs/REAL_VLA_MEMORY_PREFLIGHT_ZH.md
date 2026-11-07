# 真 LingBot-VLA 旧视觉权重的主存门禁：当前拒绝直接按原加载器实验

本机原V2部署`load_model_weights`先把目录全部`.safetensors`逐键放进`merged_weights`，最后`load_state_dict(strict=True)`；真实模型创建和视觉前向另需内存。2026-11-02凌晨的快照（单张RTX3090空闲、用户规定的Strata停机窗口）对**外部原154800 FP32 checkpoint**作[只读CPU预检](<../src/vla_full_load_preflight.py>)：原部署源码SHA仍`2caf39d560...`，6个原权重分片合计**24,905,792,676字节≈23.19GiB**，此刻`MemAvailable≈12,592,410,624字节≈11.73GiB`；仅按分片总大小+4GiB保守安全余量，就需约**27.19GiB可用主存**。因此退出码4、`safe_to_attempt_full_loader=false`，**没有启动原模型实例或GPU前向**。该界限不是实际峰值的精确测量；mmap、缓存、交换空间、模型构造、BF16化和反传会影响真实峰值。先前隔离视觉-only BF16推理曾成功不表示**此刻**可安全把原FP32六分片合并并做adapter反传。

同一研究仓库的CPU[零残差装载契约](<../src/model_attach_contract.py>)和[训练／去噪方法体小桩测试](<../tests/test_patch_model_v2_full_path.py>)只证实代码顺序与调用路径，不认证26GB原权重是否严格加载。**不可用`strict=False`、把视觉权重复制到`models/`、启动更多占GPU服务或盲拉高系统swap来绕过门禁。**后续方案应先设计、测试隔离的逐分片／meta-device或只加载必要子模块的严格、可核验权重加载器，明确全量键集合与SHA、可控峰值，再在取得训练数据批准并持单GPU lease条件下做单batch/no optimizer的BF16数值恒等→非零梯度/显存实测；其间任何显存／RAM不足即让位。原模型与原RoboTwin只读，Strata按用户预定时间恢复。

此外动作数据门仍未过：旧H50专家保存帧没有可用的提锅训练／验证连续窗口；新逐物理步命令完整成功样本只在同seed技术复演、尚未规定真实policy动作控制节奏，不准据此启动真VLA拟合或声称性能增益。
