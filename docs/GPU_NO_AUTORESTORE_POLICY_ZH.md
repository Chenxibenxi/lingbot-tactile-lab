# GPU批次收尾新规：只释放自己的租约，不再自动拉起Strata

用户已明确纠正：“不是都跟你说了不用自动拉起了咩”。**从此以本条最新规则为准**，覆盖旧实验计划里“每次或批次结束恢复工作台Strata”的句子，但**不篡改旧日志里确曾恢复的历史事实**。代理以前在seed100024试采结束后仍自动恢复了Strata，这是错误；已向用户道歉，并在本仓[AGENTS.md](<../AGENTS.md>)登记资源边界。

以后只要是本触觉仓的GPU实验：开始前须用户许可该实验并实际核唯一RTX3090空卡/无外部compute与持有本人独占GPU lease；用户愿自行在工作台停GPU模型，不要误把租约free当显存free。结束、异常或SIGTERM都**只释放本人lease并报告GPU计算进程、工作台Strata真实loaded/running与systemd unit状态**；**不得**在EXIT trap或收尾流程自动调用工作台`:8077/api/strata/start`、Strata`:8080/v1/load`或`systemctl --user start strata.service`。Strata保持停止/卸载的状态并非故障；人类之后**明确要求启动**时才另外核空卡/租约并启动一次。当前刚被前次旧钩子拉起的服务也**不自动再关闭**来“弥补”，避免额外来回切换。

本机忽略目录`work/`里的九个历史GPU shell包装器已改成EXIT只归还本人租约，其中已使用的旧技术回放/BF16一次性ID包装器直接拒绝再次执行；旧文件不会被推Git，不能靠它们充当未来实验权限。tracked源码`src/`扫描不含自动Strata start/load调用。未来新录制器、GPU训练、视觉对照均须依此规程写**新的**有界包装器并在CPU脚本静态检查通过后试运行；该规则不会自动批准新seed、H50数据、归一化、GPU训练或真实RPC能力。
