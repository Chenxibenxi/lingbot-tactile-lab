# Micro-batch基准v2：关闭过期视觉网格缓存

首轮v1独立结束，micro1完成，micro2/4因三相机缓存拆分与6/12相机输入不一致失败，未发生OOM记录。原始运行、权重、源码和v1证据保留。本修订只在新benchmark进程配置中将precompute_grid_thw统一设false，原比赛源码不改。

沿用datasets/microbatch_throughput_windows_v1.json的32个固定训练窗口、global4、micro1/2/4与累积4/2/1、相同teacher、noise/time、共同micro1预处理更新、warmup后恢复参数及新AdamW、8次计时更新。沿用原数值筛查阈值；起始参数/teacher/noise/time必须跨进程一致。所有三配置均关闭缓存，v2内部比较吞吐；不得直接将跨版本速度差解释为batch效应。

每个配置单独进程、自有GPU租约，外层1000秒，allocated上限22GiB。每种配置一次尝试，OOM或错误如实记录，不自动降配置或重试。输出使用全新data/reports/microbatch_benchmark_v2目录，不保存正式checkpoint，也不触发反馈确认准入。CPU实际源码方法测试只说明缓存机制与形状契约，完整GPU数值等价、吞吐和显存必须等待本轮结果。

三配置全部终态后统一图文报告，展示失败、数值比较和峰值显存。原反馈试运行技术门已失败，384场正式确认不依赖本基准结果转为通过。Strata不自动启动、加载、停止或重启。
