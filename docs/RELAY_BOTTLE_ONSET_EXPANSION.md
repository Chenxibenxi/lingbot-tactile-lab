# 双瓶接触起点数据补采：预登记两次独立seed试片

**动机：**旧双瓶候选训练折虽然有341个有效帧对、左右分别180／178个未来区间接触，却只有左右各3个“此前无接触→首次接触”区间，约每独立seed每手1个。增加相邻帧只会重复持续接触；先补真实独立初始场景，再看新接触事件能否通过同源质量门。两条受控开爪失败不是专家正确动作标签，继续隔离。

**预先固定，不按结果挑种子：**本机中继机 RTX3090 上的 `pick_diverse_bottles` 规则专家，两个**未曾登记为成功候选**的物理种子按顺序 `100013`、`100015`，各只运行一次，独立输出 `data/paired/relay_pickdiverse_s100013_onsetpilot_expert0/` 与 `data/paired/relay_pickdiverse_s100015_onsetpilot_expert0/`；失败照实记录status/failure，不能替换成别的seed凑成功率。历史seed100001因瓶不稳定未生成物理轨迹，不冒充抓取失手。不用视觉VLA选择成功数据，也不改变原RoboTwin任务、专家指令、评测成功判据。

**执行边界：**逐试取单GPU interactive租约TTL900s，确认单张3090、无其他compute进程、Strata依用户要求停机且不跨北京时间2026-11-02 12:00恢复窗口，`timeout --signal=TERM 660s`有界运行；一次结束即释放租约，下次重新检查。任何忙卡让位，**不抢占/不卸载别人的服务**。保证原`envs/_base_task.py` SHA为`448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101`。两个输出目录与锁不得预先存在；`--dry-run`不使用GPU/创建输出。

**训练门槛：**只有原专家planner+任务目标成功、raw/obs/frames与同源RGB HDF5所有SHA/物理步通过，才可写**单独新的候选专家卡**；旧 `relay_pickdiverse_seed_split_candidate_v1.json` train/validation/test和norm候选**保持字节不变**，新seed不自动进入训练集。成功后另做独立seed切分v2方案与接触起点统计；failed setup/planner/转码半成品只能诊断，不能混合正例。所有原始图、侧录、HDF5、视频继续Git忽略，本地小卡仍`candidate`、`training_approval=false`；目前不能报告触觉模型训练或在线RL效果。
