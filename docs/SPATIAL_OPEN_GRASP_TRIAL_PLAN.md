# 预登记：仅一次 seed100007 开爪 + 三维触点侧录

**目标：**为已交付的指连杆颜色回放补一个**新的、绝不回填旧试**的可审核仿真试次，验证三维接触点是否可随同回合RGB投进手腕相机。这是额外一条受控诊断，不是触觉VLA训练，也不替换旧 seed100000 试次。

- 唯一任务/场景 `pick_diverse_bottles`，规则专家**已成功**的本机relay物理 seed `100007`，episode0；新目录固定 `data/negative/relay_pickdiverse_s100007_open_grasp_spatial1/`，绝不重用或覆盖。与 seed100007 正例对比初始三路RGB像素和首14D指令；若不匹配，只能标明初始不匹配，不能声称严格同起点。
- 唯一动作干预仍是对规则专家两次 `grasp_actor` 的 gripper_pos 从关闭0.0改为张开1.0，不改 RoboTwin 原始文件／任务成功检查。只有 `planner_ok=true, goal_ok=false, intervened_grasps=2` 且无 probe 失败才接受隔离负例；其他结果照实失败，**不试第二个种子来挑结果**。
- 新显式 `TACTILE_CONTACT_GEOMETRY_DIR=<trial绝对路径>/geometry`，ForceProbe一次遍历非零指触点产生私有世界点sidecar，不注入模型输入或旧raw。视频转换前负例门绑定raw/obs/frames/**geometry** SHA；完成卡/HDF5与geometry SHA再独立重验，每物理步逐指点数和载荷代理必须吻合；几何小卡仅计数/SHA且 `accepted_for_expert_training=false`。
- 只有同物理步世界点 + 该帧HDF5相机内外参才允许画**二维几何投影**；没有同帧深度图和遮挡校验时必须标“投影候选，遮挡未知”，**不是可见皮肤/指尖接触力地图**。冲量÷dt仍是未标定代理量，不可用于硬件限力。
- GPU前验单机RTX3090空闲、没有任何compute进程、Strata用户停机期限内不得提前恢复；独占 interactive GPU lease且单试次 `timeout 660s`，结束释放。若GPU被他人占用就不抢；预登记和CPU单测先于运行；试后原 RoboTwin `_base_task.py` SHA仍须为 `448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101`。私有原始侧录/HDF5/图像/视频均留gitignored本地，不进入成功专家目录或GitHub。
