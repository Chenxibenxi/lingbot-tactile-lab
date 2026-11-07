# 用户批准新seed100022单次在线专家试采预算：仍须先完成录制入口和GPU安全门

用户批准先设计并最多在一个新seed上试采一次、失败不可换seed刷成功；当前[CPU在线动作生成器与时钟适配器](<H50_ONLINE_EXPERT_BEGINNER_ZH.md>)均已完成假场景测试，[隔离录制入口](<../src/record_online_expert_demo.py>)已写出原相机→HDF5→成功门与动作sidecar程序并通过**CPU dry-run/无租约拒绝**测试，但没有真实新seed运行验证，`gpu_run_allowed_now=false`仍保留直到现场门通过。用户已另行明确批准单次物理上限锁为**10,000步**（原 `take_action` 的一条TOPP不可中断，越限后立停）和外层**900秒**，高层动作最多128条；这不是已有seed100013 cap8000的自动延伸，更不能由单次贴边成功推断seed100022会成功。虽预算已批，**在线真实入口通过CPU负控、用户另行确认工作台GPU后端已停止、3090实际无compute/显存足够并拿独占租约**全部满足之前，仍不得启动GPU。

输入仅原RoboTwin源三SHA固定的本机`pick_diverse_bottles`任务/在线规则三阶段与**当前新回合**机器人/双瓶状态；禁用旧100013路标与旧图像/未来接触、禁改原官方源码。独立忽略目录`data/paired/relay_pickdiverse_s100022_online_h50_expert0/`必须全新，`lock/status`亦不得存在。隔离overlay使每个真的`Base_Task.take_action(qpos)`标明`policy_action`物理段，动作前取同回合`_take_picture`（三摄RGB/状态与帧marker）、之后读实际应用前后物理步及真实关节偏差；失败保存脱敏诊断不登记训练。成功仅当原任务目标/规划均通过、至少50条连续真实高层动作、raw/obs/frames/HDF5/completion及在线动作sidecar各SHA一致、source host/原始源与物理clock全验证后写candidate成功卡；仍仅属**一个新的train候选seed**，不能用同seed100013验证或将旧规则`move`标签混入，train/validation/test完整独立数据与负责人数据/norm审批仍缺，**adapter训练/RPC禁开**。

用户偏好开工时自行停止工作台GPU后端；任务完成或失败时持本人租约只恢复一次工作台Strata、核`health.loaded=true`再release。若工作台模型仍占卡、他人租约或waiter，立即让位，不kill别人；当前人类预算确认已获得，但用户尚未报告已停工作台GPU，真实录制入口/HDF5门尚未验完；试前dry-run与空卡门未通过前只提交方案/代码/CPU测试，**不消耗那一次新seed试采**。
