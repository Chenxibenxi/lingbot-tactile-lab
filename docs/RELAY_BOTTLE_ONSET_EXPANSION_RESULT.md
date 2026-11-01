# 双瓶任务：固定两seed新增同源专家试片（候选、未训练）

依据[试前冻结的补采方案](<RELAY_BOTTLE_ONSET_EXPANSION.md>)，只在本机中继机单张RTX3090空闲且取得租约时，依次对新的物理seed `100013`、`100015` 各跑**一次**规则专家，没有事后换seed挑成功；两次均通过planner、任务目标、同回合raw/obs/frames/HDF5及完成门SHA。每次运行结束即释放自己的租约，未提前恢复Strata；原RoboTwin `_base_task.py` SHA仍是`448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101`。

| 独立seed | 原始物理步 / 保存帧 | 有效因果图像→下一专家14D对 | 当前已有接触的对数 | 左／右“当前未接触→下一帧首次接触”区间 |
|---|---:|---:|---:|---:|
| 100013 | 1,609 / 121 | 111 | 63 | 1 / **2** |
| 100015 | 2,004 / 148 | 138 | 62 | 1 / 1 |
| **本次新增** | — | **249** | **125** | **2 / 3** |

`100013`右手在早期物理step596出现短接触，后来保存帧区间748–763又出现一次“此前窗口无接触→未来区间有接触”，原物理流右手共有5段接触；这是**同一回合的两个区间**，不能夸为两个独立seed、更不能擅断是专家主动纠错或人手滑移。相比旧双瓶候选训练折三seed左右各3个新接触起点，本次只提高**候选库存**，没有进入旧训练折，因此不能把旧训练折计数擅改成左5/右6，更没有辅助头训练数据获批。

独立新增专家卡只在 `datasets/cards/relay_pickdiverse_s100013_onsetpilot_expert0.json` 和 `datasets/cards/relay_pickdiverse_s100015_onsetpilot_expert0.json`；脱敏双试审计卡 [`relay_bottle_onset_expansion_v1.json`](<../datasets/expansion_diagnostics/relay_bottle_onset_expansion_v1.json>)逐次绑定原卡和状态SHA，并证明旧3/1/1候选split哈希不变。私有原始RGB、HDF5、物理接触JSONL/视频仍在Git忽略`data/paired/`。成功专家候选现在**提锅5条＋双瓶7条**；双瓶累计旧594＋新249＝**843对候选**，但只有旧594对属于被冻结的v1切分。若要评审新v2切分，须先独立按实际seed重新分fold、只在其新训练折拟归一化并冻结评测；**`training_approval=false`、没有模型训练或在线RL结果。**
