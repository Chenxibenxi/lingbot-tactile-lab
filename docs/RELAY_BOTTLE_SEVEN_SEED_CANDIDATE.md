# 双瓶7独立种子的候选v2切分：旧验证／测试绝不移动

新seed100013、100015各仅一次的成功规则专家回合经完整同源SHA验收后，新增249个合格因果图像→下一专家指令对。[补采结果](<RELAY_BOTTLE_ONSET_EXPANSION_RESULT.md>)已经固定其来源。为了**不把旧验证／测试回合搬进训练而泄漏**，新的CPU候选v2只把这两个seed追加到原来双瓶v1的训练折：

| 独立双瓶seed | 旧v1 | 新候选v2 |
|---|---|---|
| train | 100000、100002、100007（341对） | **另加100013、100015（共590对）** |
| validation | 100012（126对） | **仍为100012（126对）** |
| test | 100009（127对） | **仍为100009（127对）** |

旧 [`v1切分`](<../datasets/relay_pickdiverse_seed_split_candidate_v1.json>)和旧归一化文件始终不改；独立 [`v2切分`](<../datasets/relay_pickdiverse_seed_split_candidate_v2.json>)附有旧切分及每份专家来源卡SHA，`src/extend_relay_bottle_split.py`会重算所有七份专家卡与同回合物理流、成功门及HDF5，再**只从v2的五个训练seed**拟合 [`v1算法的新参数候选`](<../configs/relay_pickdiverse_train_norm_candidate_v2.json>)和[`正接触保零算法的新参数候选`](<../configs/relay_pickdiverse_contact_norm_candidate_v3.json>)；两者依然 `training_approval=false`。

**关键数据分布变化：**旧v1三训练seed原始全分箱算法的8个手指力分量中位数均为正；加入两个不同场景后的五训练seed，8/8中位数变为0（尽管8个力通道的IQR都未跌至0.1地板）。正接触保零算法的正值占完整有效分箱的范围从旧约50.56%–51.03%变为新约48.98%–49.86%，力通道IQR触0.1下限由旧4/8变为新2/8。这**不是模型成绩或证明v3归一化一定更好**，反而说明小数据集归一化对种子与接触阶段组成敏感；不应沿用旧训练折统计，也不能根据验证／测试数据事后调方法。

又只在v2的**五个训练实际seed**重验了下一专家帧接触标签：590对里，左右未来区间任一接触分别301／302对，但真正“当前过去无接触→下一帧首次接触”仅左**5**、右**6**个区间；旧v1三seed分别3／3，均未触碰验证／测试的未来标签。见[训练折首次接触v2卡](<../datasets/relay_bottle_expert_contact_onset_candidate_v2.json>)。新增seed100013右手在同一回合有短暂早期接触后另一接触段，这解释右边多出的1个区间，但不能擅称它是策略学到主动恢复。旧验证/测试各只有一个seed，无法支持普适性能/置信区间。**此v2仅供评审训练数据的候选方案，`approved_paired_episodes=[]`、无真实触觉VLA训练、无在线RL结果。**
