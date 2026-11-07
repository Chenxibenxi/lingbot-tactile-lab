# 中继机自建同回合候选数据集 v2：12 专家回合、1361 对、零训练批准

使用既有 [`build_relay_training_index.py`](<../src/build_relay_training_index.py>) 的显式 `--bottle-v2` 模式，CPU重验本机单RTX3090产生的**提锅5个＋双瓶7个**独立seed专家回合的规则成功卡、物理接触raw/观测帧完成门及同回合HDF5哈希，再真正解码每对当前观察的三路RGB **1361×3＝4083张**。因果输入只含拍照步之前的25个物理步接触；标签是下一保存帧**专家14D指令**，不是未来接触。选项默认不启用，旧v1 1112对私有索引和脱敏卡都没有改。

| 任务 | 训练折 | 验证折 | 测试折 | 当前过去已有非零接触（训/验/测） |
|---|---:|---:|---:|---:|
| 提锅（旧5seed保持不变） | 309 | 104 | 105 | 73／25／25 |
| 双瓶（原3训练seed追加固定2新seed，旧heldout冻结） | 590 | 126 | 127 | 304／70／72 |
| **两任务合计** | **899** | **230** | **232** | **377／95／97** |

逐行核验：将旧v1私有索引和新v2私有索引按 `(task, trial_id, frame_index)` 比较，旧**1112行完全相同**，只增加seed100013／100015两个成功专家的**249行**，全部是双瓶的候选训练折；总 **1361行**，没有把旧val/test挪到train，没有加入受控失败／视觉策略自回放／LXC样本。私有逐样本索引位于Git忽略的 `data/reports/relay_paired_training_index_candidate_v2.jsonl`，不能上传；Git仅保存[来源、分折、逐行私有SHA与计数脱敏卡](<../datasets/relay_paired_training_index_candidate_v2.json>)。双瓶归一化只取新训练折，见[七seed新切分与分布变化](<RELAY_BOTTLE_SEVEN_SEED_CANDIDATE.md>)。

本索引只证明**有可复核的数据来源与切分，不代表样本已获准模型拟合或评测成绩**：`approved_paired_episodes=[]`，`training_approval=false`。下一步仍需评审同任务同seed视觉-only/打乱触觉/正确触觉对照、负例语义和相同heldout评分；首次接触监督只有双瓶新训练折左5、右6个区间，见[首次接触稀缺审计](<BOTTLE_EXPERT_ONSET_LIMIT.md>)，不应直接训练此辅助头。
