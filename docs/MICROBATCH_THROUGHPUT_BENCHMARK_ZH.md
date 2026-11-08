# Micro-batch 1/2/4 吞吐基准：实施前设计

2026-11-08。回应显存未用满能否提高 batch 的问题。此处仅准备性能基准，不改变已冻结 M8 权重或运行中的反馈试验，不声称已有 batch2/4 实测。

## 当前实现事实

M8 每步选择一个训练窗口，将单样本 batch 送入模型后立即反向、裁剪并 optimizer.step。micro-batch=1、累积=1、global-batch=1。AdamW lr=1e-4、weight_decay=.01、clip=1。骨干冻结、适配器FP32，模型主体BF16。`probe_h50_training_backward.validate_batch` 对 actions/joint_mask 的形状显式要求(1,50,55)；预处理对 image_grid_thw 与其他张量采用不同的 batch 维处理。

因此不能仅把 batch 标量改成2或4，也不能无差别拼接全部输入。需要先核对真实 transformed samples 的图像 patch/grid、文本 token、state/action/掩码维度，使用模型实际要求的打包方式。不能复制同一窗口冒充多样本吞吐，也不能把尾部无效目标改成有效来减少处理成本。

## 预定性能比较

micro-batch取1/2/4；全局batch统一4，累积分别4/2/1。每种配置从同一独立临时适配器和optimizer状态开始，固定32个训练窗口顺序，只取已批准训练fold，不碰确认seed。测量时分别记录优化步数、真实样本曝光数、有效目标数、总耗时、samples/s、有效目标/s、峰值allocated/reserved显存及设备显存。预热后重置峰值统计，显式CUDA同步计时，模型载入和数据预处理耗时单列。

累积损失的归一化必须与真实有效目标权重一致，不默认把不同有效长度样本的loss简单平均就等价。第一步需用显式固定noise/time检查单样本累积与并行batch的loss/梯度差，确认模型返回loss的reduction语义之后才冻结容差。性能提升不以梯度完全逐位相同为前提，但数值偏差必须解释，不能把不同目标归一化当加速。

每种配置分别在新进程运行，以隔离内存碎片。OOM记录为该配置不可用，不反复重试、不降分辨率或减少H50后继续称同一基准。只写临时基准产物，不覆盖历史adapter或optimizer，不提交为正式研究checkpoint。

## 就绪前置项

尚需：真实样本形状清单、CPU collator和尾部mask契约测试、loss归一化审计、独立有界benchmark runner、固定32窗口清单与source SHA、GPU资源检查和自有租约。运行顺序排在当前反馈技术批次之后，禁止争抢GPU或自动停止Strata。

即使batch4吞吐更高，也不能把旧实验的相同步数解释为相同训练量。后续正式训练需另行固定样本曝光量、优化步数、学习率和global-batch；所有条件使用相同配置，再检验闭环收益。

## 2026-11-08 损失归一化源代码审计

实际执行的 `work/real_v2_meta_preview.py:1338–1359` 使用 `sum(loss * joint_mask) / sum(joint_mask)`，不是每个样本 loss 的简单平均。无效尾部较多的窗口分母较小。将 micro loss 一律除以累积次数，会改变不同有效长度样本的相对权重。

更关键的是，原始154800配置开启 `use_moe=true`，`router_z_loss_coeff=0.0001`、`sequence_wise_loss_coeff=0.001`、`sequence_wise_mode=per_sequence`。M8 runner只覆盖use_cache和num_steps，没有关闭这些辅助损失。`align_params={}`，因此深度与视频对齐损失走零分支。不能直接声称总loss仅含动作项，也不能把整个total_loss按有效动作数缩放。

后续基准需分别加权：动作项乘 N_micro/N_global（N为实际joint_mask总和）；固定每样本动作token数且per_sequence模式下，路由与序列辅助项按 B_micro/B_global。模型输出的loss_dict中router_z_loss已经detach，不能拿它替代可微辅助损失。当前返回值可保留计算图地构造 `action_weight * output[1] + sample_weight * (output[0] - output[1])`，但只允许在对齐损失确实关闭、辅助项的token数与归约语义完成核验后使用。global序列平衡模式不适用这条累积等价假设。

源代码归约审计已完成；真实多样本张量打包、路由辅助项的梯度可达性及固定noise/time下的梯度等价检查尚未完成。此处没有修改冻结训练代码，也没有启动新的GPU基准。下一步先核验路由辅助损失实现的平均单位，再实现独立基准的分项累积及CPU回归测试。

### 分项累积的CPU验证

已核验原始 `lingbotvla/ops/triton_moe_loss.py:410–428`：per_sequence平衡项先在各序列内求均值，再对序列求均值；router-z项对固定长度动作token求均值。因此辅助项在每样本路由token数相等时按样本数加权。新增独立 `src/microbatch_loss_weighting.py`，拒绝global模式、对齐损失开启或未经确认的token长度契约，保留辅助项计算图。

`tests/test_microbatch_loss_weighting.py` 的两项CPU测试通过：用有效长度1/5/2/4的四个样本验证micro1/2/4的加权loss及参数梯度与全batch一致，并明确检出简单平均的不等价；不支持的配置及无效计数会被拒绝。这是合成可微目标的归约验证，不代表真实VLA模型、多模态collator或Triton GPU数值等价已通过。尚需真实模型固定noise/time对照；现有冻结训练runner未导入本模块。

### 多样本输入打包

实际preview的embed_prefix接受images `[B,N,L,D]`，展开为`[B*N,L,D]`；grid既支持`[B,N,3]`也支持展开布局。新增独立CPU collator选择明确的`[B,N,3]`布局，将原单样本`[N,3]`按样本堆叠，其他已有singleton batch维的张量沿首维拼接。只接受train fold、相同形状/dtype的CPU张量；不自动填充不同长度文本，也不改变原有mask。teacher尾部补全仍交给现有prepare_censored_flow。

4项microbatch相关CPU测试通过，其中新增两项验证相机/触觉样本顺序、有效长度1/5/23/50的联合mask、无效尾部teacher补全及输出不别名修改原样本；拒绝heldout样本、形状不一致和padding/valid冲突。测试使用合成张量，仅验证已审计接口契约，真实训练数据shape inventory尚未生成，完整模型batch2/4仍未验证。

### 真实数据CPU盘点通过

已在CUDA_VISIBLE_DEVICES为空的进程中调用原始M7/M8共用transform，重建全部442个样本（train 402），transformed_dataset_sha256与历史M7 r0完全一致。没有初始化CUDA，也没有加载模型权重。按预定等距索引固定32个唯一train窗口，micro1/2/4共32/16/8个组全部通过真实数据collator。清单保存为datasets/microbatch_throughput_windows_v1.json，完整证据data/reports/microbatch_real_cpu_inventory_v1.json。

真实单样本形状：images [1,3,256,1536]、grid [3,3]、文本[1,72]、state [1,55]、actions/joint_mask [1,50,55]、触觉history [1,5,4,3]、触觉mask [1,5,4]。32个窗口各字段形状和dtype一致。此结果证明当前选定输入可打包，不代表GPU forward/backward或显存预算已通过。下一步实现有界GPU基准runner并检查真实路由token及分项梯度，排在当前反馈批次之后。

### GPU runner已实现，尚未执行

新增run_microbatch_benchmark.py和自有租约启动脚本。每种micro单独进程，最多1000秒，allocated上限22GiB；发生OOM/非有限梯度/归约契约不满足即失败，不自动重试。复用严格原始权重加载器与固定数据指纹。32窗口各自用冻结视觉teacher补全尾部，固定CPU生成的noise与0.2–0.8 flow time，不冒充原训练随机分布。

各进程先用相同micro1/global4做一个共同预处理更新，使零输出投影获得非零值；在相同状态下保存一次未裁剪梯度供事后比较。之后按本micro预热一步，再恢复共同状态、重建AdamW，正式计时8个global4更新。模型加载、数据转换、teacher耗时单列。实时观察各MoE层路由token数，若各样本/各调用不一致则拒绝分项累积。只保存报告与梯度诊断，不保存训练checkpoint。

启动脚本语法、Python编译、CLI帮助与4项CPU回归测试通过。尚需冻结数值比较容差、补齐跨进程起始状态/teacher/noise的哈希核验及最终汇总图文报告，再在反馈批次结束后启动。当前不能宣称该runner已通过GPU验证。

### 运行前数值标准与汇总器

已在datasets/microbatch_numeric_screen_v1.json预设BF16工程筛查阈值：整体梯度相对L2≤1%、逐参数差≤1e-6+5%参考范数、整体方向cosine≥0.999、loss差≤1e-6+0.5%参考loss。所有条件同时满足才可推荐吞吐配置；这些是工程容差，不是触觉收益统计标准，不在看见结果后放宽。

runner现保存共同预处理更新后的适配器、teacher、noise/time和样本/阈值清单哈希。汇总器先核对这些及实际路由token数，再比较未裁剪梯度，并生成吞吐/峰值显存图及中文方法、结果、局限和复现报告。新增比较器CPU测试验证方向、尺度、局部参数偏差和输入不一致会被检出，6项microbatch测试通过。GPU运行和真实汇总图仍未执行。

### 有限队列与启动条件

run_microbatch_batch.py依次执行1/2/4并自动生成一次最终图文报告。每个配置最多一次尝试，失败保留，不重试或覆盖；GPU被其他任务占用则记录暂停并退出。启动前必须确认反馈技术批次的16个注册组合全部已有终态，随后再次检查实际GPU占用并获取自有租约。冻结清单覆盖本基准代码、输入窗口清单、数值阈值和CPU盘点证据；原训练依赖继续走M7验证器和原模型可复现检查。后台队列尚未启动。
