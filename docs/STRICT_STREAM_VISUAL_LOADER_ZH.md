# 原154800完整视觉权重的逐分片严格值复制：**目前仅合成微型证明**

原deploy将六片FP32（文件合计约23.2GiB）全部装入一个字典再load_state_dict，因此受[主存门](<REAL_VLA_MEMORY_PREFLIGHT_ZH.md>)拒绝。与直接改`strict=False`不同，新[逐分片复制契约](<../src/strict_stream_visual_weights.py>)首先用index SHA、六个safetensors头部逐键/逐形状/逐FP32 dtype审计与一个**已分配的视觉原模型**state_dict全键比较；任何差异都在**写入第一个目标参数之前**拒绝。通过后每次只打开一个分片，将`get_tensor(key)`逐个copy_到已分配模型张量（允许FP32→BF16转换）；已复制键精确等于完整index才给出`StrictStreamResult`，然后[零输出adapter挂载契约](<../src/model_attach_contract.py>)才接受该特定结果或PyTorch原生strict load结果。合成中途I/O异常及未完成预检都会污染“可用于挂载”状态，必须弃用当前模型/进程。

本阶段的CPU[小模型实验](<../tests/test_strict_stream_visual_weights.py>)只使用**六个极小人工分片**验证FP32/BF16值一致、缺键/形状/RAM门失败不改已有参数、第二片I/O故障禁止事后挂触觉adapter及未分配meta不可作为目标。**没有对真实23.2GiB权重调用get_tensor、没有真实VLA权重在BF16 GPU上严格加载，没有全模型forward或梯度。** [真实meta结构门](<REAL_VLA_META_PROBE_RESULT_ZH.md>)已经证实旧视觉检查点1624键/形状匹配，但不能替代值的读取。原文件未复制进仓。

已按[逐片限内存SHA工具](<../src/pin_shard_sha.py>)只读遍历原154800全部六个分片，输出[六分片字节SHA清单](<../datasets/relay_154800_shard_sha_v1.json>)；这只固定当时文件内容，**不**追溯其来源批准或读取模型张量。真实有界试验前尚需：①实际加载时逐分片比对该清单，index SHA本身不保护数值；②单GPU lease/Strata恢复时限、核对可用CPU主存≥最大分片+余量及GPU空闲≥BF16视觉权重估计12.45GB+可量化余量；③源原YAML/本机Qwen config/隔离补丁SHA固定、`with torch.device('meta')`真实模型先验门、显式BF16+`to_empty(cuda)`内存上限；④**逐片值复制成功之后**核对GPU分配上限与原视觉-only BF16严格加载数值输出/零adapter恒等，不能拿proxy回归冒充真实VLA。训练仍需独立的真实成对数据/归一化批准和梯度显存门；目前均不满足。
