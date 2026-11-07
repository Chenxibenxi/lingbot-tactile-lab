# 一次有界真实LingBotV2类“空权重”键形状检验（GPU运行前登记）

目的：仅跨过“AST/假Qwen代码已可跑”与“**真正原LingBotV2类**能否按原154800配置创建，并与原六分片1,624个权重键及形状完全匹配”之间的门。**不加载任何权重数值、不创建实际CPU/GPU权重张量、不跑模型`forward`/`sample_actions`、不训练或开启触觉RPC能力。** [现有头部验证](<STREAMED_CHECKPOINT_METADATA_AUDIT_ZH.md>)只证明index/六shard元数据匹配，不能代替本次真实类形状验证。

[隔离元实例脚本](<../src/probe_real_v2_meta.py>)固定原V2模型源码SHA `9b3c9a2a...`、154800外部`lingbotvla_cli.yaml` SHA `daa533fe...`、原索引SHA `62dbe509...`和本机可读Qwen3-VL配置SHA `edac7703...`。原154800 YAML的`tokenizer_path=/mnt/data/users/...`在本机不可达；此次**只对配置读取路径**显式改用本机已用于视觉推理的 `/mnt/relaydata/lingbot/weights/Qwen3-VL-4B-Instruct`，如造成键／形状不一致即停止而不尝试`strict=False`。独立生成未安装的补丁源码到gitignored`work/real_v2_meta_preview.py`，继承原V2代码和完整训练配置，以`tactile_enabled=False`于`with torch.device('meta')`构造真实`LingbotVlaV2Policy`；要求全部state_dict值仍为meta，精确比较1624键/各tensor形状，绝不提前附adapter。

上游Triton在**import阶段**会查CUDA能力，即使模型本身是meta；所以只有原relay单张RTX3090、Strata按用户停机、无compute、lease可用、离北京时间2026-11-02 12:00恢复≥10分钟时才申请自己interactive租约TTL600，以`timeout 240s`一次运行。输出仅脱敏键数／匹配与CUDA已分配字节数，若model ctor、源哈希、Qwen配置、RAM、任何meta值或GPU分配>512MiB异常立即失败并释放lease；不杀别的GPU进程、不修改原LingBot/RoboTwin、不给错误键“自动修正”。运行前2项CPU配置单测及原全库测试须通过。即使成功，**只说明未载权重的真实模型结构匹配，不意味着原24GB3090能承载BF16严格权重加载、反传激活或当前未批准的训练数据**。
