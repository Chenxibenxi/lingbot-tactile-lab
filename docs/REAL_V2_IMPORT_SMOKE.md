# 隔离触觉V2补丁的真实模块导入烟测：只证明可导入

本机源码保持只读：上游 LingBot V2 `modeling_lingbot_vla_v2.py` SHA256 为 `9b3c9a2ad6823f9d58da51b732477785a1e4db0d29db80e030e094c7b605e40c`。由 [`patch_model_v2.py`](<../src/patch_model_v2.py>) 严格锚点生成 Git 忽略的 `work/model_patch_preview.py`，生成副本 SHA256 为 `a80be3e9b7f7834676a086c1ffab964e6a25ad8bc81b4572e8939e0f400b8191`；源文件未修改。GPU租约独占、Strata按用户指示停机时，以本机单张 RTX3090 真实导入这个隔离模块，打印 `REAL_ISOLATED_MODEL_MODULE_IMPORT_OK FlowMatchingV2 ... NVIDIA GeForce RTX 3090`；无checkpoint、无模型实例、无权重训练、无服务端修改，结束租约释放/GPU compute进程归零。

**重要依赖发现：**把GPU藏起来时，真实import失败在上游 `lingbotvla/ops/group_gemm` 的预调配置枚举阶段（调用 `torch.cuda.get_device_capability()`），不是触觉方法的Python语法错误。CPU-only AST/方法假体测试可以不依赖CUDA，但**真实模块导入需已取得租约并在唯一3090可用时进行**；不应为了得到CPU import通过而篡改/绕过共享上游的GPU预调逻辑。

复现过程（只在GPU空闲且有租约时；不要在其它服务占用时执行）：

1. 先用 `src/patch_model_v2.py --out work/<新的、未使用的名字>.py --create` 生成**未安装**副本；确认原源码SHA不变。副本不改官方模型，也不会加载/修改26GB视觉基座。
2. `python3 /home/admin001/ai-video-studio/tools/gpu_lease.py acquire --owner lingbot-tactile-lab/patch-module-import --ttl 600 --priority interactive --task 'bounded model import only'`，检查 Strata 未运行、`nvidia-smi` 无其他 compute 进程；失败即让位。
3. 用 `CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/src:/mnt/relaydata/lingbot/lingbot-vla-v2"` 在 eval310 Python 下以 `importlib.util.spec_from_file_location('lingbotvla.models.vla.lingbot_vla._tactile_preview', '<副本路径>')` 导入；设置 `timeout 120s`，**只检查**`FlowMatchingV2`类存在，不初始化/训练模型。
4. 最终块无论成功/失败释放本人租约；不要因此提前恢复 Strata，恢复由用户设定的北京时间2026-11-02 12:00独立提醒执行。

**未越过的门槛：**补丁尚未在真实V2构造器、BF16旧checkpoint加载、真实`forward`/`sample_actions`、训练loss、RPC批处理或闭环环境通过；模型能力 `tactile_v1` 必须继续拒绝，`datasets/provenance_v1.json:approved_paired_episodes=[]` 和两任务norm候选均不允许实际触觉训练。此项仅排除“生成源无法真实导入”这一较早的集成风险。
