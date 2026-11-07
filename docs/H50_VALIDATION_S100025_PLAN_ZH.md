# 用户另批独立validation-only seed100025 单次试采：无自动Strata恢复

[旧seed100024验证回合](<H50_VALIDATION_S100024_RESULT_ZH.md>)虽然实有80条`Base_Task.take_action`/9559物理步，却未达到双瓶任务目标，**没有**专家HDF5/完成卡/H50标签；必须保留失败证据、不重复100024或把它改成训练。用户本次**只批准一条新独立seed100025**（此前trial/lock/status均未存在），`pick_diverse_bottles`、只作validation、原规则专家`stride=20`、物理上限10,000步、墙钟900秒、最多128条高层动作。**另有必要条件**：用户亲自在工作台停GPU后端并明确说已停；代理再实测唯一RTX3090、无其他compute/free≥16GiB、Strata unit inactive、工作台报告running=false、独占本作业GPU lease可用，才运行本单次尝试。此前仅同意了试采，尚**未**确认工作台已停；在其明确确认前不可调用GPU采集脚本或代用户停止当前加载的Strata。

源代码生成器[`patch_online_expert_validation_s100025.py`](<../src/patch_online_expert_validation_s100025.py>)只从原seed100022的预登记/录制器**固定SHA**造私有Git忽略`work/online_expert_stride20_s100025_validation_v1/`隔离文件：只换seed、新trial ID、独立lease owner、`validation_candidate_only_not_assigned`和显式stride20，不改RoboTwin原Base_Task/双瓶任务/动作规划器。私有wrapper`work/run_online_expert_s100025_validation_once.sh`会先锁GPU实况/不重复路径/源码/Git clean/ffmpeg，然后TTL1500租约＋外层timeout900；**EXIT无论成功失败只release本人租约、报告GPU与Strata实际状态，绝不自动POST工作台start、Strata load或systemctl start**，遵循[最新资源规程](<GPU_NO_AUTORESTORE_POLICY_ZH.md>)。新试采不改变100022、100023、100024原始文件。

终止后逐核真实任务goal、真调用数和连续物理时钟、同回合raw/obs/frames/在线动作/HDF5/completion及所有SHA；如果goal false或calls<50，明确拒收且不生成验证H50。即便成功也只作**未获批准的验证候选**，不挪train、不借validation调stride/模型、不自动fit normalization/训练VLA/推GitHub。现只有train意图seed100023的一物理seed八个相关窗，训练fold/归一化/3090训练forward-backward及同数据A/B/C仍须另行批准。

**事后结果**：用户确已停后端并授权执行本次，但原`setup_demo`报双瓶初始`001_bottle`不稳定，零次高层动作、无实测物理步/专家HDF5；seed100025已消费不得重跑，见[阴性结果](<H50_VALIDATION_S100025_RESULT_ZH.md>)。lease已释放，Strata保持`inactive`，**没有自动拉起**。
