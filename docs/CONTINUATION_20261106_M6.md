# 运行交接：M5完成，M6采集进行中

上一个goal turn是实质进展：M5四回合完成、完整图文汇总；CPU接触观测频率诊断完成；M6固定16个采集器/来源读取器/报告/数据门实现，CPU预检16/16通过，相关9契约+3时钟测试通过；已启动真实GPU采集。

## 用户与GPU

用户明确说“可以继续用了”。实际检查GPU空、租约free后启动M6。之前secretary-digest短暂占GPU/Strata，未停止他人工作、未请求重复授权。原仓库/checkpoint只读，不自动重启Strata，GPU仅自有租约。

## 当前活进程

M6持久批次PID47000：python3 -u work/run_cwmj_m6_batch.py；日志data/reports/cwmj_m6_batch.log。每seed用work/run_cwmj_m6.sh独占lease；16个seed55–70串行，各一次，结束即时report_cwmj_m6_collection.py，最后report_cwmj_m6_batch.py形成不可覆盖data_manifest.json。原M5监督PID41204已终止且M5_COMPLETE，不要重复启动。

最新已完成55、56、57规划失败不入训练；58完整成功且严格源校验通过，56个有效窗口、20个接触窗口、6496物理步。59继续运行。应重新查询活进程和日志，不依赖此快照。观察到工具返回超时不能重跑当前seed。

## 实验结论

M4不同权重zero0/4、true1/4；全部4对在第11个动作（0始index10）、传感器历史全0时已分叉。M5同true权重、history置零：成功仍1/4且同一seed51，245动作对真实输入71；其余3seed两种输入均失败。全部同权重/同资产/起始/接触前动作门通过。未证明运行时触觉成功率收益；只保留单个成功回合的动作效率线索，不是独立确认。

报告 /home/admin001/lingbot-tactile-artifacts/cwmj-m5-closed-loop/report.md；比较图已视觉QA。报告脚本增补实际结论后已再次生成，reproducibility保存初版及更新版并更新manifest。

CPU audit_contact_observation_cadence.py：核验12回合raw/HDF5 SHA，按手指load>0与25步窗口统计末次请求之前的接触覆盖，末请求后尾段单列。请求间隔中位数约3–4秒，100ms窗口覆盖接触物理步约2.6–5.7%。不是因果效果或失败原因证明，持续接触未必需每步观察。报告/图路径contact-cadence-m4-m5/report.md，已看图。3项边界/尾段/未来时钟测试通过。

## M6数据门与代码

协议docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md已冻结并被各source_manifest SHA绑定，运行期间不要改。train55–66（12）、val67–70（4）；71–86保留独立确认。新增至少4train/2val，各>=4接触/4非接触窗口才过数据门；旧45/46只有新训练manifest显式合并才可使用，旧val49和测试51–54不进训练。尚未冻结训练预算，不自动启动拟合。

src/prepare_cwmj_m6.py生成固定隔离collector；src/cwmj_m6_samples.py严格重新读原source、同runHDF5/时钟/完成/成功、资产及私有collector源码，新增verify_collection_manifest拒绝seed/fold/stride/protocol SHA与源码漂移。src/report_cwmj_m6_collection.py保存每次完整图文；report_cwmj_m6_batch.py末尾再次源审计并生成data_manifest.json（训练未授权，只是数据可用门）。tests/test_cwmj_m6_protocol.py保护分组/确认seed禁入及篡改拒绝。

src/report_cwmj_m6_progress.py可重复生成用户可见cwmj-m6/progress.md（部分批次索引），批次本身未自动调用此脚本，需要观察进展时手动运行。src/build_tactile_seed_bank_m6.py更新datasets/tactile_seed_bank_m1_m6_v2.json，当前58entries包含M5重用test、M6预留及未来确认，按unique seed检查无跨fold泄漏；不会把同seed消融计新独立样本。完成批次后再次运行并更新研究索引/长期台账。

当前目标active且未完成；下一步完成M6批次报告QA，数据门通过才冻结3训练初始化/独立确认预算，继续正式训练和对照，不为得到正结果反复筛seed。
