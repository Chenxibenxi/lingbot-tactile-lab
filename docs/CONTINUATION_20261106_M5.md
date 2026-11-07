# 运行交接：M4完成，M5进行中

用户明确持续授权GPU训练、本机仿真、CWMJ资产替换及长期自主推进；每实验完整图文汇报。长期goal仍active，不要因探索性结果标complete。原仓库/checkpoint只读，不能重启Strata。

## 当前执行

持久监督进程PID41204，`python3 -u work/continue_m4_then_m5.py`，日志data/reports/continue_m4_then_m5.log。它正在顺序运行M5四个zero-input回合，成功后调用report_cwmj_m5_batch.py。不要重复启动。M4原exec会话98792已以143退出，其最后54zero子进程保留完成；监督器等待退出后释放自有租约、生成报告，再运行54true。未重跑或覆盖任何试验。

M4完整8回合：zero0/4、true1/4（51成功71动作）。四对初始RGB/状态/首10动作门通过。逐轨迹诊断全部在动作10（0起始）、请求1发生分叉，输入历史全0且RGB/状态完全相同，说明训练权重影响零力策略，不能直接归因运行时触觉。

M5固定M4 true权重、输入history置0，mask/age/valid保留。51已成功245动作，相对原true71；同权重/资产/初始/首10动作及首次接触前所有动作一致性门通过。当前52正在加载/运行，之后53、54。其余未得结果，不外推。M5属于看到M4部分结果后设计的机制探索，重用测试seed，不是独立确认。

## 报告

用户路径 /home/admin001/lingbot-tactile-artifacts/持续研究索引.md；cwmj-m4-closed-loop/report.md及cwmj-m4-divergence/report.md均完成并看过图。M5每回合结束即自动图文报告cwmj-m5-closed-loop/{seed}_zero/report.md。M5汇总待监督器完成后QA；更新索引和长期台账，补M5 Seed Bank记录（不入训练）。9个相关unittest通过；pytest未安装，已改用unittest。代码编译及git diff --check通过。

## 下一阶段

已写docs/CWMJ_M6_DATA_EXPANSION_PLAN_ZH.md：train55–66共12，val67–70共4；候选确认71–86不采集。CPU全局data/paired目录检查无55–86历史占用，运行时仍必须fresh-seed门。src/prepare_cwmj_m6.py生成器完成16seed CPU编译，尚未生成私有目录或运行。新增数据门4train2val各>=4接触/4非接触窗口，先过门再冻结3训练随机初始化的训练/独立确认预算。不要用M4/M5测试数据当训练专家；失败策略轨迹不是恢复监督。

M6后续可用M4 collector reader/report/launcher模式，但需显式M6命名/55–70fold绑定和报告预检拒绝分类。先完成M5汇总及解释，再开始下一阶段，保持单GPU自有租约检查。
