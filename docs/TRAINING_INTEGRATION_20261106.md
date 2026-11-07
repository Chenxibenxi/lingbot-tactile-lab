# 2026-11-06 正式训练接入：有界实验协议

授权来源：用户本轮要求“先补正式融入训练的实验，然后推进实验进行”，并已自行停止 Strata。此次授权覆盖本协议的技术验证与补数据；旧文档的“尚未授权下一步”是历史状态，不能把它当成当前停止工作的理由。既有数据审批字段保持原样，科学有效性门仍须通过。绝不自动启动 Strata，不改原比赛源码/权重，不推送远端。

## 执行顺序与预算

1. 真实 H50 训练前向/反向：固定 seed100023 首个因果有效 anchor，三摄 RGB + 当前14D状态 + 过去触觉，标签为接下来50次真实 take_action；原生 FeatureTransform 的 robotwin mapping/normalizer，将14D映射至55D并严格mask。原154800六分片逐SHA、1,624键strict load；冻结基座，只挂65,664参数零初始化adapter。固定noise seed97026/time=0.5，先比训练模式视觉/零adapter loss恒等，再一次flow-matching backward，无optimizer.step。上限900秒、CUDA allocated峰22GiB，异常退出。预期零投影使第一步GRU梯度为0，但输出投影应有非零有限梯度；任何基座梯度拒收。固定log1p身份尺度仅用于不更新参数的技术实验，不冒称正式归一化。
2. 独立验证补采：预登记新seed100026、100027，validation-only，每seed只一次，原stride20在线规则控制、10,000物理步/900秒。成功且≥50真实动作后才允许同源HDF5/completion；全部失败也保留，禁止重标为成功或重跑筛选。原test种子不访问，不用失败动作拟合。
3. 正式小训练须在同源 train/validation H50均有效后，先检查有效anchors的接触覆盖、训练折归一化可拟合性，再建立全新实验审批manifest（不覆盖旧候选）。若数据无接触/无独立验证/时钟不符，停止优化并报告实际缺口。首次技术拟合建议固定≤20步，batch1，只存adapter/optimizer及哈希。此预算仅在数据门与真实反传门通过后适用；单train/val seed不构成泛化证据。
4. 同数据对照：冻结视觉基线A；同容量、同预算的无触觉（有效mask保留、力特征零）adapter A2；真实触觉B；仅过去bins重排C。一致train/val、噪声、动作mask、步数和初始化，按seed报告而非把重叠窗口当独立样本。接触数据不足时不执行虚假的触觉增益比较。

## 接口与验收

`src/probe_h50_training_backward.py --out data/reports/<fresh>.json` 复用 `OnlineH50Candidate` 和原生训练transform；返回来源SHA、loss、每个adapter梯度、基座梯度数、峰值显存、optimizer_steps。缺失传感器仍走旧视觉，未来接触只允许出现在监督目标。

后续真实优化需再记录split/card/norm/code SHA、训练种子、学习率/优化器、adapter safetensors及验证分层指标。RPC默认关闭，闭环模型收益须另有匹配请求预算的实测；离线loss、梯度非零与零初始化恒等都不是触觉收益。无reward/terminated/truncated/next-observation合同之前不进入自动RL。

## 回滚与资源

新增入口/协议，旧数据/代码/权重只读。报告与采集输出使用独立目录且不覆盖；失败保留原因。每次GPU检查无外部compute、Strata未加载、独占租约；timeout强制结束自己的进程后只释放自己的lease。不得调用任何Strata start/load。

## 数据诊断后的预登记修订（任何新数据采集前）

首批实测：seed100023的8个H50有效anchor全部位于接触前（物理步89–1068），有效输入正力bins为0。这不能支持触觉归一化/效用比较。保留stride20验证批次，不把验证结果用于调参。新增一个独立stride10采集协议：100028固定train，100029固定validation；每seed只一次，上限128真实调用、10,000物理步、900秒，test保持不动。目的仅是检验更密的在线规划路径是否提供“接触后仍有50条真实动作”的成功专家窗口；不缩短H50、不补假动作、不把失败轨迹当专家。此变更由训练输入覆盖审计驱动，不根据验证成绩选择策略；禁止与stride20跨seed差异宣称因果收益。若新协议也无合格train/val接触数据，本批停止优化，保留全部失败和缺口。
