# CWMJ M3 资产初始化对照

在运行前冻结：重复M2八个setup诊断seed33–40，保留原随机位置/初始朝向抽样与原稳定性检查，替换两侧瓶为同一个来源可核验的CWMJ酒瓶候选。原始20种瓶分布变为单酒瓶形状，是新场景，不能沿用原任务成功率解释。仅setup、不执行专家或策略、不生成训练数据。

来源：cwmj-horizontal/results/coacd_targeted/wine_bottle_main/visual_probe/candidate。XML与3个CoACD凸包核验原manifest；实际OBJ visual按XML scale合并，源.msh缺失，不能声称逐像素保真。统一Z-up转Y-up，按template14高度均匀缩放至0.232023m；质量按s³、惯量按s⁵变换，COM及惯量轴同步旋转。PhysX材料保持原场景默认值，不把MuJoCo solimp/solref当作等价参数。质量/几何/惯量一起变化，此实验只能判定整套候选是否落稳，不能单独归因于凸分解。

转换v1的center/extents/orientation元数据单位不完整，在物理运行前弃用；v2统一到米并计算新visual bounds。两版本都保留。接触/功能点由template14推导，尚未通过抓取验证；通过落稳也不能自动放行训练。

验收：8个seed均记录完整2500稳定性物理步，原check_stable返回接受且最后200步每瓶旋转≤3°。保留原拒绝，不重试筛seed，不关闭门。随后独立新seed专家抓取验证才可考虑新场景数据契约。
