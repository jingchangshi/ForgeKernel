# 实施风险与验收登记表

本表按会话重建，已删除旧方案中的错误推断；详细设计以[plan.md](plan.md)为准。状态“待实现”不是已修复。历史/tmp日志已丢失，历史实验须重新留证。

| ID | 问题 | 处理与验收 | 当前状态 |
|---|---|---|---|
| E01 | 系统Python无torch或agent/job环境不同 | adapter统一CANN+绝对Python；检查各子进程实际import来源 | 本次只读import通过；adapter待实现 |
| E02 | CANN路径少一层cann、libhccl缺失 | 正确setup路径；记录实际compiler与库来源 | 当前import通过 |
| E03 | CANN9.1曾有asc_pack_to_high编译错误 | 采用既定9.2工具链，正式BF16冷编译确认 | 设备测试待重验 |
| E04 | 源码commit不能证明build产物一致 | commit+dirty+native hash+cold smoke，依赖变化重出证据 | hash已记录；smoke待实施 |
| E05 | TileLang缓存掩盖工具链差异 | 独立TILELANG_CACHE_DIR；最终验收新cache；不清全局缓存 | 待实现 |
| E06 | /tmp脚本inspect.py遮蔽stdlib | 在专用workspace运行，检查模块来源，不使用冲突文件名 | 新脚本采用inspect_environment.py |
| C01 | loop/submit生成文件手改被覆盖 | 修改配置源；新建/恢复/最早judge传播测试 | 待实现 |
| C02 | agent_sandbox只传主loop，judge仍失败 | policy构造点均传播机器开关 | 待实现 |
| C03 | mode_of非CUDA静默进cutedsl | 显式语言注册；未知语言失败 | 待实现 |
| C04 | arch只接受sm、KB缺index | ascend950pr注册；KB/index、prompt与provenance一致 | 待实现 |
| C05 | TileLang源码路径未提供 | 显式提供既定checkout的DSL源码路径 | 待实现 |
| D01 | 只写CUDA_VISIBLE_DEVICES | device kind→NPU可见性；物理/逻辑ID分开 | 待实现 |
| D02 | FK锁不能排除其他用户 | 测前/后占用、外部调度，明确合作锁边界 | 待实施 |
| D03 | 外层与FK嵌套同锁死锁 | 单一锁所有者/锁顺序，覆盖中断清理 | 待实现 |
| D04 | 复用旧卡1 Alarm状态 | 每run重查，不能擅自换卡或reset | 不在本轮初始化设备 |
| J01 | op树位置不符合submit派生base逻辑 | 保持<base>/operators/<op>并跑真实submit | 待实现 |
| J02 | 缺finished marker或模式吞错误 | 成功marker、失败退出码、parse负例 | 待实现 |
| J03 | sealed/recon树遗漏NPU环境 | 与candidate相同设备与环境、逐模式测试 | 待实现 |
| P01 | ncu_available未阻止driver任务 | capability落实到driver；无NCU不买job不重试 | 待实现 |
| P02 | NVIDIA clock/CUPTI/SASS skills误导 | NPU禁用不支持通道，明确N/A | 待实现 |
| N01 | VF外copy做BF16→FP32 | 同dtype搬运、VF内cast，真实BF16编译 | 待实现 |
| N02 | 设备float64验证在NPU失败 | CPU FP64算误差；RNG fork设备化；CUDA回归 | 待实现 |
| N03 | TF32开关在NPU无意义 | 明确N/A，不能谎报精度测试 | 待实现 |
| N04 | MATCH宽松floor不能证明FP32统计 | 保留集、错误BF16累加负例、冻结阈值 | 待实现 |
| N05 | 厂商与任务数值语义不同 | 同一数学judge，核实BF16舍入/epsilon/rstd接口 | 待实施 |
| T01 | geomean掩盖decode host-bound | 比率聚合MAX，保留逐shape，拒绝[0.04,0.88]负例 | 待实现 |
| T02 | _task_ref_alone丢audit | emission→parse→archive→gate全链保留 | 待实现 |
| T03 | NPU graph存在不等于计时合格 | **部分解除**：msprof设备时长可用并已用于实测；仍需正式资格（复用次数/单位/同basis/稳定性） | 部分验证 |
| T04 | plain_batch误称纯device | host-bound拒绝；显式e2e只报端到端 | 待实现 |
| T05 | target仍比PyTorch快1.2倍 | vendor独立基线、target优先级测试、逐shape门 | 待实现 |
| T06 | 静态target随设备漂移失真 | 同条件交替实测vendor，漂移资格和重测 | 待实施 |
| T07 | 0.3s热身/120s总预算被当定理 | 热身稳定性实测；区分单job和总run预算 | 待实施 |
| A01 | off模式Bash可写外部/读oracle/联网 | 如实记录边界；评测摘要与overlay、独立验收 | 待实现 |
| A02 | 将curl/pip/ssh一律禁止当用户要求 | network/cluster独立配置，依赖frozen，普通工具可用 | 最终方案已修正 |
| A03 | hook按URL文本误伤grep等 | 精简策略、负例测试，不泛化open/chr/base64拦截 | 待实现 |
| A04 | Claude路径deny双斜杠被清理 | 保留语法，绑定CLI版本做行为测试；编辑用Edit规则 | 待实施 |
| A05 | 网关环境块cwd非真实workspace | prompt明确绝对WORKSPACE；Read/Edit/Bash canary | 待实施 |
| A06 | 模型标签被当身份保证 | 只记录可观察路由/标签；凭证canary不泄露key | 待实施 |
| A07 | 厂商融合算子冒充TileLang | 审查完整候选可达依赖+执行证据，不只正则或diff | 待实现 |
| A08 | 整包禁止torch_npu/acl误伤runtime | 允许注册/分配/事件，禁止委托计算 | 待实现 |
| G01 | 新adapter只加目录未接core | env映射、前缀过滤、机器键、租约、打包测试 | 待实现 |
| G02 | stdout日志与JSON混杂 | 单JSON≤64KiB，日志独立hash，缺失计数不能PASS | 待实现 |
| G03 | 假定新adapter重复编译TileLang | reuse模式明确声明；build只构建FK并校验外部依赖 | 最终方案已修正 |
| G04 | 生产强迫出现REVERT污染优化 | 独立负例验证决策路径，真实run留可信记录 | 最终方案已修正 |
| G05 | 超时孤儿进程、恢复重新计预算 | 进程树终止、日志收集、累计预算与依赖指纹 | 待实现 |
| G06 | /tmp清理造成证据丢失 | 长效资料统一workspace；过程临时文件及时归档 | 本目录已建立 |
| G07 | 仓库HEAD自带12个失败测试（safetensors缺失、driver顺序依赖、harness_guard git权限） | 先记录基线，验收只要求“不新增失败”；safetensors可装 | 已记录基线日志 |
| G08 | 仓库示例BF16内核比厂商慢3.18x/1.17x | 目标需真实内核优化；示例只作可编译性证明 | 已实测 |

本表不替代实施后的真实测试；每条关闭时应补记录对应测试、run ID或工件摘要。
| A11 | 先前判断 `2^-7=0.0078` 容差下限无区分力，据此要求额外设计精度门 | **已实测推翻**：fp32 累加实现 max_rel 0.0041-0.0068（通过），bf16 累加实现 0.021-0.035（不通过），三族皆然，分离 4.2-5.5 倍；mean_abs 差 1e5-1e6 倍。误判根因是把绝对误差（确实饱和）当成相对误差。仍保留保留集与负例，但原因是防过拟合，不是补容差漏洞。 | 已更正，见 evidence `precision_gate_discrimination` |
| T08 | 共享机器上并发租户对计时的影响方向不可先验假定 | 三次实验结论互相矛盾（+70% / +0.2% / -0.1%）。加可审计探针（心跳+`npu-smi` AICore 利用率）复测：邻接负载确认活着（0->90%）时配对比率仅偏移 3.3%。结论：配对基本有效，但**必须给每个窗口留存证词**，无法事后分类的窗口一律作废 | 已验证，见 evidence `shared_card_timing_protocol` |
| T09 | 跨 run 比较绝对设备时间 | 同一内核 vendor 绝对时间在多次 run 间漂移 2.5 倍（0.0206-0.0514 ms），原因是租户来去。任何跨 run 的绝对值对比都是伪结论；只能用同一窗口内形成的比值 | 已验证，同 `shared_card_timing_protocol` |
| T10 | `npu_rms_norm` 接口不对称 | 返回 `(y, rstd)`，rstd 形状 `[B,1]` float32；本任务只要 y。厂商侧多算一个张量，对候选有利。eps 语义与任务公式一致（rstd/manual 比值恒为 1.0），两侧可同标准套容差 | 已核实，见 evidence `vendor_contract_check` |
| A12 | 静态正则查作弊不足 | 别名（`importlib` + `getattr` + 字符串拼接）调用 `npu_rms_norm` 时正则**零命中**，却运行期解析到禁用算子。必须结合可达代码分析 + 运行期/产物佐证（如 `.aibin` 存在且被加载），不能只靠名字黑名单 | 已验证，见 evidence `anti_cheat_screening_limits` |
| E10 | 同一工作树被多人/多会话并发编辑 | 实测 `AgentGates` 原树被另一个 `nvim`（cwd 在 tla-npu）同时改，造成自己的改动与他人改动混在一个未提交树里，且被 revert 后又被对方加回。**对策**：团队工作立即迁到 `git worktree` 独立分支（`.../sources/AgentGates`，分支 `ascend950pr-ag`），并用 `agent_gates.__file__` 断言导入来源 | 已隔离并验证（645 passed） |
| E11 | 用失败数判断 ForgeKernel 改动回归 | **假基线**：`tests/test_loop_driver.py` 在干净 HEAD 上连续三次跑得 15/17/18 个失败，集合每次不同（`driver.py` 与其测试都不在我们的 diff 里）。按 12 个固定失败去比对会追出 10 个不存在的回归 | 已实测，口径改为排除该模块后对比；见 evidence `test_baseline_correction` |
| D11 | Ascend 上 `.double()` 静默降级为 float32 | `torch.zeros(dtype=float64, device=npu)` 是真 float64，但 float32 张量 `.double()` 返回 **float32**，不报错不警告。judge 若在设备上算 float64 距离，实际只有 float32 精度。实测对现有指标影响仅 ~1e-9（远低于 0.0078 容差），不翻转判定，但**不可审计** | 已实测；对策：judge 距离一律拷回 CPU 用真 float64 算，拷贝耗时不计入设备计时 |
| E12 | 多人/多会话共用未提交工作树（复发） | 即使已建 worktree，`machine_env.py` 仍被 nvim 会话在 10:59 写回 `TLA_NPU_*`；且 `AgentGates` 原树的 `locking.py` 与 worktree 版本 sha 相同，说明**原树也留着我们的改动**，两处都在改同一逻辑 | 已隔离；原树残留改动留待最终清理，不在其中继续工作 |
| M1 | Lead 把某形状的 ulp 数字当成全表 | 我把 8192x4096 的三个 bf16 累加 ulp（3.39/3.74/5.53）当成「全 6 组合」区间，**完全漏掉 64x4096 形状**。实测全 6 组合为 **2.053-5.525 ulp**，最窄拒绝余量是 **2.05 倍**（64x4096/near_zero_rows），不是 3.4 倍。另把 `64x4096` 的 vendor 上界说成 0.6，实为 **0.601**；而 P2 在 vendor > 0.667 ulp 时才开始放宽，**只剩 0.066 ulp 余量** | 由 accept-verify 拒绝迁就、坚持上报而纠正；Lead 已从 `precision-matrix.json` 独立重算确认并更正 PLAN/evidence。**教训：局部样本不能代表全表，报告区间必须给出 n 和覆盖范围** |
| M2 | Lead 从进程名倒推进程属主 | 见到 `npu-smi` 上 PID 4066195 名为 `pytest`，即断言是队友 accept-verify 的测试进程。但 `npu-smi` 只给 PID/进程名/显存，**无法确定属主或命令行**；共享机上任何人都有可能 | 由 ag-adapter 拒绝按转述写、坚持只写磁盘事实而纠正。**这与「从 shared(3) 倒推全程有租户」是同一类过度解读**。教训：共享机器的占用记录只能陈述「有什么形状的进程」，不能陈述「是谁的」，除非有 owner 级证据 |
| M3 | Lead 引用已被覆盖的快照数字 | 我向 ag-adapter 转述 after 快照为 `04:38:36Z` / npu0=0 / npu1=3，但磁盘上 `npu-smi-after.txt` 实为 `04:40:45Z` / npu0=2 / npu1=2（更早一次采集被后来覆盖）。**我手上已无该文件** | 由 ag-adapter 按磁盘实测写而纠正。**教训：收据类数字必须每次从文件重读，不能依赖会话记忆；被覆盖的采集要保留唯一文件名**（这也是本方案 README 要求「唯一新输出文件名」的原因） |
| M4 | 文档引用不存在的证据文件 | 我重写方案 README 时引用了 `evidence/fk-nondriver-baseline-2026-11-02.json`，该文件因早前脚本有语法错误**从未真正生成**。检查脚本 `check_docs.py` 立刻以 `AssertionError` 抓住 | 已修正脚本并生成文件。**这条证明「文档链接自动校验」不是形式主义**：它抓到的是「以为写过、其实没写成」这类最难自查的错误 |
| M5 | Lead 再次从进程名倒推属主（M2 复发） | 我看到 after 快照里 PID 4066195 名为 `pytest`、700MB，即断言是 accept-verify 的测试。accept-verify 实测反驳：其测试套件 57 个测试在 `ASCEND_RT_VISIBLE_DEVICES=""` 下全通过，且**`torch` 与 `torch_npu` 均未进 `sys.modules`**（文件里的 `import torch` 在测试用假源码字符串内，不是真导入）。因此它不可能持有卡内存 | 已实测确认其为误判。**教训：同一错误在一次会话内复发，说明「看进程名推断属主」是诱人的默认动作；共享机器上唯一可陈述的是「有什么形状的进程」，属主需要 owner 级证据（用户名/命令行），`npu-smi` 不提供** |
| M6 | Lead 要求不可实现的 witness 采样率（第三次被同一成员纠正事实） | 我要求 accept 在窗口内「每 100-200ms 采一次 AICore」。实测 `npu-smi info -t usages` 每次耗时 **0.79-1.04s**、`npu-smi info` 全表 **0.80-2.39s**（Lead 独立复测确认），因此该密度**物理上不可得**，声称拿到就是伪造 witness。accept-verify 改为 canary kernel：窗口内跑一个已知内核，用其相对本次 run 中位数的膨胀（`CANARY_INFLATION=1.15`）判争用，分辨率等于测量本身，不受轮询慢的限制 | 已实测确认并采纳其方案。**教训：要求观测精度前必须先测观测本身的成本**，否则会把下属逼向伪造数据。另：这是本轮第三次我被同一成员纠正事实（M1 ulp 区间、M2/M5 进程属主、M6 采样可行性），说明**Lead 的技术断言也需要被实测复核** |
| P1 | 认为配对测量能消除争用（「共同负载在比值中抵消」） | **实测推翻**：在 AICore 99% 持续争用下，同一 artifact 的配对比值 8192x4096 从 0.3224 变 0.6615（+105%），64x4096 从 0.9939 变 0.8273（-17%），**方向相反**。因两臂敏感度不对称（大 shape 厂商受害 2.68x / 候选 1.31x；小 shape 候选 18.78x / 厂商 15.71x）。**后果严重**：2.052x 的膨胀足以让真实 0.478 的实现被报成 0.98 通过 MATCH，即把 FAIL 洗成 PASS | 已实测并改判据：contended 窗口一律排除出判定样本；quiet 且非 sparse 才计入；不足或占比<70% 判 RETEST。早先「邻接负载仅偏 3.3%、配对基本免疫」的结论只适用于**轻负载**，不可用于为重争用下的判定辩护 |
| P2 | canary kernel 作为窗口争用检测器 | 实测检出率仅 **3/9（33%）**：全程 AICore=99% 时 canary 膨胀读数在 0.139-1.186 间波动，因其单次仅约 0.06ms，噪声与信号同量级。**它不能用作否证**（不膨胀不等于 quiet） | 已降级为辅助阳性信号；主判据改为进程外 AICore 轨迹（本次 9/9 全对）+ 设备进程表。另注：单次 canary wall 成本实测中位 643ms，是无窗口多次采样的物理原因（10 次需 6.4s，远超被测量的 0.004ms） |
| P3 | 保护性字段只写不读（sparse 检查从未生效） | `samplingRateHz` 写在窗口顶层，而 `classify_window()` 从 `witness` 字典读，导致「quiet 必须可被质疑」的机制**完全失效但报告看起来有**。与此前 `audit_host_bound` 的 `rederived=None` 同类：保护看起来在测、实际没测 | 已修并加消费端测试。**通用规则**：凡报告输出的保护性字段，必须有测试从消费端读它，而非只断言被写出 |
| P4 | 提前 return 路径丢失证据摘要 | `judge_timing()` 在可用窗口不足时提前 return，该路径没有 witness 摘要——而这恰是最需要摘要的场景（读者要能区分「测量太吵」与「根本没几个窗口」） | 已抽出 `_witness_summary()` 供两条路径共用 |
