# ForgeKernel × TileLang-Ascend × AgentGates 最终方案

更新：2026-11-02。状态：方案已整理，待批准实施；尚未创建正式adapter或应用FK补丁。

## 0. 已确认的运行前提（2026-11-02 用户决定）

- 共享机器：所有卡都可使用，不必固定卡0；他人占用该卡不构成不可用。
- 不做整段独占：只在真正跑 kernel 的窗口内取协作租约，模型思考与编译期不占卡。
- 优化用 agent 模型：magpie 网关 auto/claude-sonnet-5-5（已 canary 通过）。
- 其余沿用建议：允许改 AgentGates core 映射、补丁双轨、先 60–90 分钟受控 run、online + 依赖 frozen、逐 shape 判定、先纵切、三层分工。

### 0.1 共享机器带来的三处设计变更（取代此前独占 / 安静卡 假设）

1. 设备租约改为窗口级、协作式：AgentGates 现有 npu:<id> 是 flock 排他锁且覆盖整个 recipe；改为只在声明需要设备的 recipe 上加锁、离开即释放。其他租户不受我们控制，occupancy 只记录、不作门槛。
2. FK 的 gpu_lock 仅在 submit job 期间持有（已是窗口级）；保持，且不与外层嵌套同一把锁，避免自锁。
3. 性能验收改为同窗口配对：同一测量窗口内 A/B 与 B/A 交替，使外部负载同时作用于双方；用配对差值与 bootstrap 区间判定，噪声大则重测而非判失败；收据记录测量时刻的卡占用。

相应地，此前卡必须安静、被占即 BLOCKED、独占才能计时的表述全部作废：设备被他人使用是正常状态，不是环境阻塞。

## 1. 目标、范围与当前事实

在当前Ascend950PR服务器，以ForgeKernel驱动TileLang-Ascend优化RMSNorm；通过AgentGates完成准备、构建、回归、真卡运行、优化与独立验收。最终BF16候选必须在每个规定形状上持平或超越厂商torch_npu.npu_rms_norm；闭环跑通不等于性能达标。

新adapter拟命名forgekernel-ascend-npu，落在${WORKSPACE}/AgentGates/agent_gates/adapters/forgekernel-ascend-npu/。结构参考现有tilelang-hivm-npu，但首轮采用宿主机local delivery，不直接复制Docker/root前提。

角色：AgentGates管执行、资源租约、证据和验收；ForgeKernel保留teacher→judge→calibrate→student→KEEP/REVERT；TileLang/CANN提供编译运行后端；独立验收不采信agent自报成绩。

旧/tmp方案、脚本、job日志已被清理。本文按会话重建并纳入勘误，不是旧文件原样恢复。历史实验只作线索；原始证据丢失的性能数值不得直接用于新验收。本次新增只读环境检查及其报告保存在本目录。

当前修订（2026-11-02重新读取，三个工作树均干净）：

| 仓库 | commit |
|---|---|
| ForgeKernel | 5030503d3ff71851aa34c14b9caf13ebfa9d9a78 |
| AgentGates | a72c5c9aa7712835e1c210289291e00d0bd0deb7 |
| TileLang | 8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b |

## 2. TileLang构建复用：明确采用现有构建

直接复用${WORKSPACE}/tilelang源码与build产物，不重复构建另一份TileLang，也不由adapter自动pip升级或修改这个共享checkout。

- Python：${CONDA_ENV}/bin/python3。
- CANN初始化：/data/pri/cann-9.2.0/cann/set_env.sh。实际ASCEND_HOME_PATH与compiler路径由doctor记录，不能只看目录名称推断版本。
- Python包：${WORKSPACE}/tilelang/tilelang；Ascend语言包来自其ascend/language目录。
- 原生库：${WORKSPACE}/tilelang/build/lib/libtilelang.so及相关TVM库。
- 当前torch=2.10.0+cpu；torch_npu=2.10.0.post7.dev20261011；TileLang=0.1.15+cuda.git8e2d3c82。版本字符串含cuda不意味着Ascend后端缺失，需看实际模块、编译和设备执行。
- 2026-11-02只读检查：四个import成功、原生库存在并已计算hash。并未初始化NPU或运行kernel，也未证明graph/BF16编译有效。

复用是一个明确的依赖模式：build gate构建和安装适配后的ForgeKernel，TileLang作为外部已有构建做资格验证。记录源码commit、dirty状态、模块来源、原生库hash、CANN/compiler、Python/torch/torch_npu；任一变化使相关证据过期，需重新doctor/smoke/性能基线。工作树干净不能证明库对应源码，因此真卡cold-cache smoke必做。

若现有构建失败，停止并给出具体诊断，再进入单独的TileLang重建流程；不静默换wheel、切版本或覆盖原build。新的FK Python依赖使用私有环境或明确受控的解释器配置，运行前断言仍加载指定TileLang/torch，不改共享Conda。

复用编译器构建不等于复用旧kernel缓存。每run设置独立TILELANG_CACHE_DIR，首次验收从空cache重编译；FK迭代可复用同依赖指纹缓存，最终accept再使用新cache。禁止删除他人全局cache。

## 3. 重要勘误：后续不得沿用的结论

| 历史说法 | 修正 |
|---|---|
| 8192大shape把geomean host/device降到0.19，所以decode设备时间有效 | 错误。timing.workloads_performance以及shape聚合对host_bound_ratio取MAX，小shape的host-bound不会被平均消除。 |
| 参考归档通过说明计时有效 | _task_ref_alone曾漏导出最大ratio及逐workload审计，导致baseline重算ratio of geomeans。须修复JSON→parse→archive→gate信息丢失；[0.04,0.88]必须按无效device基线处理。 |
| FP32原型PASS代表真实BF16已通 | 错误。此前主要是256/64、FP32；BF16曾因VF外T.copy类型转换报错。8192/64 BF16须正式验证。 |
| 跑快于PyTorch 1.2倍就是目标达成 | 错误。最终竞争对手是厂商融合算子，不能让arch_ref或显式target优先级恢复成宽松PyTorch目标。 |
| eval只是torch.cuda机械替换 | 还包括float64验证、RNG、graph、precision、能力降级与JSON审计。 |
| sandbox off意味着网络工具全开或全关 | 两者独立；权限提示、deny和hook需与配置一致。off失去OS文件/网络隔离。 |
| PATH和LD_LIBRARY_PATH永远存在且无法修 | PATH通常存在，LD_LIBRARY_PATH不一定；setdefault无法覆盖已有值。adapter显式初始化环境即可解决。 |
| resume不重写loop.toml，删了重跑 | 普通_drive会重新生成；不得靠删除状态修配置，须测试新建/恢复/judge全链传播。 |
| CUDA公共eval源码必须字节不变 | gpu_entry的CUDA渲染可要求一致；共享eval会变，要求行为和协议回归。 |
| 生产运行必须自然出现KEEP和REVERT | 用独立测试验证两条决策路径；真实优化保留可信决策，不人为制造生产失败。 |

## 4. Adapter包和持久工作区

计划包结构：

    forgekernel-ascend-npu/
      manifest.json
      README.md
      env.example
      scripts/gate.py
      scripts/prepare.py
      scripts/launch.py
      scripts/accept_rmsnorm.py
      patches/series.json
      patches/0001-*.patch ...
      tasks/rmsnorm-ascend/
      tests/

所有recipe脚本、patch、任务、计时器及评测清单进入manifest摘要。基础FK checkout是输入；独立工作树精确应用固定patch，记录base commit、patch hash和最终树hash。不fuzzy apply、不自动覆盖有冲突工作区。

本方案目录本方案目录仅用于方案、说明和只读预检，不混入正式源码。实施时另建显式runtime work root，包含sources、venv、runs/<id>、cache/<fingerprint>/<id>、artifacts/<id>。长效方案、脚本、日志和结果全部落在workspace，不把/tmp作为唯一存储。工具内部临时文件可用/tmp，但离开步骤前必须归档必要证据。

## 5. AgentGates核心接入

现有schemaVersion=1提供固定argv、configCheck、blockedExitCodes、gate依赖、observationChecks。新增目录之外还要：

1. engine的ENV_KEYS、OWN_PREFIX、DEVICE_ADAPTERS、EXCLUSIVE_VARIABLE及前缀过滤增加FK_ASCEND_。
2. machine_env增加同一严格键集合、配置发现、work/build路径；core和adapter各自解析必须有一致性测试。
3. registry和打包收录新adapter、patch与task；复用evidence、executor、verify/report。
4. stdout只输出一个≤64KiB JSON对象；长日志单独归档并hash。零测试、全skip、缺JSON、非零、超时不能PASS。
5. observation由受固定摘要保护的脚本从真实样本计算，保留计数与原始证据，不能由agent输出true自证。
6. exit75仅用于明确环境阻塞；数值/精度/性能失败为FAIL，不能变成SKIP。
7. manifest仍用支持的gateType=deterministic-test：指本次产物按固定判据验证，不宣称LLM搜索可确定复现。

sourceWrite=false意味着输入checkout不改写；外部work root写入行为必须明确声明，并验证实际engine/pack权限兼容。工作树创建和patch应用只由prepare执行。

## 6. 环境、模型和工具策略

配置采用严格NAME=value数据文件，不执行字符串、不展开shell替换。独立FK_ASCEND_前缀涵盖：CANN_SETUP、PYTHON、TILELANG_SOURCE/BUILD、WORK_ROOT/BUILD_SUBDIR、CLAUDE_BIN、设备分配、AGENT_SANDBOX、AGENT_NETWORK、依赖策略、provider端点/模型/密钥变量名、job及总预算。

adapter启动时先source经过路径校验的CANN脚本，再选择绝对Python并构造PATH/库路径，将一致环境传给judge、teacher、student、job runner、accept。不能只初始化GPU job而漏掉agent。避免工作目录同名inspect.py遮蔽标准库。

机器级FK_AGENT_SANDBOX=off保持已讨论决定；FK默认strict，auto/off行为明确。完整传播local.env→Machine→生成配置→policy，最早write_judge也需生效。loop.toml/submit.toml是生成物，不作为配置源手改。

工具建议（待实施配置固定）：

- 正常Read/cat/ls/grep/find/Edit/git/Python/compiler可用；提示词给真实WORKSPACE绝对路径，避免网关环境块的假cwd。
- 推荐本任务network=online，可用curl查询文档；ssh可做允许的诊断。正式GPU执行仍走FK受记录的submit/probe渠道。
- pip工具存在，但正式优化测量期间依赖frozen；新增依赖需在私有环境锁版本/hash、重新build出证据。不改共享Conda/CANN/TileLang。
- 保留offline作为显式实验选项，不把它当用户已要求；单改student_online不能放开ssh，因为no_cluster另有逻辑，需分别配置。
- 不为阻拦而泛化open/chr/base64正则；误伤大且不能产生强隔离。路径deny的Claude双斜杠语法须保留并做版本化验证；Edit规则覆盖文件编辑工具。
- sandbox off下deny/hook是有限策略检查，不是OS安全边界。保护judge/工具链/凭证的完整性不能只靠hook。

Claude Code是客户端，计划走magpie兼容Anthropic messages API。历史验证不等于现在凭证可用，启动做小canary；密钥通过指定环境变量注入，不放argv、配置报告或日志。返回model标签不能独立证明底层模型身份。

## 7. ForgeKernel补丁范围

| 包 | 实现范围 |
|---|---|
| P1 policy | sandbox strict/auto/off；独立network/cluster策略；notice/hook一致；judge及resume配置传播；真实工具canary |
| P2 device | device kind、ASCEND_RT_VISIBLE_DEVICES、backend/物理ID锁、local runner；去除CUDA限定文案 |
| P3 registry | task/loop language显式映射tilelang-ascend；ascend950pr架构；未知语言报错；kernel/provenance/KB/prompt路径一致 |
| P4 entry/reference | NPU gpu_entry与sealed reference树共用环境/设备/超时；关闭NVIDIA clock lock及CUPTI安装；完成marker及错误码契约 |
| P5 eval | fk_dev设备层；tensor/stream/event/sync/RNG；NPU TF32明确N/A；float64验证转CPU；CUDA行为回归 |
| P6 profiling | driver真正消费ncu_available；无NCU不提交不重试；timeline/stall/SASS/census能力化及skill过滤；NPU profiler另行开发 |
| P7 task/KB | BF16真实任务、TileLang-Ascend KB index、DSL源码位置、SIMT/SIMD/VF规则；测量支持的硬件事实 |
| P8 timing/target | NPU graph计时资格；最差workload审计全链；数学judge/vendor baseline/搜索target分离；逐shape厂商约束及漂移重测 |
| P9 eligibility | 完整候选及可达helper的计算委托检查，允许torch_npu注册/事件等基础设施，拒绝融合算子代算；regex配合实际执行证据 |

不整包禁止torch_npu/acl import，避免挡住后端注册；也不能只扫本轮diff或只看产物存在。补丁由adapter固定摘要和顺序，修改后重跑相关回归。

## 8. RMSNorm正确性和基线契约

x[M,4096]、w[4096]、y[M,4096]均BF16，M=8192/64，epsilon=1e-6；FP32统计与缩放，最终一次BF16舍入。数学公式与现有任务一致；合法输入范围与测试抽样需分清，任何语义限制都必须在task版本中明确。

- semantic judge：独立高精度定义；校准后冻结，不作为最终速度对手。
- vendor：torch_npu.npu_rms_norm(x,w,1e-6)[0]。冻结调用方式、版本、输出分配、流与计时basis；该API还输出rstd，报告接口差异，不能人为加慢wrapper。
- candidate：实际TileLang-Ascend生成内核；验收从归档源码重新编译。

多seed/group和保留集覆盖典型值、近零、离群、正负weights及合法极值。阈值在看候选成绩前固定；厂商也须满足同一语义，不能用不等价结果比较速度。

BF16校准normalized max error floor可到2^-7=0.0078125（2026-11-02 ulp实测：**2^-7正是bf16一个ulp的相对上界**——实测相邻bf16间隙最大0.0078125、舍入偏移最大0.00389恰为半ulp）。因此**正确舍入的bf16实现可以恰好落在阈值上，余量为0是bf16的数学性质，不是门槛漏洞**。三条硬约束：(1) 比较必须用 `<=` 含等号，写成 `<` 会把厂商与正确实现都判失败；(2) 判别力用ulp倍数论证——正确实现1.000 ulp，故意bf16累加**全6组合实测2.053-5.525 ulp**（最窄为64x4096/near_zero_rows的2.053），**最窄拒绝余量2.05倍**，全部被拒，判据有效；(3) 不用「余量百分比」论证判别力。仍须加入故意BF16累加等代表性错误实现负例并锁死「等号必须通过」的测试。precision不可执行或缺记录不能PASS。

leaderboard.task_error和precision.rel_l2存在设备端float64路径，NPU不支持时移CPU FP64验证；不能只改truth fallback。DMA搬运不可在VF外做BF16→FP32 cast，要先同类型copy再在SimdVF/SimtVF转换。FP32原型不能替代这个测试。

## 9. 计时资格和厂商性能验收

### 9.1 先过计时门，再优化

**2026-11-02实测进展**：TileLang自带msprof计时后端在本机可用，返回的是torch_npu.profiler解析出的逐kernel设备时长（dur_ns/n_repeat，已排除cache-flush kernel），因此是天然的host-free内核时间basis，不必依赖NPUGraph。已用它同时测candidate与vendor取得下表数字。仍需正式资格验证：重放次数、cache flush、pipeline稳定、双方同一basis、数值与单位。

若msprof资格不通过，再验证torch.npu.NPUGraph/graph可否捕获vendor及candidate，并验证重放实际执行、输出/输入更新、时间单位、流同步、batch/replay敏感性。双方须用同一basis。

- 支持且验证通过：定义真实NPU graph host-free basis，端到端接入FK字段/HOST_FREE语义；不得冒称CUDA graph/CUPTI。
- 不支持：不将plain_batch标为host-free、不用--allow-host-bound-ref跳过。可显式用同口径e2e做部署性能比较，但结论必须写“端到端”；纯设备性能资格继续未通过。
- host-bound每workload判，聚合取最差；_task_ref_alone、parse、archive须保留完整审计。增加[0.04,0.88]负例防止大shape掩盖decode。

### 9.2 起跑差距（2026-11-02实测，msprof设备时长，空cache）

| shape | 仓库example内核 | vendor npu_rms_norm | 相对vendor | 数值误差(max_rel) |
|---|---|---|---|---|
| 8192x4096 bf16 | 0.1634 ms | 0.0514 ms | **0.315x（慢3.18倍）** | 0.0078125（vendor 0.007752） |
| 64x4096 bf16 | 0.004683 ms | 0.004019 ms | **0.858x（慢1.17倍）** | 0.00543（与vendor相同） |

结论：BF16契约在本工具链上可编译、数值正确；但仓库示例内核明显慢于厂商，**持平需要真实内核工程（UB/流水/每核行数/reduce），不是接线工作量**。

数值门槛的一点更正：先前写「0.0078125的MATCH下限没有区分力」是**错的**，它把「零余量」误当成「无区分力」。ulp实测表明 2^-7 是bf16一个ulp上界，厂商在本shape恰为**1.000 ulp**、正确fp32实现也是1.000 ulp，故零余量属预期；而故意bf16累加**全6组合为2.053-5.525 ulp**，全部被拒绝。**最窄的一处是64x4096/near_zero_rows（2.053 ulp vs 上限1.000 ulp = 2.05倍）**，不是3.4倍——早先只看了8192一个形状，把该形状的数字误当成全表，已更正。门槛有区分力，但**必须用 `<=`、且判别力以ulp倍数而非余量百分比表述**。保留集与负例仍要做，理由是防过拟合，不是补容差漏洞。

### 9.3 正式目标

smoke可以先对PyTorch验证链路，但正式target来自同计时窗口vendor_ms/target_speedup。初始speedup=1.0，不能沿用对PyTorch的全局ratio=1.2。检查显式target和arch baseline优先级，避免悄悄选错来源。

逐shape验证vendor约束；FK若暂只按aggregate搜索，aggregate仅作引导，不作最终通过依据。应在搜索目标中加入最差shape约束/验收反馈，避免FK提前结束但accept永远不通过。

### 9.4 初版待冻结的验收规则

同一测量窗口内配对：相同输入/输出分配、warmup、stream、capture；A/B与B/A交替，至少15组配对窗口、至少3轮独立进程。报告逐shape中位数、分布、配对bootstrap 95%区间和原始样本。

**2026-11-02实测更正——配对不能消除重争用（推翻本方案早先的断言）**：先前写「以配对差值消除共同负载」，实测在 AICore 99% 的持续争用下**不成立**。同一 artifact 两次测量：8192x4096 配对比值 0.3224→0.6615（**+105%**，候选被美化），64x4096 0.9939→0.8273（**−17%**，候选被丑化）。原因是两臂对争用的敏感度不对称（大 shape 上厂商受害 2.68×、候选仅 1.31×；小 shape 上候选受害 18.78×、厂商 15.71×），比值无法抵消。**偏差方向随 shape 反转，因此争用不是保守误差，它能把 FAIL 洗成 PASS**：2.052× 的比值膨胀足以使真实比值 0.478 的实现被报成 0.98 从而通过 MATCH。

因此验收判据改为：**contended 窗口一律排除出判定样本**（记录并展示，但不进点估计与 bootstrap）；可用样本为 quiet 且非 sparse 的窗口，数量不足 9 个或占比 <70% 一律判 RETEST。窗口 contended 的主判据是**进程外的 AICore 轨迹 + 设备进程表**（本次 9/9 窗口 AICore=99% 全部正确标出）；canary kernel 因单次仅约 0.06ms、噪声与信号同量级，实测检出率仅 **3/9（33%）**，故降级为**辅助阳性信号，绝不可用作否证**。

该规则已在 `accept_rmsnorm.py:judge_timing` 实现并有回归测试。三项配套实测（见 `evidence/exclusion-rule-replay-2026-11-02.json`）：
- **重放旧数据**：干净 run 的 10/10 窗口皆 quiet，新规则剔除为空，原判定 64x4096 RETEST / 8192x4096 FAIL **不变**，正式结论稳定；争用 run 的 8192x4096 由 FAIL **改为 RETEST**（旧代码在 0 个 quiet 参照下仍下 FAIL，正是被修的缺陷）。
- **回归探针**：临时改回 `sample = list(usable)`，5 个测试立即失败；恢复后 74 passed。证明防护是承重的，不是装饰。
- **分支可达性**：`usable < 9` 的检查原先排在 quiet 规则之后、永不可达（quiet 样本是 usable 子集），已前移并加测试。这是本项目第三次发现「保护看起来在测、实际没测」，通用规则：凡报告输出的保护性字段，必须有测试从消费端读它。

- speedup=vendor_latency/candidate_latency。
- MATCH：各shape点估计≥1.0，且95%下界≥0.98。2%为预声明工程等价容忍，不是统计证明更快。
- EXCEED：所有规定shape的95%下界>1.0；只部分超越就如实报告。
- 冷编译及warm稳态分别报告，编译成本不混入稳态；保留足够计时精度。
- 接受测试不调候选/阈值；波动、争用或基线漂移导致证据不可靠时不作性能PASS。

原子设备job尽量≤120s，可拆多job；总优化可按小时，不能把120s当所有流程上限。warmup按稳定性资格测定，0.3s不是已验证常数。

## 10. Gate DAG、锁和运行预算

| Gate | 验收内容 |
|---|---|
| doctor | 严格配置、复用TileLang指纹、CANN/Python来源；显式设备及最小模型canary |
| build | 独立FK树精确打patch，构建wheel/安装，CLI/import正确；TileLang是复用依赖资格检查，不伪装重新构建 |
| regression | adapter/core协议和FK相关测试；collected>0且非全skip，覆盖负例、policy、设备、parse、target与两条决策路径 |
| npu-smoke | BF16冷cache编译、calibrate、真实submit、设备同步、marker与parse；不是性能PASS |
| timing/baseline | 合格计时、正确vendor、冻结逐shape基线 |
| optimize | 真实teacher/student、提交/job/KEEP决策、best kernel、预算和恢复日志 |
| accept | 新进程/新cache重编译best，独立精度、真实性和逐shape厂商性能判定 |

依赖顺序build→regression→smoke→timing/baseline→optimize→accept。可对绑定的已有run单独accept；--all只有选择新adapter才运行，不串入旧TileLang sanity。

外层timeout>内层预算+TERM/KILL清理+报告余量；teacher/student与job可有各自预算但总截止明确。resume不重置已耗预算；源码/patch/task/基线/依赖变化必须重新资格检查，不继续引用过期成绩。

AgentGates npu:<id>和FK锁只协调合作任务，不能排除其他租户。统一物理设备标识及锁顺序，避免外层持锁内层再取同锁自锁。优先按job/测量窗口取租约，模型思考不长占卡；若现有engine只能整recipe锁定，应明确记录整段占用并设计单一锁所有者，不宣称已支持细粒度锁。

已分配设备必须显式指定；logical npu:0不等于物理卡0。健康和占用每run重查，历史卡1 Alarm不是当前事实；不自动换卡、reset设备、杀他人进程。不能用重复UUID或logical映射API伪造独立物理身份验证。

## 11. 证据、实现真实性与失败语义

归档manifest/script/patch/task/judge/tolerance/basis摘要、base/final source指纹、环境和原生库hash、设备选择/健康/占用、脱敏模型配置、预算、FK status/job日志、最终候选及helper、编译产物、逐shape错误/计时原始样本和判定。

harness_guard/overlay保护评测文件；独立accept复核摘要，不信任agent宣告PASS。运行trace/调用审查补充静态快捷路径检测；不以hash存在证明执行。off模式不宣称能防同UID恶意进程。

超时/中断清理完整进程树，收集所有相关后台工作；API不可用、编译失败、数值FAIL、性能FAIL、环境阻塞分别标识。性能未达标保留最佳正确内核，报告“闭环已通，厂商门槛未达”，继续优化，不粉饰成成功。

## 12. 实施顺序与交付

1. adapter/core接入、严格机器配置、独立FK工作区和复用TileLang doctor；host回归先行。
2. FK基础适配、真实agent工具回合、BF16 cold-cache submit smoke，保留CUDA行为回归。
3. 修复审计丢失并通过计时资格，建立厂商正确性/性能基线，冻结评分策略。
4. 接通厂商target和真实优化闭环；SIMT/SIMD、每核行数、UB/reduce/流水线由测量驱动。
5. 独立accept、AgentGates verify/report，交付源码、补丁、完整run及可复验材料。

不能预先保证移植本身就超过厂商。成功标志始终是：真实TileLang BF16实现、精度正确、计时合格、每shape满足冻结厂商门槛。
