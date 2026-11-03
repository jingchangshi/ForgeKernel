# TLA 与我们的 L2 策略：一条来自厂商编译器自身的证据链

**作者**：lead。**日期**：2026-11-03。**状态**：源码级结论，未经设备实测。
**范围**：只回答一个问题——R84 的读路径 L2 策略 `NOTALLOC_CLEAN`，在 TLA 里能否表达。
本文不改任何结论；它给 task-8/9/10 的调研提供一条已核对到底的硬件事实。

标签沿用全项目约定：**[verified: 路径:行]** / **[source-only]** / **[inferred]** / **[unknown]**。

---

## 0. 一句话

**我们的 L2 策略与 TLA 向量路径的 `eviction_policy` 不是同一个字段，且后者在 C++ 模板层被硬编码收窄成 2 个值。
这不是"还没接"的缺口，是一条从 MLIR 属性一路到硬件 intrinsic 都很窄的通路。**

而厂商编译器自己的文档，用几乎和我们相同的措辞，说明了**为什么**这件事对向量路径做不到。

---

## 1. 我们实际赢在哪（回顾，已有证据）

R84 的 prefill 提升可分解为两个独立量：

| 改动 | 性质 | 独立收益 |
|---|---|---|
| `N_CORES` 64 → 56 | host 侧整数 | 主项 |
| 读路径 `NORMAL_FV` → `NOTALLOC_CLEAN` | 一个 `l2_cache_ctrl` 参数 | **1.103×** |

依据：`docs/ascend950pr/core-count-and-l2-breakthrough.md`、`evidence/r65-l2-breakthrough.json`。
所以下面这条链如果断了，迁到 TLA 就会**丢掉 1.103×**，回到约 1.02 的几何改动单独水平。

---

## 2. 字段宽度：硬件有 6 个读策略，TLA 向量路径只走 2 个

### 2.1 硬件真实枚举 [verified: /data/pri/cann-9.2.0/cann/asc/include/c_api/utils/enum.h:17-24]

```cpp
enum class asc_load_l2_cache_mode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM  = 1,
    NORMAL_PERSISTENT   = 2,
    NOTALLOC_KEEP       = 4,
    NOTALLOC_CLEAN      = 5,   // <-- 我们用的
    NOTALLOC_DROP       = 6
};
```

注意 3 号位是空的——**这个字段比"逐出优先级"宽**。`NOTALLOC_*` 整组不是一个优先级，
而是"这条流**不要**在 L2 里分配"的独立指令，这正是我们 128 MB 工作集撞 128 MB L2 时需要的语义。

### 2.2 TileLang 把 6 个全都暴露了 [verified: devkit-lessons-memory.md:132-138]

`tilelang/ascend/language/copy_op.py:167-199` 把读策略表原样列出并**后缀匹配**（`:202-227`），
raw int 也接受，所以 `NOTALLOC_CLEAN=5` 直达 C API（`codegen_ascend.cc:1407-1417`
`asc_copy_gm2ub_align(..., static_cast<asc_load_l2_cache_mode>(...))`）。**没有过滤**。

### 2.3 TLA 向量路径把它收窄成 2 个值 [verified]

三层，逐层都窄：

| 层 | 位置 | 该路径上的策略字段 |
|---|---|---|
| TLA Python | `tla/dma.py:134-141` `LoadGM2UBParams` | `eviction_policy: EvictionPolicy`（**无** `l2_cache_ctrl`） |
| TLA Python 枚举 | `tla/dma.py:21-24` | `EvictNormal`/`EvictFirst`/`EvictLast` 三个**名字** |
| MLIR 属性 | `HIVMAttrs.td:1223-1231` | `EvictNormal=1, EvictFirst=2, EvictLast=3` |
| C++ 模板枚举 | `Template/include/RegBase/DMAUtils.h:39` | `enum class EvictionPolicy : uint32_t { EvictFirst = 0, EvictLast = 1 };` |
| 落到 intrinsic | `Template/lib/RegBase/Vector/Copy3D.cpp:144` | `uint8_t l2_cache_ctl = static_cast<uint8_t>(eviction_policy);` |

**关键在最后两行**：MLIR 层是 1/2/3，C++ 模板层是 0/1，而 `Copy3D.cpp:144` 直接
`static_cast<uint8_t>` 进 `l2_cache_ctl`。也就是说向量 GM→UB 拷贝送给硬件的
cache 控制字节，取值范围被人为压到 `{0,1}`。**5（NOTALLOC_CLEAN）在这条路径上不是"没暴露"，
是无处安放。**

对照：Cube 的 GM→L1 路径是宽的 —— `tla/dma.py:149-155` `LoadGM2L1Params` **有** `l2_cache_ctrl: int`，
`:266-275` 直送 `l2_cache_mode` 属性，`Template/lib/RegBase/Cube/include/DMA/ND2NZ.h:61` 传真实值。
所以"TLA 有 l2_cache_ctrl"这句话**本身是真的，但对我们的路径不成立**——这是我们这次调研最容易误判的地方。

---

## 3. 厂商编译器文档自己说了同一件事

BishengIR 的 `ND2NZOp` 文档，原文 [verified: AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td:381-387]：

> `l2_cache_mode`: the raw value of the MTE2 descriptor's cache control field,
> absent meaning the hardware default of 0. **LoadOp spells the same field as an
> EvictFirst/EvictLast enum, which is all the vector copies use, but the field is
> wider than that enum** and a matmul needs the rest of it: **keeping a streamed-once
> operand out of L2 entirely is what stops it evicting the operand that is re-read,
> and no eviction *priority* can express that.**

这段话独立地、用厂商自己的语言，陈述了三件我们花了 R65 和 R84 两轮才测出来的事：

1. 同一个硬件字段，`LoadOp` 只拼成 `EvictFirst/EvictLast`，**"all the vector copies use"**；
2. **字段比那个枚举宽**（`the field is wider than that enum`）；
3. **"把只流过一次的操作数完全排除在 L2 之外，是阻止它逐出被重读操作数的办法，
   而任何逐出*优先级*都表达不了这件事"** —— `no eviction priority can express that`。

第 3 点是 R84 §3 的机制论证，在厂商文档里有一句同义的话。这不只是"我们的结论被支持"，
它说明**这条限制是编译器结构性的，不是 TLA 团队漏接了一个参数**。所以"等 TLA 补上"不是一个
有依据的期望，除非上游改 `LoadOp` 的属性集或让向量拷贝走宽路径。

---

## 4. 这对迁移决策意味着什么

三点，按确定性排序：

1. **确定**：当前 TLA 源码树上，向量 GM→UB 读路径**无法**表达 `NOTALLOC_CLEAN` [verified，§2.3]。
   因此"把 v33 原样迁到 TLA"这一动作，按现有证据会**丢掉那 1.103×**。
2. **证伪了我自己原先提议的实验**（implementer-v33 的发现，我已独立核实三处）：
   我原本提出"实测 `EvictFirst` vs 硬件默认，能否复现 ~1.10×"。**这个实验没有意义**，因为
   `EvictFirst` 就是默认值：
   - `lib/Dialect/HIVM/Transforms/ConvertToHIVMOp.cpp:312-313`：
     `// Default set to evict_first (Ascend950 / Triton cache hint).` 每个 load 无条件写成 EvictFirst。
   - `lib/Conversion/HIVMToStandard/regbase/HIVMToStandard.cpp:68-71`：映射表只有
     `{EvictFirst->0, EvictLast->1}`；`:1228-1236` 对任何其它取值打印 warning 并**强制回落**到 EvictFirst。
   [verified，两处均实读]
   所以 `EvictFirst` 不是 TLA 提供的杠杆，**它就是现状**；"EvictFirst vs 默认"是默认对默认，
   测到的是噪声。真正有意义的对照（默认 vs 字节 5）恰恰是 TLA 表达不了的。
   **因此该实验不是"未做"，是"无从做"**——这一条从 [inferred] 直接变成**结构性不可测**。
3. **写路径反而无害**（修正我原先的推断）：`StoreUB2GMParams` 确实只有 `atomic_kind`
   [verified: `tla/dma.py:143-147`]，但 v33 的写策略是 `NORMAL_FV`，**就是硬件默认 0**，
   所以写路径**没有损失** [verified：v33 发射源码 `asc_store_l2_cache_mode=0`]。
   这一条我原先标 [inferred：可能另有通路]，现在可以关掉：不需要通路。

**因此本文的结论不是"不要迁"，而是**：迁移的收益必须扣除一个已测得的 1.103× 损失，
且这个损失**没有上诉渠道**——不是"实验还没做"，是"该路径上没有可测的替代量"。
任何把"TLA 有 l2_cache_ctrl"当作该损失不存在的论证，都误解了它只在 Cube 路径上成立。

---

## 4bis. 为什么同一个硬件字段，一个 DSL reach 得到、另一个 reach 不到

这一节解释机制，不是重复结论 [verified: implementer-v33 的实测，我复核了其引用的落点]。

**TileLang 根本不过 HIVM。** `codegen_ascend.cc:1417-1433` 的 `EmitGmToUbufCopy_` 直接发射
CANN C-API：`asc_copy_gm2ub_align(..., static_cast<asc_load_l2_cache_mode>(arg8))` ——
强制转换到**宽硬件枚举**，`5` 原样通过。
发射出的 v33 设备源码里三个 DMA 站点分别是：W 读 `=0`、**X 读 `=5`（我们的 1.103×）**、Y 写 `=0`。
所以 1.103× **确实是交付产物里的字节 5**，不是文档推断。

**TLA 走 HIVM，而 HIVM 那一层的属性类型就是窄的。** 收窄点在
`Template/include/RegBase/DMAUtils.h:39` 的 C++ 枚举 `{EvictFirst=0, EvictLast=1}`，
以及 `HIVMToStandard.cpp:1233-1235` 那句 "Only support EvictFirst & EvictLast Policy for now"。

**这个区分对决策很重要**：它不是"TLA 少接了一个参数"，而是**两条 codegen 路径到达同一硬件字段的
方式不同**。前者可以等上游补；后者需要上游改 `LoadOp` 的属性类型或让向量拷贝绕开 HIVM 的窄枚举。
**这就是"什么会让本条翻案"的准确答案**，也是本文对 §5 的补充。

---

## 5. 本文没有证明的

- **⚠️ 更正（2026-11-03，task-12）**：本节初版说"本机没有任何 wheel 含 `tla`"并据此称
  "TLA 在本机不可安装"。**那个结论是错的。** 真的部分是：默认的 `s00653124_tilelang` 环境没有 `tla`，
  磁盘上那 6 个 wheel 文件的 `language/extra/tla/` 均为 0 文件 [verified，我复核]。
  但那答的是"wheel 文件里有没有"，不是"本机有没有能用的"。**本机另有
  `/home/s00653124/.conda/envs/s00653124_tla`**，装 `triton_ascend-3.6.0.dev20261101+git2a00c605`，
  含 21 个 tla .py，**可导入、可编译、可在 NPU 上执行**，tla-npu 门实测 **PASS**
  （run `9e8c59f1c356badc32a0e0d396af4659`）。见 [`how-to-run.md`](how-to-run.md)。
  **所以"没有人跑过 TLA"这句要收窄成："没有人跑过我们的 rmsnorm 内核的 TLA 版。"**
- **本文的 L2 结论未因此改变。** 我按 lead 的要求，把 §2.3 的字段核对**从已安装 wheel 重做了一遍**
  （`$CONDA_ENVS/s00653124_tla/.../extra/tla/dma.py`，非源码树）：`LoadGM2UBParams` 字段仍是
  `[pad_mode, pad_value, left_padding_num, right_padding_num, eviction_policy]`，**无 `l2_cache_ctrl`**；
  `LoadGM2L1Params.l2_cache_ctrl` 仍在（Cube GM→L1）；`EvictionPolicy` 仍是
  `[EvictNormal, EvictFirst, EvictLast]`，**无任何 NOTALLOC 值**。**字节 5 依旧无处安放。**
- 没有验证 `EvictFirst` 的实际带宽效果（§4 第 2 点的实验未做）。
- 没有核 `StoreUB2GMParams` 之外是否还有别的写策略通路（§4 第 3 点）。
- 没有评估 `LoadGM2L1Params.l2_cache_ctrl` 能否被"借用"到向量路径——从 §2.3 的五层看不像，
  但那是设计问题，超出本文范围。

---

## 6. 复现命令

```bash
# 硬件枚举
sed -n '17,32p' /data/pri/cann-9.2.0/cann/asc/include/c_api/utils/enum.h
# TLA 两层枚举 + 收窄点
sed -n '21,24p;134,155p' /home/s00653124/workspace/triton-ascend/third_party/ascend/language/tla/dma.py
grep -n "EvictFirst\|EvictLast\|EvictNormal" /home/s00653124/workspace/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td
sed -n '39p' /home/s00653124/workspace/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/lib/Template/include/RegBase/DMAUtils.h
sed -n '144p' /home/s00653124/workspace/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/lib/Template/lib/RegBase/Vector/Copy3D.cpp
# 厂商文档原话
sed -n '381,387p' /home/s00653124/workspace/triton-ascend/third_party/ascend/AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td
# 磁盘 wheel 文件里有没有 tla（初版把它误当成"本机有没有可用的"）
for w in $(find /home/s00653124 /data/pri -name "triton_ascend*.whl"); do echo -n "$w "; unzip -l "$w" | grep -c "extra/tla/"; done
# 本机真正可用的 TLA 在专用 conda 环境里（这才是干净的检查）
ls /home/s00653124/.conda/envs/s00653124_tla/lib/python3.11/site-packages/triton/language/extra/tla/*.py | wc -l
# 从【已安装 wheel】核对 L2 字段（不是源码树）
E=/home/s00653124/.conda/envs/s00653124_tla/lib/python3.11/site-packages/triton/language/extra/tla
sed -n '134,146p' $E/dma.py; grep -n "l2_cache_ctrl" $E/dma.py; sed -n '21,24p' $E/dma.py
```
