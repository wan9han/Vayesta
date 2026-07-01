# 体系展开与结果收集过程的优化与性能分析

> 本文档详细记录了 E 级聚乙烯（PE）弱扩展方案中两个求解器外流程——**体系展开（generation）**与**结果收集（combine）**——的性能瓶颈定位、优化措施、实测数据（本机 + 内网双环境）及后续改进方向。

---

## 目录

1. [流程概述](#1-流程概述)
2. [展开过程：原始瓶颈定位](#2-展开过程原始瓶颈定位)
3. [展开过程：优化措施（6 项）](#3-展开过程优化措施6-项)
4. [展开过程：实测性能数据](#4-展开过程实测性能数据)
5. [收集过程：算法与文件系统开销](#5-收集过程算法与文件系统开销)
6. [收集过程：实测性能数据](#6-收集过程实测性能数据)
7. [内外网对比与环境差异分析](#7-内外网对比与环境差异分析)
8. [已识别的后续优化方向](#8-已识别的后续优化方向)
9. [附录：复现方法](#9-附录复现方法)

---

## 1. 流程概述

弱扩展方案二（分块 + MFCC/MBE）将 PE 长链切成 N 个 block，每节点独立求解一个 block，辅以 (N−1) 个 H₂ cap 和 (N−1) 个 dimer（MBE(2) 二体修正）。求解器之外有两个串行流程：

| 流程 | 何时执行 | 做什么 | 复杂度 |
|---|---|---|---|
| **展开（generation）** | 求解前 | 生成 PE 链 → 切 block/cap/dimer → 写 SIESTA FDF 输入 | O(N log N) ~ O(N²)（优化前）|
| **收集（combine）** | 求解后 | 读所有 siesta.out → 提取能量 → MFCC(1)+MBE(2) 求和 | O(N) |

两者都是**单进程串行**，在 E 级规模下可能成为长尾。本文档记录了对其的系统性优化。

---

## 2. 展开过程：原始瓶颈定位

原始 `weak_scale_pe.py` 的展开流程分 4 阶段：

```
_generate_chain  →  _build_fragments  →  _write_inputs  →  _write_launch_artifacts
(生成链+解析)       (cKDTree+建片段)     (写FDF+赝势)       (写run_local.sh等)
```

逐一分析，发现**三个 O(N²) 瓶颈**和一个**架构性浪费**：

### 瓶颈 1：H→C 映射的暴力搜索（O(N²)）

```python
# _build_fragments 中，每个 H 扫描全部碳取 argmin
for h in h_idx:                              # 遍历每个 H（~2N/3 个）
    d = np.linalg.norm(cpos - coords[h], axis=1)   # 与【所有】碳求距离（~N/3 个）
    h_to_c[h] = cs[int(np.argmin(d))]
```

复杂度 O(n_H × n_C) = O(N²)。在 22680 节点（1.81 亿原子）下，外推约 **1209 天**，完全不可行。

### 瓶颈 2：block 构造的 H 归属扫描（O(N_nodes × N_H)）

```python
# 每个 block 扫描全部 H 找属于自己的
for b in range(num_nodes):
    atoms = list(carbons) + [h for h in h_idx if h_to_c[h] in cset]
    #                                        ^^^^^^^^^^^^^^^^^^^^^^^^
    # 每个 block 扫全部 ~1.2 亿 H，共 22680 个 block → 2.7 万亿次查找
```

这是一个**隐藏的第二个 O(N²)** 瓶颈——22680 节点实测中卡住 9 分钟+ 才发现。复杂度 O(N_nodes × N_H)。

### 瓶颈 3：gen.py 的 Python 逐原子循环

```python
# gen.py 生成碳骨架 + 侧链 H，每原子一次 dict append
for i in range(n_carbons):
    atoms.append({'species': 1, 'x': ..., 'y': ..., 'z': ...})
```

6048 万碳 + 1.21 亿 H = 1.81 亿次 Python 循环 + dict 创建，峰值内存 ~70-90 GB，耗时 ~10 分钟。

### 架构浪费：FDF 序列化-解析往返

```
gen.py（子进程）                    weak_scale_pe（父进程）
  构造 1.81 亿 dict                   解析 12.7 GB 文本
  → 格式化 12.7 GB FDF 文本   →   regex + split 181M 行
  → stdout 管道                →   重建 Molecule 对象
```

两个 Python 进程之间走 12.7 GB 文本往返，纯属浪费（~5-8 分钟）。

### 其他开销

- **赝势逐目录复制**：C.psf + H.psf（~260 KB）拷进每一个 block/dimer/cap/full 目录，68000 目录 = ~14.5 GB 冗余拷贝。
- **full 基线**：未分块整链的参考输入，E 级下 SIESTA 根本跑不动，其 FDF 单独 ~13 GB。

---

## 3. 展开过程：优化措施（6 项）

### 优化 1：cKDTree 替代暴力搜索（O(N²) → O(N log N)）

```python
# 优化后
from scipy.spatial import cKDTree
tree = cKDTree(cpos)                    # 建树 O(n_C log n_C)
_, nn = tree.query(coords[h_idx], k=1)  # 全部 H 的最近碳 O(n_H log n_C)
```

**正确性保证**：cKDTree 的 k=1 查询 = 欧氏距离最近碳，与原 `argmin(norm)` 语义完全一致；PE 的 C–H 键长 1.09 Å 远短于 C–C 1.54 Å，无并列。实测 n_c = 8 / 50 / 200 / 1000 / 4266 五档，brute == cKDTree **逐位相同**。

### 优化 2：h_by_carbon 反向索引消除第二 O(N²)（O(N_nodes × N_H) → O(N)）

```python
# 优化后：一次性反向索引，每个 block 直接查
h_by_carbon = defaultdict(list)
for h in h_idx:
    h_by_carbon[h_to_c[h]].append(h)
# block 构造改为：
atoms = list(carbons) + [h for c in carbons for h in h_by_carbon.get(c, ())]
```

验证：block/dimer 的 H 原子集与旧法**完全相同**（n_c = 200/1000/4266 断言通过）。

### 优化 3：向量化 PE 链生成（Python 循环 → numpy 数组）

新增 `energy_first/pe_chain.py`，碳骨架和侧链 H 用 numpy 数组运算一次性生成：

```python
# 碳骨架（3 行替代 for 循环）
i  = np.arange(n_carbons)
cx = i * dx
cy = np.where(i % 2 == 0, -dy/2.0, dy/2.0)

# 侧链 H（2 行）
sign = np.where(cy > 0, 1.0, -1.0)
hy   = cy + sign * hy_base
```

**正确性验证**：`generate_pe_chain(n)` 与 `gen.generate_polyethylene(n, 1.54, 1.09)` 在 n = 2 / 8 / 50 / 200 / 1000 / 4266 全部 **8 位小数完全相同**。（注意：gen.py CLI 默认 cc=1.54 / ch=1.09，非函数默认 1.53 / 1.10。）

### 优化 4：进程内调用，消除 FDF 往返

```python
# weak_scale_pe._generate_chain，默认 --in-process
from energy_first.pe_chain import generate_pe_chain
elements, coords = generate_pe_chain(n_c)  # 直接拿数组，无子进程、无 FDF 文本
mol = Molecule(list(elements), coords, label)
```

`--no-in-process` 可切回旧的 gen.py 子进程路径做对照。

### 优化 5：赝势共享（符号链接，省 ~14.5 GB）

```python
# --shared-pseudo（默认开）：只写一份到 _pseudos/，其余目录符号链接
shared_file = shared_dir / f"{el}.psf"     # 一份真实拷贝
dst.symlink_to(os.path.relpath(shared_file, job_dir))  # 其余符号链接
```

### 优化 6：full 基线开关（省 ~13 GB）

`--no-full-baseline` 跳过 E 级下无法运行的整链基线目录。两优化合计省 22680 节点目录 ~27 GB（65 GB → 38 GB）。

### 额外优化：h_by_carbon 分组向量化

```python
# numpy argsort + split 替代 Python 循环
hc = cs_arr[nn]                       # 每个 H 的碳下标（数组）
order = np.argsort(hc, kind="stable") # 按碳分组
uniq, starts = np.unique(hc[order], return_index=True)
groups = np.split(h_idx[order], starts[1:])
h_by_carbon = dict(zip(uniq.tolist(), [g.tolist() for g in groups]))
```

---

## 4. 展开过程：实测性能数据

### 4.1 小规模（1–16 节点，秒级）

**本机（dev，x86 113GB / 11TB 本地盘）—— 向量化后：**

| 节点 | wall | chain | fragments | write_inputs |
|---|---|---|---|---|
| 1 | 0.21s | 0.00s | 0.11s | 0.02s |
| 2 | 0.27s | 0.00s | 0.13s | 0.07s |
| 4 | 0.39s | 0.01s | 0.17s | 0.12s |
| 8 | 0.57s | 0.02s | 0.21s | 0.26s |
| 16 | 0.97s | 0.04s | 0.32s | 0.55s |

chain 从向量化前的 0.12-0.41s 降到 0.00-0.04s（**10× 提速**）。

**内网（ARM HPC 512GB / 共享存储）—— 向量化后：**

| 节点 | wall | chain | fragments | write_inputs | total |
|---|---|---|---|---|---|
| 1 | 3.60s | 0.01s | 2.38s | 0.16s | 2.57s |
| 2 | 3.49s | 0.02s | 1.86s | 0.71s | 2.62s |
| 4 | 4.31s | 0.06s | 1.96s | 1.18s | 3.24s |
| 8 | 5.73s | 0.08s | 2.27s | 2.47s | 4.88s |
| 16 | 17.69s | 0.15s | 3.71s | 12.93s | 16.93s |

### 4.2 全机 22680 节点（1.81 亿原子）

| 阶段 | 本机（dev） | 内网 | 倍数 | 说明 |
|---|---|---|---|---|
| chain | 34.6s | 212.7s | 6.1× | `list(elements)` 181M str 对象 |
| fragments | 563.5s | 6407.6s | **11.4×** | cKDTree 单线程 + ARM 单核弱 |
| **write_inputs** | **797.8s** | **16073.0s（4.5h）** | **20.1×** | 37GB 写共享存储，**2.3 MB/s** |
| artifacts | 1.5s | 162.2s | 108× | 68000 个 run_local.sh |
| **总 wall** | **1397s（23min）** | **22855s（6.3h）** | **16.4×** | |

### 4.3 向量化前后对比（本机 22680）

| 阶段 | 向量化前 | 向量化后 | 变化 |
|---|---|---|---|
| chain | 561.6s | 34.6s | **16× ↓** |
| fragments | 388.0s | 563.5s | 1.5× ↑（待排查）|
| write_inputs | 758.0s | 797.8s | ≈持平 |
| **总 wall** | **1710.6s（28.5min）** | **1397s（23.4min）** | **18% ↓** |

### 4.4 关键发现

1. **chain 向量化成功**（561→35s，16×），但 `list(elements)` 仍占 34.6s（1.81 亿元素 numpy→Python list 转换）。
2. **fragments 意外变慢**（388→564s）：可能因 numpy `lexsort`/`argsort` 在 1.81 亿规模上比 list comp + Python sort 更慢，待排查。
3. **write_inputs 是最长阶段**（798s / 16073s），完全由磁盘写带宽决定，算法无法优化。
4. **内网 write_inputs 灾难性慢**（4.5 小时，2.3 MB/s）——共享/网络文件系统的写瓶颈。

---

## 5. 收集过程：算法与文件系统开销

### 算法

`combine_results.py` 的收集计算是 **O(N) 求和**：

```
E^(1) = Σ E(block_i) − Σ E(cap_c)                        # MFCC(1)
E^(2) = E^(1) + Σ_c [E(dimer_c) − E(block_c) − E(block_{c+1}) + E(cap_c)]  # MBE(2)
```

纯求和本身极快（22680 节点 < 0.6s）。**但实际瓶颈不在求和，而在读取文件**：combine 需要打开 + 读取 + regex 扫描 ~68000 个真实大小的 siesta.out 文件（block ~1.4 MB、dimer ~2.5 MB、cap ~27 KB，共 ~90 GB）。

### 文件系统开销分解

每个文件的 combine 处理 = `open()` + `read()` + `TOTAL_RE.findall(text)` + 取 `[-1]`。对 1.4 MB 的 block siesta.out（28000 行），`findall` 要 regex 扫描全文找 "siesta:.*Total =" 匹配——这是 CPU + I/O 混合开销。

### 测量方法

`bench_combine.py` 使用真实 siesta.out 模板（取自内网 ws8x8000 实测），复制到 N 节点，放入真实 `combine_results.py` 执行——测量的是**真实的文件读取 + 正则扫描 + 求和**全链路时间，而非 1 行假文件的玩具版。

---

## 6. 收集过程：实测性能数据

### 6.1 小规模（1–16 节点）

| 节点 | 文件数 | 字节 | 本机 combine | 内网 combine |
|---|---|---|---|---|
| 1 | 1 | 0.00G | 0.04s | 0.16s |
| 2 | 4 | 0.01G | 0.05s | 0.25s |
| 4 | 10 | 0.01G | 0.08s | 0.43s |
| 8 | 22 | 0.03G | 0.16s | 0.81s |
| 16 | 46 | 0.06G | 0.30s | 1.90s |

内网比本机慢约 **5-6×**（文件系统读取 + 正则开销）。

### 6.2 全机 22680 节点

| | 本机（dev） | 内网 | 倍数 |
|---|---|---|---|
| 文件数 | 68038 | 68038 | — |
| 字节 | 90.55 GB | 90.55 GB | — |
| 造数据 | 301.7s | 1283.7s | 4.3× |
| **combine** | **540.8s（9min）** | **2708.1s（45min）** | **5.0×** |
| 其中 parse+sum | — | 2706.6s | — |

内网 combine 2708s 的有效读带宽：90 GB / 2708s ≈ **33 MB/s**（本机 167 MB/s）。

### 6.3 对比：玩具版 vs 真实文件版

| 方法 | 22680 combine | 说明 |
|---|---|---|
| 玩具版（1 行假文件）| 0.59s | 仅测求和，不含文件 I/O |
| **真实文件版** | **540.8s（本机）/ 2708s（内网）** | 含文件读取 + 正则 |

差距 **900-4600×**——文件 I/O 完全主导。

---

## 7. 内外网对比与环境差异分析

### 7.1 逐阶段倍数

| 阶段 | 本机 | 内网 | 倍数 | 根因 |
|---|---|---|---|---|
| chain（CPU）| 34.6s | 212.7s | 6.1× | ARM 单核弱 + `list()` 转换 |
| fragments（CPU+I/O）| 563.5s | 6407.6s | 11.4× | cKDTree 单线程 + numpy ARM 效率 |
| write_inputs（I/O）| 797.8s | 16073.0s | **20.1×** | **共享存储写 2.3 MB/s** |
| combine 读（I/O）| 540.8s | 2708.1s | 5.0× | 共享存储读 33 MB/s |
| artifacts（I/O）| 1.5s | 162.2s | 108× | 68000 小文件元数据写 |

### 7.2 有效带宽推算

| 操作 | 本机 | 内网 | 判断 |
|---|---|---|---|
| 写 FDF（write_inputs）| 46 MB/s | **2.3 MB/s** | 网络存储写死瓶颈 |
| 读 siesta.out（combine）| 167 MB/s | 33 MB/s | 网络存储读也慢 5× |
| 复制文件（fabricate）| ~300 MB/s | ~70 MB/s | 读比写快 30× |

### 7.3 结论

**内网 22680 展开耗时 6.3 小时，其中 4.5 小时（72%）是写共享存储**。这不是算法问题，是部署架构问题——生成输出写到了 `/share/honpas/xzz/`（网络/共享文件系统），其写带宽仅 2.3 MB/s。

---

## 8. 已识别的后续优化方向

### 立即见效（部署层面，不改代码）

| 方向 | 预期收益 | 难度 |
|---|---|---|
| **写节点本地盘**（`/tmp`、`/scratch`、`/dev/shm`）| write_inputs 4.5h → 秒级 | 仅改 `--out-dir` |
| **combine 输入放本地** | 45min → ~9min | 同上 |

### 代码层面

| 方向 | 预期收益 | 难度 | 说明 |
|---|---|---|---|
| **cKDTree.query(workers=−1)** | fragments 6408s → ~300-500s | 一行 | 吃满 28 核 |
| **write_inputs multiprocessing.Pool** | 4.5 万文件并行格式化+写 | 中 | 需先 `iostat` 确认不是单盘 IO bound |
| **`list(elements)` 消除** | chain 35s → ~1s | 低 | Molecule 直接接受 numpy array |
| **分布式按片段生成** | 每节点 O(8000)，彻底弱扩展 | 高 | 每节点仅生成自己的 block，不走中心化 |
| **fragments 变慢排查** | 388→564s 回退 | 低 | 对比 lexsort vs list-comp 在 1.81 亿规模 |

### 优先级排序

1. **🔴 写本地盘**（0 改动，4.5h → 秒级）——内网最关键
2. **🟡 cKDTree workers=−1**（一行，fragments 10× ↓）
3. **🟡 `list(elements)` 消除**（chain 再 35× ↓）
4. **🟢 分布式按片段生成**（根治，但工程量大）

---

## 9. 附录：复现方法

### 展开 benchmark

```bash
cd Vayesta/benchmark
# 小规模（秒级）
python3 bench_gen.py --gen-script <gen.py> --pseudo-dir <pseudos> [--pythonpath <venv>]
# 全机（22680 节点）
python3 bench_gen.py --gen-script <gen.py> --pseudo-dir <pseudos> --nodes 22680
```

`bench_gen.py` 调用 `weak_scale_pe.py`（默认 `--in-process` 向量化生成），记录 wall 与 `[gen-time]` 分项。

### 收集 benchmark

```bash
cd Vayesta/benchmark
# 小规模（秒级）
python3 bench_combine.py
# 全机（22680 节点，~90GB 造+读）
python3 bench_combine.py --nodes 22680
```

`bench_combine.py` 复制真实 siesta.out 模板（`templates/block.out` ~1.4MB、`dimer.out` ~2.5MB、`cap.out` ~27KB）到 N 节点，放入真实 `combine_results.py` 执行。

### 文件说明

| 文件 | 用途 |
|---|---|
| `benchmark/bench_gen.py` | 展开 benchmark（portable，argparse 无硬编码） |
| `benchmark/bench_combine.py` | 收集 benchmark（portable，模板自带） |
| `benchmark/templates/*.out` | 真实 siesta.out 模板（取自内网 ws8x8000） |
| `benchmark/templates/combine_results.py` | 真实 combine 脚本 |
| `exascale_sim/gen_projection.py` | h_to_c 标度外推工具（brute vs cKDTree） |
| `exascale_sim/combine_sim.py` | 纯求和时间模拟（不含文件 I/O） |
| `generation_optimization.md` | 生成优化工程文档（本文件的前身） |

### 版本历史

| commit | 内容 |
|---|---|
| `fabb4047` | cKDTree H→C + 计时 + E级外推 |
| `ceda179c` | 赝势共享 + full 基线开关 |
| `4fa2678c` | 修第二个 O(N²)（block-H 扫描）+ full-runner 崩溃修复 |
| `278b6a75` | 向量化 PE 生成 + 进程内调用 + h_by_carbon 向量化 |
| `64e9fa2a` | benchmark 目录建立 |
| `a1aa3e22` | combine benchmark 改用真实 siesta.out |
| `8f636f92` | 英文文件名 + bench_gen 可移植化 |
| `28473392` | exascale_sim 脚本可移植化 |
