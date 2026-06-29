# 性能 Benchmark：展开（生成）与收集（combine）

在 1 / 2 / 4 / 8 / 16 / **22680** 节点规模下测试"体系展开"与"结果收集计算"两过程的耗时。配置：每节点 8000 原子，`--no-full-baseline --shared-pseudo`。

## 收集（combine）—— 用 16×500 真实结果做模板复制到各规模

`bench_combine.py` 以 16×500 真实能量（`template_16x500.json`）为模板，cycle 到 N 节点，造 siesta.out + schedule.json，执行 MFCC(1)+MBE(2) 收集计算并计时。

| 节点 | 造数据 | **combine** | 文件数 |
|---|---|---|---|
| 1 | 0.00s | 0.0000s | 1 |
| 2 | 0.00s | 0.0001s | 4 |
| 4 | 0.00s | 0.0001s | 10 |
| 8 | 0.00s | 0.0002s | 22 |
| 16 | 0.00s | 0.0006s | 46 |
| **22680** | 2.12s | **0.591s** | 68038 |

**结论：收集过程 O(N) 求和，全机 22680 节点仅 0.59 s，完全不构成瓶颈。** 16 节点 E_total=-492590.5 eV 与真实结果一致（验证 combine 逻辑正确）。

## 展开（生成）—— `weak_scale_pe.py` 生成 block/dimer/cap 输入

`bench_gen.py` 调 weak_scale_pe.py 在各规模生成，记录 wall 与 `[gen-time]` 分项。

| 节点 | wall | chain(gen+parse) | fragments(h_to_c+build) | write_inputs |
|---|---|---|---|---|
| 1 | 0.33s | 0.12s | 0.08s | 0.04s |
| 2 | 0.37s | 0.11s | 0.10s | 0.08s |
| 4 | 0.48s | 0.16s | 0.12s | 0.13s |
| 8 | 0.74s | 0.24s | 0.18s | 0.24s |
| 16 | 1.28s | 0.40s | 0.28s | 0.51s |
| **22680** | **~28 min** | ~8 min | ~6 min | ~12.5 min |

22680 节点（1.81 亿原子）：实测 wall **28 分 14 秒**、峰值内存 **81.4 GB**、输出 **37 GB**（22680 block + 22679 dimer + 22679 cap，每个 input.fdf + 赝势符号链接 + run_local.sh）。其中：
- chain：gen.py 生成 6048 万碳（Python 循环）+ 解析 12.7 GB FDF；
- fragments：cKDTree 6000 万碳建树 + 1.21 亿 H 查询 + 建 22680+22679 个片段分子；
- write_inputs：格式化并落盘 ~37 GB FDF 文本（**当前最长阶段**）。

**结论：生成可行**（28 min / 81 GB / 37 GB）。两个原 O(N²) 瓶颈（h_to_c、block-H 分配）均已修为 O(N)/O(N log N)。现长尾是 write_inputs 的中心化 FDF 落盘与 gen.py 的 Python 循环——由"分布式按片段生成"可消除（每节点 O(8000)）。

## 复现

```bash
python3 bench_combine.py          # 收集 benchmark（全规模，秒级）
python3 bench_gen.py              # 展开 benchmark（1/2/4/8/16，秒级）
python3 bench_gen.py 22680        # 22680 节点（~28 min，81 GB 内存）
```

> 注：`gen_*`、`combine_*` 为测试输出（22680 的 gen 目录 ~37 GB），已 gitignore。
