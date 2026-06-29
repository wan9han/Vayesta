# 性能 Benchmark：展开（生成）与收集（combine）

在 1 / 2 / 4 / 8 / 16 / **22680** 节点规模下测试"体系展开"与"结果收集计算"两过程的耗时。配置：每节点 8000 原子，`--no-full-baseline --shared-pseudo`。

## 收集（combine）—— 复制真实 siesta.out 到 N 节点，用完整 combine_results.py

`bench_combine.py` 把**真实** siesta.out 模板（`templates/block.out` ~1.4MB、`dimer.out` ~2.5MB、`cap.out` ~27KB，取自内网实测）复制到 N 节点的 block/cap/dimer 目录，放入真实的 `combine_results.py` 并执行——测量的是真实的"打开+读取若干个真实大小文件 + 正则 + 求和"的文件系统开销，而非 1 行假文件。

| 节点 | 文件数 | 字节 | 造数据 | **combine** |
|---|---|---|---|---|
| 1 | 1 | 0.00G | 0.0s | 0.04s |
| 2 | 4 | 0.01G | 0.0s | 0.05s |
| 4 | 10 | 0.01G | 0.0s | 0.08s |
| 8 | 22 | 0.03G | 0.0s | 0.16s |
| 16 | 46 | 0.06G | 0.0s | 0.30s |
| **22680** | 68038 | ~90G | TBD | **TBD（内外网实测）** |

22680 节点需读 ~90 GB / 68038 文件，由内外网分别实测填入。按 16 节点（0.06GB→0.30s）外推，纯读取约 ~7 min（磁盘带宽主导），另含 68000 次文件打开/正则开销。

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
# 收集 benchmark：复制真实 siesta.out 模板到 N 节点 + 跑真实 combine_results.py。
#   1/2/4/8/16 秒级；22680 需 ~90GB 磁盘、~10min（造数据 + 读 90GB）。
#   无外部依赖（纯标准库，模板自带在 templates/）。
python3 bench_combine.py                   # 默认 1/2/4/8/16
python3 bench_combine.py --nodes 22680     # 全机规模
# 展开 benchmark（gen-script/pseudo-dir 必填；--pythonpath 仅当系统 python3 缺 numpy/scipy 时）
python3 bench_gen.py --gen-script <gen.py> --pseudo-dir <pseudos> [--pythonpath <venv>]
python3 bench_gen.py --gen-script <gen.py> --pseudo-dir <pseudos> --nodes 22680   # ~28min, 81GB 内存
```

> 注：`gen_*`、`combine_*` 为测试输出（22680 的 gen 目录 ~37 GB），已 gitignore。
