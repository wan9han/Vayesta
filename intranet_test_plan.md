# 华为内网测试方案与指令

> 本文件描述如何在华为内网 HPC 专用芯片集群上完整复现本项目的三大验证：
> ①蛋白质 MFCC+MBE(2) 普适性 ②PE 弱扩展正确性与性能 ③全机收集性能。
> 每项给出前置准备、精确指令、预期结果。

---

## 0. 前置准备

### 0.1 拉取代码

```bash
cd /share/honpas/xzz/energy_first
git clone ssh://git@github.com/wan9han/Vayesta.git Vayesta-test
cd Vayesta-test
```

如果已有仓库，直接拉取两个分支：
```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first   # 或你现有的仓库路径
git fetch origin
git checkout energy_first   # PE 弱扩展 + benchmark
git checkout protein        # 蛋白质验证（基于 energy_first）
```

### 0.2 确认赝势

```bash
# 需要 C/H/N/O 四种赝势
PSEUDO_DIR=/share/honpas/xzz/energy_first/pseudo   # 改成你的赝势目录
ls $PSEUDO_DIR/{C,H,N,O}.psf
# 如果缺 N.psf / O.psf，从 SIESTA 安装目录拷：
# cp /path/to/siesta/Tests/Pseudos/N.psf $PSEUDO_DIR/
# cp /path/to/siesta/Tutorials/Bases/O/O.psf $PSEUDO_DIR/
```

### 0.3 确认 SIESTA 可执行文件

```bash
SIESTA_BIN=/share/honpas/xzz/siesta-20260520/siesta/build-clang/Src/siesta
# 或你内网的 SIESTA 路径
$SIESTA_BIN --version 2>&1 | head -3   # 确认能跑
```

### 0.4 确认 Python + numpy + scipy

```bash
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
# 如果缺，用内网 pip 安装或加载已有 module
```

### 0.5 设置环境变量（后续命令中引用）

```bash
export SIESTA_BIN=/share/honpas/xzz/siesta-20260520/siesta/build-clang/Src/siesta
export PSEUDO_DIR=/share/honpas/xzz/energy_first/pseudo
export GEN_SCRIPT=/share/honpas/xzz/energy_first/Vayesta-energy_first/gen.py
export OUT_BASE=/share/honpas/xzz/energy_first/test_results
mkdir -p $OUT_BASE
```

---

## 1. 蛋白质 MFCC+MBE(2) 验证（protein 分支）

### 目的

证明 MFCC+MBE(2) 算法不仅适用于聚乙烯（C-C 键切割，C/H 二元素），也适用于蛋白质骨架（C'-N 肽键切割，C/H/N/O 四元素）。这是回应评审"验证算法普适性"的核心证据。

### 指令

```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first   # 你的仓库路径
git checkout protein

# 单节点跑聚甘氨酸 MFCC+MBE(2) 验证
# 6 个甘氨酸残基（44 原子），切 1 个和 2 个肽键
# 约 12 个 SIESTA 小作业（每个 < 1 分钟）
python3 protein_validate.py --siesta-bin $SIESTA_BIN \
  --n-residues 6 \
  --cuts 1 2 \
  --work-root $OUT_BASE/protein_gly6 --pseudo-dir pseudos

# 更大体系（可选）：12 残基，切 2/3/4 个肽键
python3 protein_validate.py --siesta-bin $SIESTA_BIN \
  --n-residues 12 \
  --cuts 2 3 4 \
  --work-root $OUT_BASE/protein_gly12 --pseudo-dir pseudos
```

### 预期结果

```
polyglycine(6): 44 atoms (C/H/N/O=12/20/6/6), 5 peptide bonds
E_full = -6326.76xx eV

=== 1 cut(s) ===
  MFCC(1): -6325.8xxxx  err=+0.9xxxx eV (+0.9xxxx/cut)
  MBE(2) : -6326.76xxxx  err=+0.0000 eV          ← 精确复现

=== 2 cut(s) ===
  MFCC(1): -6324.9xxxx  err=+1.8xxxx eV (+0.9xxxx/cut)
  MBE(2) : -6326.6xxxxx  err=+0.1xxxx eV (+0.05xxxx/cut)  ← 17x 改善
```

**判据**：
- 1 切口 MBE(2) 误差 = 0.0000（N=2 恒等，与 PE 一致）
- 2 切口 MBE(2) 每切口误差 < 0.1 eV（三体残差）
- MFCC(1) 每切口误差 ~0.9 eV（与 PE 的 ~1.24 eV 同量级）

### 需要关注的点

- N.psf / O.psf 的基组是否与 C.psf / H.psf 一致（SZ）。如果 SIESTA 报 basis 警告，需要换赝势。
- 如果 SCF 不收敛，加大 `MaxSCFIterations`（在 `energy_first/molecule.py` 的 `write_siesta_fdf` 中，当前 100）。
- 内网 ARM 单核速度可能不同，但 44 原子的小体系应在 1-2 分钟内完成。

---

## 2. PE 弱扩展正确性 + 性能（energy_first 分支）

### 目的

在内网 HPC 专用芯片集群上复现弱扩展测试：1/2/4/8/16 节点 × 8000 原子/节点，验证并行效率 ~100%，以及 16×500 分块 MBE(2) 能量精度。

### 2.1 正确性验证（快速，单节点即可）

```bash
git checkout energy_first

# 1×8000 整链基准（E_full）
python3 weak_scale_pe.py \
  --atoms-per-node 8000 --num-nodes 1 \
  --pseudo-dir $PSEUDO_DIR --out-dir $OUT_BASE/ws_1x8000 \
  --gen-script $GEN_SCRIPT --solution-method ntpoly \
  --hosts <内网节点1的IP>

cd $OUT_BASE/ws_1x8000
# 跑 SIESTA（需要内网的 mpirun + NTPoly 环境）
# 手动跑或用生成的 run_local.sh
bash ./full/run_local.sh    # E_full

# 16×500 分块 MBE(2)
cd /share/honpas/xzz/energy_first/Vayesta-energy_first
python3 weak_scale_pe.py \
  --atoms-per-node 500 --num-nodes 16 \
  --pseudo-dir $PSEUDO_DIR --out-dir $OUT_BASE/ws_16x500 \
  --gen-script $GEN_SCRIPT --solution-method ntpoly \
  --hosts <节点1> <节点2> ... <节点16>

cd $OUT_BASE/ws_16x500
bash ./submit_per_node_local.sh
python3 combine_results.py
# 对比 E_mbe2 vs E_full，误差应 ~0.037 eV
```

### 2.2 弱扩展性能（多节点）

```bash
# 生成 1/2/4/8/16 节点的输入
for N in 1 2 4 8 16; do
  python3 weak_scale_pe.py \
    --atoms-per-node 8000 --num-nodes $N \
    --pseudo-dir $PSEUDO_DIR --out-dir $OUT_BASE/ws_${N}x8000 \
    --gen-script $GEN_SCRIPT --solution-method ntpoly \
    --no-full-baseline \
    --hosts <节点1> <节点2> ... <节点N>
done

# 每个规模分别跑（内网 mpirun + NTPoly）
for N in 1 2 4 8 16; do
  cd $OUT_BASE/ws_${N}x8000
  bash ./submit_per_node_local.sh
  cd -
done

# 收集每节点的 BLAS / solver 时间
# 用 ws_stats.py（需要拷到输出目录）
cp /share/honpas/xzz/energy_first/Vayesta-energy_first/benchmark/../ws_stats.py \
   $OUT_BASE/ws_${N}x8000/  # 对每个 N
cd $OUT_BASE/ws_${N}x8000 && python3 ws_stats.py
```

### 预期结果

| 节点 | BLAS dgemm (ms/call) | solver PSM (ms/step) | 弱扩展效率 |
|---|---|---|---|
| 1 | ~111 | ~139 | 100% |
| 2 | ~112 | ~140 | ~100% |
| 4 | ~111 | ~139 | ~100% |
| 8 | ~111 | ~140 | ~100% |
| 16 | ~111 | ~140 | ~100% |

正确性：E_mbe2(16×500) vs E_full(1×8000) 误差 < 0.05 eV。

### 需要关注的点

- 内网用 `SolutionMethod ELSI` + `ELSI.Solver ntpoly`（脚本已配置），需确认 ELSI/NTPoly 环境正常。
- `--hosts` 填内网实际节点 IP（8 台 71.20.27.* + 8 台 71.20.16.*）。
- `--ssh-user` 如果需要（如 `--ssh-user xzz`）。
- 如果 `submit_per_node_local.sh` 的 UCX/MPI 参数与内网不一致，手动调 `honpas_env.sh`。

---

## 3. 展开 + 收集 Benchmark（energy_first 分支）

### 目的

测量全机 22680 节点规模下，体系展开（生成）和结果收集（combine）两个流程的耗时。

### 3.1 收集 benchmark（无外部依赖，秒级 → ~45min）

```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first/benchmark

# 小规模验证（秒级）
python3 bench_combine.py

# 全机 22680（造 90GB + 读 90GB，~45min）
python3 bench_combine.py --nodes 22680
```

### 3.2 展开 benchmark

```bash
# 小规模（秒级）
python3 bench_gen.py \
  --gen-script $GEN_SCRIPT \
  --pseudo-dir $PSEUDO_DIR \
  --nodes 1 2 4 8 16

# 全机 22680（向量化生成 + 写 37GB，~6h on 共享存储）
# ⚠ 建议 --out-dir 用节点本地盘（/tmp 或 /scratch），不要写 /share/
python3 bench_gen.py \
  --gen-script $GEN_SCRIPT \
  --pseudo-dir $PSEUDO_DIR \
  --nodes 22680
```

### 预期结果

| 指标 | 本机（dev）实测 | 内网预估 |
|---|---|---|
| combine 22680 | 541s（9min） | ~2700s（45min） |
| gen 22680 wall | 1397s（23min） | ~6h（共享存储写瓶颈） |
| gen chain | 35s（向量化） | ~210s |
| gen write_inputs | 798s | ~16000s（4.5h） |

### 关键建议

**gen 的 --out-dir 一定要用本地盘**：
```bash
# ❌ 慢（共享存储写 2.3 MB/s）
--out-dir /share/honpas/xzz/test_results/gen_22680

# ✅ 快（本地盘写 ~GB/s）
--out-dir /tmp/gen_22680
# 或
--out-dir /dev/shm/gen_22680
```

如果用本地盘，gen 22680 的 write_inputs 从 4.5h → ~分钟级。

---

## 4. 结果汇总模板

跑完后，把以下数据发给我（手打或粘贴日志关键行）：

### 4.1 蛋白质验证

```
polyglycine(N): XX atoms, X peptide bonds
E_full = -XXXX.XXXX eV
1 cut:  MBE(2) err = X.XXXX eV
2 cuts: MBE(2) err = X.XXXX eV (X.XXXX/cut)
```

### 4.2 PE 弱扩展

```
# ws_stats.py 输出
# SUMMARY  dgemm=XXX  PSM=XXX  GFLOPS=XXXX  wall=XXX
# OLD      dgemm=XXX  ...
# NEW      dgemm=XXX  ...
```

### 4.3 Benchmark

```
# bench_combine.py
22680 | 68038 | 90.55G | XXXs | XXXs

# bench_gen.py
22680 | XXXs | chain=XXX  fragments=XXX  write_inputs=XXX
```

---

## 5. 排错指南

| 问题 | 原因 | 解决 |
|---|---|---|
| `ModuleNotFoundError: numpy/scipy` | Python 缺库 | `pip install numpy scipy` 或 `module load python/...` |
| SIESTA 报 `Pseudopotential file not found: N.psf` | 缺 N/O 赝势 | 从 SIESTA Tests/Pseudos/ 拷 |
| `submit_per_node_local.sh` MPI 报错 | UCX/MPI 环境差异 | 调 `honpas_env.sh` 中的 UCX_TLS / NET_DEVICE |
| gen 22680 目录为空很久 | 正常（chain+fragments 阶段不写文件）| 等 ~15-20min 后 block 开始出现 |
| combine 22680 很慢 | 共享存储读 33 MB/s | 正常现象（内网 5x 慢于本地） |
| SCF 不收敛 | 赝势基组不匹配 | 换 SZP 或加大 MaxSCFIterations |
