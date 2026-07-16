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

## 1. 蛋白质 MFCC+MBE(2) 验证（protein 分支，TRS2 求解器）

### 目的

证明 MFCC+MBE(2) 算法在 **TRS2（NTPoly 密度矩阵纯化）求解器**下，不仅适用于聚乙烯（C-C 键切割），也适用于蛋白质骨架（C'-N 肽键切割，C/H/N/O 四元素）。这是回应评审"验证算法普适性 + 专用求解器"的核心证据。

### 关键：为什么是 ACE/NME 封端的 α-螺旋聚甘氨酸

裸聚甘氨酸 `-(NH-CH2-CO)n-` 在 TRS2 下 **永不收敛**，原因有二（均已实测）：
1. **端基电荷转移态 ~0.06 eV**（与链长无关）：自由 N 端 -NH2（浅 HOMO）→ 自由 C 端 -COOH（深 LUMO）。
2. β-strand 还有随链长收缩的带隙（0.70 eV @16 残基 → 0.11 eV @64 残基）。

NTPoly 用**尖锐 0/1 占据投影**做纯化（提高 ElectronicTemperature 无效），无法分辨 <~0.3 eV 的能隙 → SCF 振荡。

**修复（三件套，都已写入代码）**：
- **ACE/NME 端基封端** `Ac-(Gly)n-NHMe`：去掉端基 CT 态，能隙 0.06 → ~1.0 eV（且随链长增长，是真实的酰胺体带隙）。`energy_first/peptide_chain.py:generate_polyglycine`。
- **α-螺旋构象** (phi=-57, psi=-47)：紧凑折叠，稳定的 ~1 eV 体带隙；PCA 旋转把螺旋轴对齐 x，网格保持 ~L×5×5（线性）而非 L³。
- **`SCF.H.Converge .false.`**（`molecule.py` ntpoly 分支）：~1 eV 能隙边缘仍有一个轨道使 dHmax 振荡（~3-8 eV），但密度矩阵和总能量已收敛；按 DM 判据（dDmax<1e-4）宣告收敛。对 PE（大能隙）无害。

### 指令

```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first   # 你的仓库路径
git checkout protein

# 单节点 MFCC+MBE(2) 验证，TRS2 求解器（默认）
# 16 残基封端 α-螺旋（124 原子），切 2 个内部肽键
# protein_validate.py 默认 --solution-method ntpoly（即 TRS2）
python3 protein_validate.py --siesta-bin $SIESTA_BIN \
  --n-residues 16 \
  --cuts 2 \
  --solution-method ntpoly \
  --work-root $OUT_BASE/protein_gly16 --pseudo-dir pseudos

# 注：protein_validate.py 单进程跑（无 mpirun），TRS2 在单核上较慢；
# 内网若要多核，用 weak_scale_protein.py（每节点 mpirun）。
```

### 预期结果（本机 dev 实测，16 残基 / 2 切口，全 TRS2 收敛）

```
capped alpha-helix polyglycine(16): 124 atoms, 17 amide bonds (15 internal), 2 cuts
E_full    = -18074.0940 eV   (TRS2, 36 步收敛)

E_MFCC(1) = -18069.9710   err = +4.123 eV (+2.06/cut)   ← 常规 H-cap 误差
E_MBE(2)  = -18074.0920   err = +0.002 eV (+0.001/cut)  ← TRS2+MBE(2) 精确复现
```

### 更大规模（48 残基 / 4 切口，全 TRS2，本机 dev 实测）

5 片段（~70 原子/片段）+ 4 二聚体 + 4 cap + full（348 原子），14 个 SIESTA 作业全部 TRS2 收敛：

```
E_full    = -51714.9312 eV   (TRS2, 56 步收敛)
E_MFCC(1) = -51701.3395       err = +13.592 eV (+3.398/cut)
E_MBE(2)  = -51714.5753       err = +0.356 eV  (+0.089/cut)   ← 38x 改善
每个切口二体增量: -3.24, -3.31, -3.36, -3.33 eV
```

**判据**：
- 所有作业（full / 片段 / 二聚体 / cap）在 TRS2 下都收敛（dDmax<1e-4，`SCF cycle converged`）。
- MBE(2) 每切口误差：2 切口 ~0.001 eV/cut；4 切口 ~0.09 eV/cut（随切口数/片段增大，三体残差增长；C'-N 肽键有极性，三体残差大于 PE 的 C-C）。
- MFCC(1) 每切口误差 ~2-3 eV（C'-N 肽键切割的 H-cap 误差，被 MBE(2) 二聚体校正抵消）。
- 若需 <0.01 eV/cut，可升级到 MBE(3)（三体校正，成本 ~N² 个三聚体）。

### 需要关注的点

- **切口只取内部肽键**：ACE-N0 和 C'n-NME 是封端产生的两个端基酰胺键，不能切（否则把 cap 切下来）。代码里用 `bonds[1:-1]` 排除。
- N.psf / O.psf 基组与 C.psf / H.psf 一致（SZ）。若 SIESTA 报 basis 警告，换赝势。
- 内网 ARM 单核较慢；TRS2 收敛约 30-45 步，124 原子单核 ~10-15 分钟/作业。
- 若 dHmax 不收敛但 dDmax<1e-4 且能量稳定 → 正常（小能隙边缘轨道），`SCF.H.Converge .false.` 已处理。

---

## 1b. 蛋白质多节点弱扩展 sweep（protein 分支，TRS2，一键 1/2/4/8/16）

### 目的

用 `weak_scale_protein_all.py` 一次性跑完 1/2/4/8/16 节点弱扩展（每节点固定残基数，总规模随节点数增长），全部在 **TRS2 + MBE(2)** 下完成。证明蛋白质体系也能弱扩展，且 MFCC+MBE(2) 在多切口下能量正确。

### 指令

```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first   # 你的仓库路径
git checkout protein

# --hosts 有默认值（weak_scale_pe.DEFAULT_HOSTS 里的 16 个内网节点），不用手填。
# --remote-out-dir 默认 = --big-out 的绝对路径；big-out 必须放在各节点共享可见的 /share/ 下。
# --executor 默认 submit（即 submit_per_node_local.sh，SSH 登录各节点跑 run_local.sh）。
python3 weak_scale_protein_all.py \
  --residues-per-node 50 \
  --nodes 1 2 4 8 16 \
  --big-out /share/honpas/xzz/energy_first/test_results/ws_protein_sweep
# 如要覆盖默认 hosts：
#   --hosts <节点1> <节点2> ... <节点16>
```

**参数说明**：
- `--residues-per-node`：每节点残基数（固定 = 弱扩展）。建议 50（封端 α-螺旋 ~360 原子/节点，TRS2 收敛、HBM 内存够）。内网若要更大矩阵可加到 100，但每节点 SCF 步数和时间会涨。
- `--hosts`：**有默认值**（16 个内网 IP：8 个 71.20.27.* + 8 个 71.20.16.*，同 `weak_scale_pe.py`）。每个规模自动取前 N 个节点，**不考虑机器复用**。
- `--big-out`：顶层输出文件夹，**必须放 /share/ 共享路径**（各节点要能 SSH 进去 cd 到子目录跑 run_local.sh）。
- `--executor`：默认 `submit`（推荐 SSH launcher）；`dev-local` 是单机测试模式（本机 mpirun，不 SSH，不需内网）。

### 输出结构

```
ws_protein_sweep/
  n01/ n02/ n04/ n08/ n16/            # 每个规模一个文件夹（小→大）
  weak_scale_summary.json             # 跨规模汇总（顶层）
  每个 nXX/ 内：
    schedule.json              ← json #1（生成调度）
    weak_scaling_results.json  ← json #2（MFCC/MBE(2) 能量，combine 产出）
    block_*/siesta.out  dimer_*/siesta.out  cap_*/siesta.out   # 各作业日志
    submit_per_node_local.sh  combine_results.py  launch_logs/
```

n=1 特殊：0 切口 → 无 dimer/cap，但 `combine_results.py` 仍产出第二个 json（MFCC(1) only，单 block 即整链基线）。

### 预期（弱扩展判据）

- 每个规模所有作业在 TRS2 下收敛（`weak_scaling_results.json` 里 `E_total_ev` 非 null，`missing_outputs` 为空）。
- n≥2 的 `method` = `MBE(2)`（n=1 是 MFCC(1) 整链基线）。
- 每残基能量 `E_total / (residues_per_node × N)` 随 N 增大趋于稳定（边界/cap 效应减弱）；R 够大（≥50）时各规模每残基能量接近一致。
- 各节点 block 的 SCF 墙钟时间大致恒定（弱扩展效率，看 `launch_logs/block_*.log`）。

---

## 1c. PEG 大规模弱扩展（TRS2，PE 同等规模）★推荐的大规模案例

### 为什么用 PEG 而不是蛋白质做大规模

蛋白质（肽键骨架）是共轭酰胺高分子，能隙随链长收缩到 ~0.1 eV（实测：64 残基 0.105 eV，128 残基 0.099 eV，连真实泛素折叠都只有 0.08 eV）—— **TRS2 在 >~100 残基就发散**，做不到 PE 规模。这是物理本质（酰胺 π 共轭沿键贯穿，与构象/折叠无关），调参/封端/切位都救不了。

**PEG（聚乙二醇 -(CH₂CH₂O)-）是饱和 σ 键高分子**（无共轭 π），是绝缘体，能隙 **~1 eV 且不随链长收缩**（实测 64 单元 1.04 eV，128 单元 0.99 eV）。所以 **TRS2 在任意尺度都干净收敛**（898 原子 18 步收敛，dHmax 0.005 eV，无需任何 trick）。C/H/O（赝势现成）、切 C-O 醚键（新键型，证明 MFCC 普适到 C-C 之外）、生物相容性高分子。**这是能做到 PE 同等规模的案例。**

### 单节点 MFCC+MBE(2) 验证

```bash
cd /share/honpas/xzz/energy_first/Vayesta-energy_first
# 16 单元 PEG，切 1/2 个 C-O 键，全 TRS2。约 5 个小作业。
python3 peg_validate.py --siesta-bin $SIESTA_BIN \
  --units 16 --cuts 1 2 \
  --work-root $OUT_BASE/peg_val --pseudo-dir pseudos
```

预期（dev 实测）：MBE(2) 每切口误差 ~0.01 eV（1 切口精确复现）。

### 一键 1/2/4/8/16 节点弱扩展 sweep（PE 同等规模）

```bash
# --units-per-node 1100 ≈ 7700 原子 ≈ PE 的 8000 原子/节点矩阵规模。
# --hosts 默认 16 个内网节点；--big-out 必须放 /share/ 共享路径。
python3 weak_scale_peg_all.py \
  --units-per-node 1100 \
  --nodes 1 2 4 8 16 \
  --big-out /share/honpas/xzz/energy_first/test_results/ws_peg_sweep
# 默认 --executor submit（submit_per_node_local.sh，SSH 推荐 launcher）
```

**输出结构**（每个规模一个文件夹，日志 + 2 个 json）：
```
ws_peg_sweep/
  n01/ n02/ n04/ n08/ n16/
  weak_scale_summary.json          # 跨规模汇总
  每个 nXX/: schedule.json, weak_scaling_results.json,
            block_*/siesta.out, dimer_*/siesta.out, cap_*/siesta.out
```

**判据**：
- 每个规模所有作业在 TRS2 下收敛（`weak_scaling_results.json` 里 `E_total_ev` 非 null）。
- n≥2 的 `method` = `MBE(2)`；MBE(2) 每切口误差 < 0.05 eV。
- 每单元能量 `E_total/(units_per_node×N)` 随 N 增大稳定（弱扩展一致性）。
- 各节点 block 的 SCF 墙钟时间大致恒定（弱扩展效率）。

### 需要关注的点
- PEG 能隙 ~1 eV（稳定），TRS2 平静收敛，**不需要** `SCF.H.Converge .false.`（但 molecule.py ntpoly 分支带了这个设置，对 PEG 无害）。
- 切口只取内部 C-O 醚键（O 连 2 个 C）；代码用 `_internal_co_bonds` 自动排除 2 个末端 C-OH 键。
- 片段化用**重原子连通图**（`fragment_peg`），避免 PEG 紧密螺旋里 H···H(<1.7Å) 把片段错误连通。

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
