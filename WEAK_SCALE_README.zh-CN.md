# 弱扩展测试中文说明

本文说明如何使用 [weak_scale_pe.py](/home/xzz2/huawei-siesta-energy-first/Vayesta/weak_scale_pe.py) 生成并运行当前项目的弱扩展测试。

## 1. 脚本做什么

`weak_scale_pe.py` 会做三件事：

1. 按目标规模生成一条聚乙烯（PE）链。
2. 把整条链切成 `num_nodes` 个连续 block，每个 block 分给一个节点单独计算。
3. 为每个 block 和每个 cap 生成输入文件、节点本地运行脚本、顶层提交脚本和结果汇总脚本。

现在默认推荐的运行方式是：

- 每个节点只跑自己的 `block_xxxx/run_local.sh`
- 顶层用 `submit_per_node_local.sh` 通过 `ssh` 并发触发
- 所有 block 结束后，再单独计算 `cap_*`
- 最后用 `combine_results.py` 汇总能量

这比“首节点一个总 `mpirun` 扇出到所有节点”的方式更接近真实弱扩展结构。

## 2. 运行前准备

需要准备以下内容：

- Python 环境，且安装了 `numpy`
- 仓库根目录下的 `gen.py`，用于生成聚乙烯链
- `pseudo-dir`，目录里至少有 `C.psf` 和 `H.psf`
- 计算节点可访问的共享目录
- 节点之间已配置免密 `ssh`
- 节点上可用的 SIESTA、MPI、UCX、XPMEM 环境

脚本会自动生成一个 `honpas_env.sh`，把当前约定的 HONPAS 环境变量写进去。如果你们内网真实路径不同，可以通过命令行参数覆盖。

`weak_scale_pe.py` 现在默认使用仓库根目录下的 [gen.py](/home/xzz2/huawei-siesta-energy-first/Vayesta/gen.py)。只有当你想改用别的 PE 生成脚本时，才需要显式传 `--gen-script`。

## 3. 如何指定每个节点算多少原子

用参数 `--atoms-per-node`。

例如：

```bash
python3 weak_scale_pe.py \
  --atoms-per-node 5000 \
  --num-nodes 4 \
  --pseudo-dir /path/to/pseudos \
  --out-dir /share/weak_scale/ws4
```

这里的含义是：

- 目标是每个节点大约 5000 个原子
- 一共 4 个节点
- 因此总体系目标规模大约是 `5000 * 4 = 20000` 个原子

注意：这是“目标值”，不是绝对精确值。

原因是 PE 的总原子数满足：

```text
C_n H_(2n+2)
总原子数 = 3n + 2
```

所以脚本会选一个最接近 `atoms_per_node * num_nodes` 的合法 PE 规模，再按连续碳链近似均分到各节点。

实际每个节点分到多少原子，可以看：

- 生成时终端输出
- 生成后的 `schedule.json`

## 4. 常用命令

### 4.1 生成弱扩展测试目录

```bash
python3 weak_scale_pe.py \
  --atoms-per-node 5000 \
  --num-nodes 4 \
  --procs-per-node 16 \
  --pseudo-dir /share/honpas/xzz/siesta-20260520/testcase \
  --out-dir /share/honpas/xzz/ws/ws4 \
  --remote-out-dir /share/honpas/xzz/ws/ws4 \
  --ssh-user xzz
```

几个关键参数：

- `--atoms-per-node`：每节点目标原子数
- `--num-nodes`：节点数，也等于 block 数
- `--procs-per-node`：每个节点的 MPI rank 数
- `--gen-script`：可选，覆盖默认的仓库内 `gen.py`
- `--pseudo-dir`：赝势目录
- `--out-dir`：当前机器写出的目录
- `--remote-out-dir`：远端节点实际看到的共享目录路径
- `--ssh-user`：可选，指定 `ssh` 登录用户名
- `--block-slice-num`：写进每个 block（及 full 基线）`run_local.sh` 的 `NTPOLY_SLICE_NUM`，默认 1
- `--dimer-slice-num`：写进每个 dimer `run_local.sh` 的 `NTPOLY_SLICE_NUM`，默认 1

### 4.2 如果要改用别的 PE 生成脚本

例如切回内网原始 `gen.py`：

```bash
  --gen-script /share/honpas/xzz/siesta-20260520/testcase/gen.py
```

### 4.3 如果 HONPAS 路径不是默认值

可以补这些参数：

```bash
  --pkg-root /share/honpas/xxx/siesta-20260520 \
  --mpi-prefix /share/hmpi2.4.1/hmpi-v2.4.1-huawei \
  --siesta-app /share/honpas/xxx/siesta/build-clang/Src/siesta \
  --env-sh /share/honpas/xxx/siesta-20260520/env.sh
```

### 4.4 如果节点列表不是默认 16 台

默认 `--hosts` 内置 16 个节点（`71.20.27.21-36` 和 `71.20.16.{12,22,32,42,132,142,152,162}`）。要换一批时：

```bash
  --hosts 71.20.27.21 71.20.27.22 71.20.27.23 71.20.27.24
```

`--num-nodes` 不能大于 `--hosts` 提供的机器数量。

### 4.5 NTPOLY 切片数（`--block-slice-num` / `--dimer-slice-num`）

`NTPOLY_SLICE_NUM` 控制密度矩阵纯化（TRS2）把矩阵切成几片，目的是让单片工作集放进 HBM。脚本在生成时用**两个变量分别描述 block 和 dimer 这两个过程**：

- `--block-slice-num N`：写进每个 block（以及 full 基线）`run_local.sh` 的 `NTPOLY_SLICE_NUM`
- `--dimer-slice-num N`：写进每个 dimer `run_local.sh` 的 `NTPOLY_SLICE_NUM`（dimer ≈ 2× block，同样每节点原子数下通常需要比 block 更大的切片数）

两者**不指定时默认都是 1**，且必须 ≥1。所选值会写进 `schedule.json` 的 `ntpoly_slice_num` 字段，并在生成时打印一行 `NTPOLY_SLICE_NUM -> block=.., dimer=..`。cap 是 H₂ 小任务、用 diagonali，固定为 1，不受这两个参数影响。

例如大体系给 block 用 2、dimer 用 3：

```bash
python3 weak_scale_pe.py ... --block-slice-num 2 --dimer-slice-num 3
```

具体取值取决于每节点 HBM 容量；建议借助 SIESTA 内置计时器（`UseTreeTimer` 等，已在生成的 FDF 中默认开启）实测求解时间和内存占用后再定档，切片过多反而因通信开销变慢。

## 5. 生成后目录里有什么

假设输出目录是 `/share/honpas/xzz/ws/ws4`，会生成：

- `schedule.json`
- `honpas_env.sh`
- `submit_per_node_local.sh`
- `launch_head_mpirun.sh`
- `combine_results.py`
- `rankfile_template.txt`
- `block_0000/ ... block_000N/`
- `cap_0000/ ... cap_000M/`

其中：

- `submit_per_node_local.sh` 是推荐方式
- `launch_head_mpirun.sh` 是保留的旧方案，只建议做对照

每个 `block_xxxx` 目录里会有：

- `input.fdf`
- `C.psf` / `H.psf`
- `run_local.sh`

每个 `cap_xxxx` 目录里也会有对应的：

- `input.fdf`
- `C.psf` / `H.psf`（实际上 cap 只会用到 H）
- `run_local.sh`

## 6. 推荐运行流程

### 6.1 先生成输入

见上面的生成命令。

### 6.2 再执行推荐提交脚本

进入输出目录：

```bash
cd /share/honpas/xzz/ws/ws4
bash ./submit_per_node_local.sh
```

这个脚本会：

1. 通过 `ssh` 登录每个 host
2. 在对应的 `block_xxxx` 目录里执行 `bash ./run_local.sh`
3. 等所有 block 结束
4. 在 `CAP_HOST` 上顺序跑所有 `cap_*`
5. 最后本地执行 `python3 combine_results.py`

如果你想指定 cap 在哪台机器上跑：

```bash
CAP_HOST=71.20.27.21 bash ./submit_per_node_local.sh
```

如果远端共享目录和生成时不一致，也可以临时覆盖：

```bash
REMOTE_OUT_DIR=/share/honpas/xzz/ws/ws4 bash ./submit_per_node_local.sh
```

## 7. 如何看结果

汇总完成后会生成：

- `weak_scaling_results.json`

主要字段含义：

- `E_mfcc_ev`：MFCC 组合后的总能量
- `missing_outputs`：没成功提取能量的输出文件

时间信息请直接看各目录里的 `siesta.out`。当前脚本不再额外包一层 `time`，避免不同节点 shell 实现不一致。

## 8. 建议的测试方式

建议固定 `atoms_per_node`，逐步扩大 `num_nodes`：

```text
atoms_per_node = 5000
num_nodes = 1, 2, 4, 8
```

例如：

```bash
python3 weak_scale_pe.py ... --atoms-per-node 5000 --num-nodes 1 --out-dir /share/.../ws1
python3 weak_scale_pe.py ... --atoms-per-node 5000 --num-nodes 2 --out-dir /share/.../ws2
python3 weak_scale_pe.py ... --atoms-per-node 5000 --num-nodes 4 --out-dir /share/.../ws4
python3 weak_scale_pe.py ... --atoms-per-node 5000 --num-nodes 8 --out-dir /share/.../ws8
```

然后分别运行各自目录下的 `submit_per_node_local.sh`，比较每组的：

- `E_mfcc_ev`
- `missing_outputs`

## 9. 当前方案和旧方案的区别

旧方案：

- 从首节点起一个总 `mpirun`
- 用一个大 rankfile 把不同 block 扇到不同节点

当前推荐方案：

- 每个节点本地只起自己的 `mpirun`
- 顶层只负责 `ssh` 并发触发
- 更接近“每台机器独立求自己的局部 block”的实际弱扩展结构

## 10. 注意事项

- `--atoms-per-node` 是近似目标，不保证每个 block 原子数完全一致。
- `--remote-out-dir` 最好显式指定成所有节点都能看到的共享路径。
- 运行前先确认 `honpas_env.sh` 里写入的路径和你们内网实际环境一致。
- 当前脚本默认 block 用 `ntpoly`，cap 用 `diagonali`。
- 当前汇总脚本只提取能量；运行时间请从各自的 `siesta.out` 中读取。
- `submit_per_node_local.sh` 依赖免密 `ssh`。
- 如果某个节点输出里没有 `siesta: ... Total = ...`，`combine_results.py` 会把它列到 `missing_outputs`。

## 11. 最小示例

```bash
python3 weak_scale_pe.py \
  --atoms-per-node 5000 \
  --num-nodes 4 \
  --hosts 71.20.27.21 71.20.27.22 71.20.27.23 71.20.27.24 \
  --pseudo-dir /share/honpas/xzz/siesta-20260520/testcase \
  --out-dir /share/honpas/xzz/ws/ws4 \
  --remote-out-dir /share/honpas/xzz/ws/ws4 \
  --ssh-user xzz

cd /share/honpas/xzz/ws/ws4
bash ./submit_per_node_local.sh
```

运行结束后查看：

```bash
cat /share/honpas/xzz/ws/ws4/weak_scaling_results.json
```
