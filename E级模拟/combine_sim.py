#!/usr/bin/env python3
"""E级汇总时间模拟：fabricate 22,680 节点的 block/cap/dimer 能量，实测 combine_results
的 MFCC(1)+MBE(2) 求和时间（O(N) 求和，数据为编造，仅测时间）。"""
import sys, time
sys.path.insert(0, "/tmp/detopt_venv/lib/python3.10/site-packages")
import numpy as np
N=22680
print(f"=== combine 模拟：{N} 节点（{N} blocks + {N-1} caps + {N-1} dimers）===")
rng=np.random.default_rng(0)
t0=time.perf_counter()
blocks=rng.uniform(-61600,-61500,N)          # 编造 block 能量
caps=np.full(N-1,-28.767449)                 # H2 cap
dimers=rng.uniform(-123200,-123000,N-1)      # 编造 dimer 能量 (~2×block)
t1=time.perf_counter()
# combine_results 的核心计算
e_mfcc=float(blocks.sum()-caps.sum())
e_mbe2=e_mfcc
for k in range(N-1):
    e_mbe2 += dimers[k]-blocks[k]-blocks[k+1]+caps[k]
t2=time.perf_counter()
print(f"  fabricate data : {t1-t0:.3f} s")
print(f"  MFCC(1)+MBE(2) 求和 : {t2-t1:.3f} s   <- 汇总流程时间")
print(f"  E_total = {e_mbe2:,.1f} eV (编造数据)")
print(f"\n结论：{N} 节点汇总耗时 {t2-t1:.3f} s，可忽略（O(N) 求和），不构成 E 级瓶颈。")
