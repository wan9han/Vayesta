#!/usr/bin/env python3
"""E级 combine 时间模拟：fabricate N 节点的 block/cap/dimer 能量，实测
MFCC(1)+MBE(2) 求和时间（O(N) 求和，数据为编造，仅测时间）。

纯标准库，无 numpy/外部依赖、无硬编码路径。默认 N=22680，可用 --nodes 改。
注：这是"纯求和"的算法时间模拟（不计文件读写）；如要测真实文件读取代价，
用 benchmark/bench_combine.py（复制真实 siesta.out + 跑真实 combine_results.py）。
"""
import argparse
import random
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=22680)
    args = ap.parse_args()
    N = args.nodes
    rng = random.Random(0)
    print(f"=== combine 模拟：{N} 节点（{N} blocks + {N-1} caps + {N-1} dimers）===")
    t0 = time.perf_counter()
    blocks = [rng.uniform(-61600, -61500) for _ in range(N)]
    caps = [-28.767449] * (N - 1)
    dimers = [rng.uniform(-123200, -123000) for _ in range(N - 1)]
    t1 = time.perf_counter()
    e_mfcc = sum(blocks) - sum(caps)
    e_mbe2 = e_mfcc
    for k in range(N - 1):
        e_mbe2 += dimers[k] - blocks[k] - blocks[k + 1] + caps[k]
    t2 = time.perf_counter()
    print(f"  fabricate data : {t1-t0:.3f} s")
    print(f"  MFCC(1)+MBE(2) sum : {t2-t1:.3f} s   <- combine compute time")
    print(f"  E_total = {e_mbe2:,.1f} eV (fabricated)")
    print(f"\n结论：{N} 节点纯求和耗时 {t2-t1:.3f} s（O(N)，不计文件 IO）。")


if __name__ == "__main__":
    main()
