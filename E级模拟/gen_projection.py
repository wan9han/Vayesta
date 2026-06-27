#!/usr/bin/env python3
"""E级生成/汇总时间模拟：实测小规模 h_to_c（旧 brute vs 新 cKDTree）与 combine，
拟合标度后外推到 22,680 节点（8000 原子/节点，n_c≈6.05e7）。"""
import sys, subprocess, time
sys.path.insert(0, "/home/xzz2/huawei-siesta-energy-debug/Vayesta")
sys.path.insert(0, "/tmp/detopt_venv/lib/python3.10/site-packages")
import numpy as np
from scipy.spatial import cKDTree
from energy_first.molecule import parse_gen_fdf_text
GEN="/home/xzz2/huawei-siesta/testcases/gen.py"

def gen_mol(n_c):
    env={"PYTHONPATH":"/tmp/detopt_venv/lib/python3.10/site-packages","PATH":"/usr/bin:/bin"}; p=subprocess.run(["python3",GEN,str(n_c)],capture_output=True,text=True,check=True,env=env)
    return parse_gen_fdf_text(p.stdout,label=f"C{n_c}")

def setup(mol):
    coords=mol.coords
    cs=[i for i,e in enumerate(mol.elements) if e=="C"]
    cs.sort(key=lambda i:(coords[i,0],coords[i,1],coords[i,2]))
    return coords, coords[cs], cs, [i for i,e in enumerate(mol.elements) if e=="H"]

def brute(coords,cpos,cs,h_idx):
    return {h: cs[int(np.argmin(np.linalg.norm(cpos-coords[h],axis=1)))] for h in h_idx}

def kdtree(coords,cpos,cs,h_idx):
    if not h_idx: return {}
    tree=cKDTree(cpos); _,nn=tree.query(coords[h_idx],k=1); nn=np.atleast_1d(nn)
    return {h: cs[int(j)] for h,j in zip(h_idx,nn)}

print("=== 实测：h_to_c（旧 brute / 新 cKDTree）===")
print(f"{'n_c':>8} {'atoms':>8} {'brute(s)':>10} {'kdtree(s)':>10}")
brute_n=[]; brute_t=[]; kd_n=[]; kd_t=[]
for n_c, do_brute in [(2000,True),(4000,True),(8000,True),(16000,True),
                      (32000,False),(100000,False)]:
    mol=gen_mol(n_c); coords,cpos,cs,h_idx=setup(mol)
    if do_brute:
        t0=time.perf_counter(); brute(coords,cpos,cs,h_idx); tb=time.perf_counter()-t0
        brute_n.append(n_c); brute_t.append(tb)
    else: tb=float('nan')
    t0=time.perf_counter(); kdtree(coords,cpos,cs,h_idx); tk=time.perf_counter()-t0
    kd_n.append(n_c); kd_t.append(tk)
    print(f"{n_c:>8} {mol.natoms:>8} {tb:>10.3f} {tk:>10.4f}")

# 拟合：brute t=a*n^2 ; kdtree t=b*n*log2(n)
a=np.mean([t/n**2 for n,t in zip(brute_n,brute_t)])
b=np.mean([t/(n*np.log2(n)) for n,t in zip(kd_n,kd_t)])
NC_E=round((22680*8000-2)/3)   # 22680 节点 × 8000 原子 -> n_c
t_brute_E=a*NC_E**2
t_kd_E=b*NC_E*np.log2(NC_E)
print(f"\n=== 外推到 22,680 节点（n_c={NC_E:,}, {3*NC_E+2:,} 原子）===")
print(f"  h_to_c 旧 (brute, O(N^2)):   {t_brute_E:,.0f} s  =  {t_brute_E/86400:,.1f} 天")
print(f"  h_to_c 新 (cKDTree, O(NlogN)): {t_kd_E:,.1f} s  =  {t_kd_E/60:.1f} 分钟")
print(f"  提速比: {t_brute_E/t_kd_E:,.0f} ×")
