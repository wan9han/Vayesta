import argparse

import numpy as np

def generate_polyethylene(n_carbons, cc_bond_length=1.53, ch_bond_length=1.10):
    """
    生成聚乙烯 (CnH2n+2) 分子的坐标。
    
    参数:
    n_carbons (int): 碳原子数量
    cc_bond_length (float): C-C 键长 (埃)
    ch_bond_length (float): C-H 键长 (埃)
    
    返回:
    atoms (list): 包含原子信息的列表 [{'element': 'C', 'x':..}, ...]
    """
    
    # 基础几何参数 (四面体角度)
    theta = 109.471 * np.pi / 180.0  # 四面体角弧度
    phi = theta / 2.0
    
    # 碳骨架增量 (Zigzag pattern along X-axis)
    # dx = d * sin(theta/2)
    # dy = d * cos(theta/2)
    dx = cc_bond_length * np.sin(phi)
    dy = cc_bond_length * np.cos(phi)
    
    atoms = []
    
    # --- 1. 生成碳原子骨架 ---
    carbon_coords = []
    for i in range(n_carbons):
        # X 坐标线性增加
        # 为了让中心在原点附近，可以减去一半长度，这里简化为从0开始或以第一个键中心为原点
        # 参照您的文件，两个C关于原点对称。
        # 这里我们先生成，最后统一平移中心到 (0,0,0)
        
        x = i * dx
        # Y 坐标交替: 0, dy, 0, dy... 或者 -dy/2, dy/2...
        # 您的文件规律：偶数位 y = -0.428, 奇数位 y = +0.428
        # y_offset = dy / 2
        y = -dy/2.0 if (i % 2 == 0) else dy/2.0
        z = 0.0
        
        carbon_coords.append([x, y, z])
        atoms.append({'species': 1, 'label': 'C', 'x': x, 'y': y, 'z': z})

    # --- 2. 生成侧链氢原子 ---
    # 侧链氢原子位于垂直于 C-C 键走向的平面内 (YZ平面投影)
    # 几何计算：
    # H 的 x 坐标与 C 相同
    # H 的 y, z 坐标由 C-H 键长和角度决定
    
    # C-H 向量在 YZ 平面的投影长度
    # 考虑到 H-C-H 平面与 C-C-C 平面垂直，且平分角相同
    # 简化的几何构建：
    # 两个 H 的矢量结合应该平衡掉 C-C 键产生的 Y 方向分量
    
    # H_dy 计算: 
    # 投影在 YZ 平面上，H-C-H 夹角在投影上不是 109.5，而是需要计算。
    # 但更简单的方法是利用四面体顶点的矢量方向。
    # 对于 sp3 碳，四个键指向正四面体顶点。
    
    # 预计算 H 的偏移量
    # 偶数碳 (Y负): 两个 C-C 键分别指向 (+dx, +dy, 0) 和 (-dx, +dy, 0) [相对方向]
    # 另外两个 C-H 键应该指向 Y 负方向和 Z 正负方向
    
    # 计算 C-H 在 Y 和 Z 方向的分量
    # 几何推导结果：
    # hy = ch_bond_length * sin(phi) = ch_length * 0.816
    # hz = ch_bond_length * cos(phi) ? No.
    
    # 直接利用矢量几何:
    # 键角的一半是 phi = 54.74度
    # H 的 y 分量 (向外延伸): dy_h = ch_bond * sin(theta_complex)
    # 您的文件数据: C_y = 0.428, H_y = 1.060 -> diff = 0.632
    # C_z = 0.000, H_z = 0.863
    # L = sqrt(0.632^2 + 0.863^2) = 1.07 (C-H键长)
    # tan(beta) = 0.863 / 0.632
    
    # 通用比例因子
    scale_factor = cc_bond_length / 1.53  # 基于您文件的基准
    h_scale_factor = ch_bond_length / 1.10
    
    # 提取自文件 0008.fdf 的标准化几何矢量 (单位化后再乘以键长)
    # 基础偏移量 (基于 ch_bond_length = 1.10 Å)
    # Z轴偏移
    hz_base = 0.863 * (ch_bond_length / 1.07) 
    # Y轴偏移 (相对于碳原子的增量)
    hy_base = 0.632 * (ch_bond_length / 1.07)
    
    for i in range(n_carbons):
        cx, cy, cz = carbon_coords[i]
        
        # 确定 H 原子向哪个 Y 方向延伸
        # 如果 C 在下方 (y < 0)，H 朝更下方延伸 (y 减小)
        # 如果 C 在上方 (y > 0)，H 朝更上方延伸 (y 增加)
        sign = 1.0 if cy > 0 else -1.0
        
        # 添加两个侧链 H (一个 z+, 一个 z-)
        atoms.append({'species': 2, 'label': 'H', 'x': cx, 'y': cy + sign * hy_base, 'z': hz_base})
        atoms.append({'species': 2, 'label': 'H', 'x': cx, 'y': cy + sign * hy_base, 'z': -hz_base})

    # --- 3. 生成端基封端氢原子 (End Caps) ---
    # 位于链的两端，补全 sp3
    # 起始端 (i=0): 沿着 C0-C1 的反向延长线，但在 XY 平面内
    # C0 坐标 (0, -y, 0), C1 坐标 (dx, y, 0) -> Vector C1->C0 = (-dx, -2y, 0)
    # 封端 H 应该位于 C0-C1 键的反向，保持键角。
    # 您的文件中，端基 H 位于 Z=0 平面。
    # 坐标参考: C1(-0.634, -0.428), 端基 H(-1.500, 0.201)
    # dX = -0.866, dY = +0.629. L = 1.07.
    
    # 起始端 H (连接到 atoms[0])
    c0 = carbon_coords[0]
    c1 = carbon_coords[1]
    # 简易算法：利用对称性，端基 H 相当于“上一个虚拟碳”的位置，但缩短为 C-H 键长
    # 虚拟 C(-1) 的方向矢量
    v_start_x = c0[0] - dx
    v_start_y = -c0[1] # 翻转 Y
    
    # 归一化方向向量并乘以 CH 键长
    vec_x = v_start_x - c0[0]
    vec_y = v_start_y - c0[1]
    norm = np.sqrt(vec_x**2 + vec_y**2)
    h_start_x = c0[0] + (vec_x / norm) * ch_bond_length
    h_start_y = c0[1] + (vec_y / norm) * ch_bond_length
    
    atoms.append({'species': 2, 'label': 'H', 'x': h_start_x, 'y': h_start_y, 'z': 0.0})
    
    # 终止端 H (连接到 atoms[-1])
    cn = carbon_coords[-1]
    cn_prev = carbon_coords[-2]
    # 虚拟 C(n+1) 的方向
    v_end_x = cn[0] + dx
    v_end_y = -cn[1] # 翻转 Y
    
    vec_x = v_end_x - cn[0]
    vec_y = v_end_y - cn[1]
    norm = np.sqrt(vec_x**2 + vec_y**2)
    h_end_x = cn[0] + (vec_x / norm) * ch_bond_length
    h_end_y = cn[1] + (vec_y / norm) * ch_bond_length
    
    atoms.append({'species': 2, 'label': 'H', 'x': h_end_x, 'y': h_end_y, 'z': 0.0})

    # --- 4. 坐标中心化 (可选) ---
    # 将几何中心移到 (0,0,0)
    xs = [a['x'] for a in atoms]
    ys = [a['y'] for a in atoms]
    zs = [a['z'] for a in atoms]
    center_x = (min(xs) + max(xs)) / 2.0
    center_y = (min(ys) + max(ys)) / 2.0
    center_z = (min(zs) + max(zs)) / 2.0
    
    for a in atoms:
        a['x'] -= center_x
        a['y'] -= center_y
        a['z'] -= center_z
        
    return atoms

def print_fdf_format(atoms, system_label="Polyethylene"):
    """输出为 SIESTA .fdf 格式"""
    atoms = order_polyethylene_for_contiguous_blocks(atoms)
    n_atoms = len(atoms)
    
    print(f"SystemLabel      {system_label}")
    print(f"NumberOfAtoms    {n_atoms}")
    print(f"NumberOfSpecies  2")
    print("%block ChemicalSpeciesLabel")
    print("    1    6  C")
    print("    2    1  H")
    print("%endblock ChemicalSpeciesLabel")
    print("")
    print("AtomicCoordinatesFormat NotScaledCartesianAng")
    print("%block AtomicCoordinatesAndAtomicSpecies")
    
    # 格式化打印：X Y Z Species_ID
    # 对齐格式
    for a in atoms:
        print(f"    {a['x']:12.6f}    {a['y']:12.6f}    {a['z']:12.6f}    {a['species']}")
        
    print("%endblock AtomicCoordinatesAndAtomicSpecies")

    print("""
PAO.BasisType    split
PAO.BasisSize      SZ
SolutionMethod     ELSI
PAO.SplitNorm    0.150000
PAO.EnergyShift    0.020000  Ry
Harris_functional    false
XC.functional    LDA
XC.Authors    PZ
SpinPolarized    false
MeshCutoff    100.000000 Ry
kgrid_cutoff    0.000000 Bohr
ElectronicTemperature    300.000000 K
MaxSCFIterations    50
DM.NumberPulay    0
DM.MixingWeight    0.250000

UseSaveData		false
WriteCoorInital    true
WriteCoorStep    false
WriteForces		false
WriteKpoints		false
WriteEigenvalues		false
WriteKbands		false
WriteBands		false
WriteWaveFunctions		false
WriteMullikenPop		0
WriteDM		true
SaveHS		true
WriteOrbitalIndex	true
WriteCoorXmol		false
WriteCoorCerius		false
WriteMDXmol		false
WriteMDhistory		true
""")

def order_polyethylene_for_contiguous_blocks(atoms):
    """按链段输出原子，避免连续分块时把 C/H 大范围拆开。"""
    n_atoms = len(atoms)
    n_carbons = (n_atoms - 2) // 3
    if n_carbons < 2 or 3 * n_carbons + 2 != n_atoms:
        return atoms

    carbons = atoms[:n_carbons]
    side_hydrogens = atoms[n_carbons : 3 * n_carbons]
    end_hydrogens = atoms[3 * n_carbons :]
    ordered = []
    for carbon_start in range(0, n_carbons, 2):
        carbon_end = min(carbon_start + 2, n_carbons)
        ordered.extend(carbons[carbon_start:carbon_end])
        ordered.extend(side_hydrogens[2 * carbon_start : 2 * carbon_end])
        if carbon_start == 0:
            ordered.append(end_hydrogens[0])
        if carbon_end == n_carbons:
            ordered.append(end_hydrogens[1])
    return ordered

def main():
    parser = argparse.ArgumentParser(description="Generate a polyethylene SIESTA FDF input.")
    parser.add_argument("num_carbons", nargs="?", type=int, default=2300)
    parser.add_argument("--cc-bond", type=float, default=1.54)
    parser.add_argument("--ch-bond", type=float, default=1.09)
    args = parser.parse_args()

    if args.num_carbons < 2:
        raise SystemExit("num_carbons must be at least 2")

    molecule = generate_polyethylene(args.num_carbons, args.cc_bond, args.ch_bond)
    print_fdf_format(molecule, system_label=f"PE_{args.num_carbons}C")


if __name__ == "__main__":
    main()
