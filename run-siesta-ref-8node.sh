#!/bin/bash
set -euo pipefail
set -x


# ===== 基本参数 =====
export p_perdie=36          # 每个 NUMA 域用于计算的核心数（38 核里前 2 核保留）
export ndie=16              # 每节点 NUMA 数
export nnode=8
host1=${1:-71.20.27.21}
host2=${2:-71.20.27.22}
host3=${3:-71.20.27.23}
host4=${4:-71.20.27.24}
host5=${5:-71.20.27.33}
host6=${6:-71.20.27.34}
host7=${7:-71.20.27.35}
host8=${8:-71.20.27.36}
host_config="${host1}:16,${host2}:16,${host3}:16,${host4}:16,${host5}:16,${host6}:16,${host7}:16,${host8}:16"

# ===== 机器常量（与参考脚本保持一致） =====
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export CORE_PER_NUMA=38
export CORE_BIAS=2
export NUMA_BIAS=0
export NUM_NUMAS=16
export NET_DEVICE_ALL="hns_0:1,hns_1:1,hns_3:1,hns_2:1,hns_4:1,hns_5:1,hns_7:1,hns_6:1"

# ===== 应用/目录 =====
PKG_ROOT=/share/honpas/xzz/siesta-20260520
CASE_DIR=${PKG_ROOT}/testcase
APP=${PKG_ROOT}/siesta/build-clang/Src/siesta
ENV_SH=${PKG_ROOT}/env.sh
INPUT=8000.fdf
RANKFILE=./rankfile
LOG=siesta.pe.${INPUT}.8node.log
MPI_PREFIX=/share/hmpi2.4.1/hmpi-v2.4.1-huawei/
MPIRUN=${MPI_PREFIX}/bin/mpirun

# ===== 应用特定环境 =====
cd "${CASE_DIR}"
source "${ENV_SH}"

XPMEM_HOME=/share/hmpi2.4.1/xpmem-2.7.3
UCX_HOME=/share/hmpi2.4.1/hucx-v2.4.1-huawei
UCG_HOME=/share/hmpi2.4.1/xucg-v2.4.1-huawei
MPI_HOME=/share/hmpi2.4.1/hmpi-v2.4.1-huawei
BISHENG_LIB="/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/lib:/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/lib64"
BISHENG_PATH="/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin"

# 强化包内环境，避免依赖任何节点的 ~/.bashrc
export OPAL_PREFIX="${MPI_PREFIX}"
export LD_LIBRARY_PATH="${MPI_HOME}/lib:${UCG_HOME}/lib:${UCX_HOME}/lib:${XPMEM_HOME}/lib:${BISHENG_LIB}:${LD_LIBRARY_PATH}"
export LIBRARY_PATH="${MPI_HOME}/lib:${UCG_HOME}/lib:${UCX_HOME}/lib:${XPMEM_HOME}/lib:${BISHENG_LIB}:${LIBRARY_PATH}"
export PATH="${MPI_HOME}/bin:${UCG_HOME}/bin:${UCX_HOME}/bin:${XPMEM_HOME}/bin:${BISHENG_PATH}:${PATH}"

export OMPI_CC=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/clang
export OMPI_CXX=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/clang++
export OMPI_FC=/share/honpas/xzz/siesta-20260520/BiShengCompiler-4.2.0.2-aarch64-linux/bin/flang

export OMP_NUM_THREADS=36
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export NTPOLY_SLICE_NUM=1

export MEMKIND_HBW_NODES=16-31
export VERBS_LOG_LEVEL=0
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,sm

ulimit -s unlimited

# ===== 参考脚本风格：保留 bind 参数生成，仅用于打印/核对 =====
generate_bind_param() {
    local p_perdie=$1
    local ndie=$2
    local bind_param=""

    for (( die=0; die<ndie; die++ )); do
        local start_core=$(( die * CORE_PER_NUMA + CORE_BIAS ))
        local end_core=$(( start_core + p_perdie - 1 ))
        bind_param+="${start_core}-${end_core}"
        if (( die < ndie - 1 )); then
            bind_param+="," 
        fi
    done

    echo "${bind_param}"
}

# ===== 自动生成 2 节点 rankfile 到 testcase 目录 =====
generate_rankfile() {
    local file=$1
    local h1=$2
    local h2=$3
    : > "${file}"

    for (( r=0; r<16; r++ )); do
        local start=$(( r * CORE_PER_NUMA + CORE_BIAS ))
        local end=$(( start + p_perdie - 1 ))
        echo "rank ${r}=${h1} slots=${start}-${end}" >> "${file}"
    done

    for (( r=0; r<16; r++ )); do
        local grank=$(( r + 16 ))
        local start=$(( r * CORE_PER_NUMA + CORE_BIAS ))
        local end=$(( start + p_perdie - 1 ))
        echo "rank ${grank}=${h2} slots=${start}-${end}" >> "${file}"
    done
}

generate_rankfile "${RANKFILE}" "${host1}" "${host2}" "${host3}" "${host4}" "${host5}" "${host6}" "${host7}" "${host8}"
bind_param=$(generate_bind_param "${p_perdie}" "${ndie}")

echo "[INFO] host1=${host1}"
echo "[INFO] host2=${host2}"
echo "[INFO] host3=${host3}"
echo "[INFO] host4=${host4}"
echo "[INFO] host5=${host5}"
echo "[INFO] host6=${host6}"
echo "[INFO] host7=${host7}"
echo "[INFO] host8=${host8}"
echo "[INFO] host_config=${host_config}"
echo "[INFO] bind_param=${bind_param}"
echo "[INFO] APP=${APP}"
echo "[INFO] CASE_DIR=${CASE_DIR}"
echo "[INFO] RANKFILE=${RANKFILE}"
echo "[INFO] MPI_PREFIX=${MPI_PREFIX}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"


sync
echo 3 > /proc/sys/vm/drop_caches


# 可选：动态库检查。默认关闭，避免环境敏感时 ldd 触发异常。
if [[ "${RUN_LDD_CHECK:-0}" == "1" ]]; then
    ldd "${APP}" | tee "ldd.siesta-m6.txt"
fi

# 正式执行
"${MPIRUN}" --allow-run-as-root \
  --prefix "${MPI_PREFIX}" \
  -host "${host_config}" \
  -wdir "${CASE_DIR}" \
  -np 128 \
  -x PATH -x LD_LIBRARY_PATH -x OPAL_PREFIX \
  -x NUMA_BIAS -x CORE_BIAS -x p_perdie -x ndie -x nnode \
  -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
  -x NTPOLY_SLICE_NUM -x MEMKIND_HBW_NODES \
  -x VERBS_LOG_LEVEL -x UCX_LOG_LEVEL -x UCX_TLS=${UCX_TLS} \
  -x NUM_NUMAS=16 -x NODES=8 -x RANKS_PER_NODE=16 \
  -x NET_DEVICE_ALL \
  -x UCX_RC_VERBS_ROCE_LOCAL_SUBNET=y -x UCX_UD_VERBS_ROCE_LOCAL_SUBNET=y \
  --rankfile "${RANKFILE}" --mca rmaps_rank_file_physical true \
  --mca coll ^ucg --mca pml ucx --mca btl ^vader,tcp,openib,uct,ofi,usnic \
  -x UCX_UD_VERBS_ALLOC=thp,md,mmap,heap -x UCX_RC_VERBS_ALLOC=thp,md,mmap,heap -x UCX_RC_VERBS_TX_MIN_SGE=2 \
  --mca orte_launch_agent "$PKG_ROOT/launch_orted.sh" \
  "${APP}" "${INPUT}" |& tee "${LOG}"
