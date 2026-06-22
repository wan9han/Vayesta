#!/bin/bash

set -euo pipefail

PKG=/share/honpas/xzz/siesta-20260520
source "$PKG/env.sh"

rmmod -f sdma-dae
insmod /usr/lib/modules/5.10.0/kernel/drivers/misc/sdma-dae/sdma_dae.ko share_chns=160 safe_mode=0

exec "/share/hmpi2.4.1/hmpi-v2.4.1-huawei/bin/orted" "$@"