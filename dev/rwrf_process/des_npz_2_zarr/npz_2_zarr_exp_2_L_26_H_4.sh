#!/bin/bash
UTIL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # parent of des_npz_2_zarr

PYTHONPATH="$UTIL_DIR:$PYTHONPATH" python npz_2_zarr_ori.py \
    --invariants "lsm,orog" \
    --lon-min 120 --lon-max 122.5 --lat-min 21.5 --lat-max 25.5 \
    --domain-size 224,128 \
    --var-highres "t2m,u10,v10,qpepre" \
    --cache-highres /workspace/NCDR_StormCast/des/npz/rwrf \
    --cache-lowres  /workspace/NCDR_StormCast/des/npz/era5 \
    --zarr-base /workspace/NCDR_StormCast/des/zarr_exp2_L_26_H_24 \
    --experiment-name stormcast_test \
    --train-ranges "2019/08/01:2020/08/31" \
    --valid-ranges "2020/09/01:2021/08/31" \
    --split both \
    --log-level DEBUG --log-file /workspace/physicsnemo/dev/rwrf_process/des_npz_2_zarrzarr_exp2_L_26_H_24.log \