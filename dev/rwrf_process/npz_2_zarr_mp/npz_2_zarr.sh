#!/bin/bash
UTIL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # parent of des_npz_2_zarr

PYTHONPATH="$UTIL_DIR:$PYTHONPATH" python npz_2_zarr.py \
    --invariants "lsm,orog" \
    --lon-min 121 --lon-max 125 --lat-min 21 --lat-max 25 \
    --domain-size 256,256 \
    --cache-highres /workspace/NCDR_StormCast/des/npz/rwrf \
    --cache-lowres  /workspace/NCDR_StormCast/des/npz/era5 \
    --zarr-base /workspace/NCDR_StormCast/des/zarr \
    --experiment-name stormcast_test \
    --train-ranges "2019/08/06:2019/08/06" \
    --valid-ranges "2019/09/01:2019/09/02" \
    --split train \
    --log-level DEBUG --log-file /workspace/physicsnemo/dev/rwrf_process/des_npz_2_zarr/npz_2_zarr.log \