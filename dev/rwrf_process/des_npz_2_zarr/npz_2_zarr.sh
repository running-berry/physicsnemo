#!/bin/bash
UTIL_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # parent of des_npz_2_zarr

PYTHONPATH="$UTIL_DIR:$PYTHONPATH" python npz_2_zarr.py \
    --var-lowres "u10,v10,t2m" \
    --var-highres "u10,v10,t2m,qpepre" \
    --var-dummy "t2m" \
    --invariants "lsm,orog" \
    --lon-min 121 --lon-max 125 --lat-min 21 --lat-max 25 \
    --domain-size 256,256 \
    --cache-highres /workspace/NCDR_StormCast/des/npz/rwrf \
    --cache-lowres  /workspace/NCDR_StormCast/des/npz/era5 \
    --zarr-base /workspace/NCDR_StormCast/des/zarr \
    --experiment-name stormcast_small \
    --train-ranges "2019/08/01:2019/08/31" \
    --valid-ranges "2019/09/01:2019/09/07" \
    --split train \