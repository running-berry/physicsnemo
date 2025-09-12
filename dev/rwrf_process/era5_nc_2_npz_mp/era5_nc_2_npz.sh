#!/bin/bash

python3 era5_nc_2_npz.py \
    --input-dir /workspace/NCDR_StormCast/des/era5_rwrf_nc/era5 \
    --output-dir /workspace/NCDR_StormCast/des/npz/era5 \
    --start-date 2019/08/01 \
    --end-date 2021/08/31 \
    --variable mslp sp t2m u10 v10 tp tcwv \
        q1000 q850 q500 q250 \
        t1000 t850 t500 t250 \
        u1000 u850 u500 u250 \
        v1000 v850 v500 v250 \
        z1000 z850 z500 z250 \