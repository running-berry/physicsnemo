#!/bin/bash
# All hours (no spaces between commas!)
python check_rwrf_npz_missing.py \
  --root <path_to_rwrf_npz_dir>\
  --start-date 2019/08/01 \
  --end-date   2022/09/01 \
  --hours 00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
  --csv-out missing.csv \