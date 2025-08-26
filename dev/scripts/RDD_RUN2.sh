#!/bin/bash

LOGFILE="job_id.log"
DATE="2025-08-26"

# 每一欄的 script
#SCRIPTS=("RDD1.sh" "RDD2.sh" "RDD3.sh" "RDD4.sh")
SCRIPTS=("RDD2.sh" "RDD3.sh")

# 表格時間，對應四欄
#TIMES_COL1=("05:30:00" "08:30:00" "09:00:00" "12:00:00" "12:30:00" "15:30:00" "16:00:00" "19:00:00")
TIMES_COL2=("05:45:00" "09:15:00" "12:45:00" "16:15:00" "19:30:00")
TIMES_COL3=("07:30:00" "11:00:00" "14:30:00" "18:00:00" "21:30:00")
#TIMES_COL4=("07:45:00" "10:45:00" "11:15:00" "14:15:00" "14:45:00" "17:45:00" "18:15:00" "21:15:00")


# 把所有欄位放在陣列方便迴圈
#ALL_TIMES=(TIMES_COL1[@] TIMES_COL2[@] TIMES_COL3[@] TIMES_COL4[@])
ALL_TIMES=(TIMES_COL2[@] TIMES_COL3[@])

# 迴圈跑四欄
for i in {0..1}; do
    SCRIPT=${SCRIPTS[$i]}
    TIMES=(${!ALL_TIMES[$i]})

    for t in "${TIMES[@]}"; do
        OUTPUT=$(sbatch --begin="${DATE}T${t}" $SCRIPT)
        JOBID=$(echo $OUTPUT | awk '{print $4}')

	# 等待節點分配（避免記錄到空）
        sleep 5
        NODE=$(squeue -j $JOBID -o "%N" -h)

        echo "$(date '+%F %T') $JOBID ${DATE}T${t} $SCRIPT $NODE" >> $LOGFILE
        echo "Submitted $SCRIPT at ${DATE}T${t}, JobID=$JOBID, Node=$NODE"
    done
done
