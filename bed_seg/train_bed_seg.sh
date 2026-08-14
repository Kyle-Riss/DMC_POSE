#!/usr/bin/env bash
set -euo pipefail
source /home/dmc/anaconda3/etc/profile.d/conda.sh
conda activate pose-cuda
cd /home/dmc/AI/DMC_POSE/bed_seg

python prepare_dataset.py "$@"
python train_bed_seg.py
