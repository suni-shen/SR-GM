#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

cd "${PROJECT_ROOT}"
python -m graphslim.train_all -G 0 -M gcond -D ele-fashion -R 0.001 --feat_name clip --setting ind 

python -m graphslim.train_all -G 0 -M gcond -D nc_ar_cloth -R 0.001 --feat_name imagebind --setting trans --metric roc_auc 

python -m graphslim.train_all -G 0 -M gcdm -D ele-fashion -R 0.001 --feat_name clip --SR True --d_text 512 --lambad 10000.0 -E 1500 --setting ind 
python -m graphslim.train_all -G 1 -M gcdm -D ele-fashion -R 0.001 --feat_name clip -E 1500 --setting ind 