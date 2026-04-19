#!/usr/bin/env bash
#SBATCH -J "rwll-al-grid"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=4G
#SBATCH --array=0-23
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Uncomment if using NN (GPU)
#SBATCH --gpus=1
##SBATCH --constraint=lovelace

set -euo pipefail

mkdir -p logs configs

eval "$(mamba shell hook --shell bash)"
mamba activate rwll_env_2
export PYTHONNOUSERSITE=1

# ---- Parameters ----
NN_LAYERS=(1 3 5 7 9 11)
NN_UPDATE_INTERVALS=(0 5 25 50)

i=$SLURM_ARRAY_TASK_ID

layer_idx=$(( i % 6 ))
interval_idx=$(( i / 6 ))

LAYER=${NN_LAYERS[$layer_idx]}
UPDATE=${NN_UPDATE_INTERVALS[$interval_idx]}

echo "[INFO] Running nn_layer=$LAYER"

# ---- Config paths ----
BASE_CONFIG="config_main.yaml"
TMP_CONFIG="configs/config_layer_${LAYER}_update_${UPDATE}.yaml"

# ---- Generate config ----
python make_config.py "$BASE_CONFIG" "$TMP_CONFIG" "$LAYER" "$UPDATE"

# Optional stability (recommended from earlier debugging)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

: "${DSET:=blobs}"
: "${METRIC:=raw}"
: "${ITERS:=60}"
: "${K:=0}"
: "${CHEATK:=50}"
: "${NUMCORES:=${SLURM_CPUS_PER_TASK:-1}}"
: "${RUN_IMPORT_CHECK:=0}"

RESULTSDIR="results_${DSET}_2/"
mkdir -p "$RESULTSDIR"

# ---- Run pipeline ----
python test_al_gl.py \
    --dataset "$DSET" \
    --metric "$METRIC" \
    --numcores "$NUMCORES" \
    --iters "$ITERS" \
    --resultsdir "$RESULTSDIR" \
    --config "$TMP_CONFIG" \
    --K "$K" \
    --cheatK "$CHEATK"

python accuracy_al_gl.py \
    --dataset "$DSET" \
    --metric "$METRIC" \
    --numcores "$NUMCORES" \
    --iters "$ITERS" \
    --resultsdir "$RESULTSDIR" \
    --config "$TMP_CONFIG"

python compile_summary.py \
    --dataset "$DSET" \
    --resultsdir "$RESULTSDIR" \
    --config "$TMP_CONFIG" \
    --iters "$ITERS"

echo "[DONE] nn_layer=$LAYER"