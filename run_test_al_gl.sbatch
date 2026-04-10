#!/bin/bash --login
#
# Slurm batch script for running the active-learning pipeline in this repo.
#
# Usage examples:
#   sbatch run_test_al_gl.sbatch
#   sbatch --export=ALL,DSET=box,METRIC=raw,CONFIG=config_toy.yaml,RESULTSDIR=results,ITERS=100,K=8,NUMCORES=4 run_test_al_gl.sbatch
#
# Notes:
# - The active-learning loop itself is CPU-bound and parallelized with joblib.
# - Request a GPU only when your config uses representation.type: nn, because that
#   is the only path in the current code that can train on CUDA.

#SBATCH -J "rwll-al"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=1024M
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Uncomment this line when running a config that uses representation.type: nn.
#SBATCH --gpus=1

# Optional: avoid older Pascal GPUs (e.g., P100/sm_60) if your torch build is new.
##SBATCH --constraint=hopper|lovelace|ampere
# export MKL_DEBUG_CPU_TYPE=5
# export OPENBLAS_CORETYPE=Haswell
# export CUDA_VISIBLE_DEVICES=""

set -euo pipefail

mkdir -p logs

# Defaults can be overridden with: sbatch --export=ALL,VAR=value ...
: "${DSET:=mnist-mod3}"
: "${METRIC:=vae}"
: "${CONFIG:=config_mlp.yaml}"
: "${RESULTSDIR:=results}"
: "${ITERS:=10}"
: "${K:=0}"
: "${CHEATK:=50}"
: "${NUMCORES:=${SLURM_CPUS_PER_TASK:-1}}"
: "${RUN_IMPORT_CHECK:=0}"
: "${GL_KNN_METHOD:=kdtree}"

mamba activate rwll_env
export PYTHONNOUSERSITE=1

export GL_KNN_METHOD
echo "[diag] GL_KNN_METHOD=$GL_KNN_METHOD"
python -c "import torch; print('[diag] torch=', torch.__version__); print('[diag] torch_arch_list=', getattr(torch.cuda, 'get_arch_list', lambda: [])())"

python test_al_gl.py \
    --dataset "$DSET" \
    --metric "$METRIC" \
    --numcores "$NUMCORES" \
    --iters "$ITERS" \
    --resultsdir "$RESULTSDIR" \
    --config "$CONFIG" \
    --K "$K" \
    --cheatK "$CHEATK"

python accuracy_al_gl.py \
    --dataset "$DSET" \
    --metric "$METRIC" \
    --numcores "$NUMCORES" \
    --iters "$ITERS" \
    --resultsdir "$RESULTSDIR" \
    --config "$CONFIG"

python compile_summary.py \
    --dataset "$DSET" \
    --resultsdir "$RESULTSDIR" \
    --config "$CONFIG" \
    --iters "$ITERS"