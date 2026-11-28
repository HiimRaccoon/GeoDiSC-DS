#!/usr/bin/env bash
set -euo pipefail
: "${GEOBENCH_CONDA_SH:?Set the privately verified Conda activation path}"
: "${GEOBENCH_CONDA_ENV:?Set the verified environment name or path}"
source "$GEOBENCH_CONDA_SH"
conda activate "$GEOBENCH_CONDA_ENV"
exec srun --ntasks=1 python -m geobench execute-job --root "$1" --manifest "$2" --index "${SLURM_ARRAY_TASK_ID:?}"
