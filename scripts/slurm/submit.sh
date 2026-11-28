#!/usr/bin/env bash
set -euo pipefail
module load slurm/slurm/25.05
exec python -m geobench submit-array "$@"
