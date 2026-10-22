#!/usr/bin/env bash
# Run one LamaH-ICE/SUMMA configuration per SLURM array element on Elja.
#
# Recommended submission (the array size is derived from the config count):
#   bash scripts/hpc/lamah_ice_summa_array.sbatch /path/to/summa-configs
#
# Limit concurrency (default: 8):
#   MAX_CONCURRENT=4 bash scripts/hpc/lamah_ice_summa_array.sbatch /path/to/summa-configs
#
# Direct submission is also supported:
#   sbatch --array=0-58%8 --export=ALL,CONFIG_DIR=/path/to/summa-configs \
#     scripts/hpc/lamah_ice_summa_array.sbatch
#
# Every config must have a unique domain.name. Shared forcing and LamaH-ICE
# source data may be referenced by all configs, but should be staged before
# submitting the array so that tasks do not race while downloading/extracting.

#SBATCH --job-name=lamah-summa
#SBATCH --output=logs/lamah_summa_%A_%a.out
#SBATCH --error=logs/lamah_summa_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

SCRIPT_PATH=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")
CONFIG_PATTERN=${CONFIG_PATTERN:-config_lamahice_*SUMMA*.yaml}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}

# When invoked with bash on the login node, count configs and submit the job.
if [[ -z ${SLURM_JOB_ID:-} ]]; then
    CONFIG_DIR=${1:-${CONFIG_DIR:-}}
    if [[ -z $CONFIG_DIR || ! -d $CONFIG_DIR ]]; then
        echo "Usage: bash $0 CONFIG_DIR" >&2
        exit 2
    fi

    mapfile -t SUBMIT_CONFIGS < <(
        find "$CONFIG_DIR" -maxdepth 1 -type f -name "$CONFIG_PATTERN" -print | LC_ALL=C sort
    )
    if (( ${#SUBMIT_CONFIGS[@]} == 0 )); then
        echo "No configs matching '$CONFIG_PATTERN' in $CONFIG_DIR" >&2
        exit 2
    fi

    mkdir -p logs
    LAST_INDEX=$(( ${#SUBMIT_CONFIGS[@]} - 1 ))
    echo "Submitting ${#SUBMIT_CONFIGS[@]} LamaH-ICE/SUMMA configs (max $MAX_CONCURRENT concurrent)"
    exec sbatch \
        --array="0-${LAST_INDEX}%${MAX_CONCURRENT}" \
        --export="ALL,CONFIG_DIR=$(cd -- "$CONFIG_DIR" && pwd),CONFIG_PATTERN=$CONFIG_PATTERN" \
        "$SCRIPT_PATH"
fi

: "${CONFIG_DIR:?Submit with CONFIG_DIR set or use the login-node wrapper shown above}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"

mapfile -t CONFIGS < <(
    find "$CONFIG_DIR" -maxdepth 1 -type f -name "$CONFIG_PATTERN" -print | LC_ALL=C sort
)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#CONFIGS[@]} )); then
    echo "Array index $SLURM_ARRAY_TASK_ID is outside the ${#CONFIGS[@]} discovered configs" >&2
    exit 2
fi

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

module purge
module load foss/2023a
module load CMake/3.26.3 Python/3.11.3 netCDF-Fortran/4.6.1 GDAL/3.7.1

# Optional activation hooks for Berglind's installation. Usually only one is
# needed; both are intentionally controlled by environment variables.
if [[ -n ${SYMF_VENV:-} ]]; then
    # shellcheck disable=SC1090
    source "$SYMF_VENV/bin/activate"
fi
if [[ -n ${SYMF_CONDA_ENV:-} ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$SYMF_CONDA_ENV"
fi

if [[ -n ${SYMF_REPO:-} ]]; then
    cd "$SYMF_REPO"
fi

echo "Job:       $SLURM_JOB_ID"
echo "Task:      $SLURM_ARRAY_TASK_ID"
echo "Host:      $HOSTNAME"
echo "Config:    $CONFIG"
echo "Started:   $(date --iso-8601=seconds)"
echo "Python:    $(command -v python)"
echo "SYMFLUENCE: $(command -v symfluence)"
echo "GDAL:      $(gdal-config --version)"

# This creates the domain, acquires/preprocesses its data, writes SUMMA inputs,
# runs SUMMA, and postprocesses the result. Override WORKFLOW_STEPS at submit
# time if only part of the pipeline is wanted.
read -r -a STEPS <<< "${WORKFLOW_STEPS:-setup_project create_pour_point acquire_attributes define_domain discretize_domain process_observed_data acquire_forcings model_agnostic_preprocessing build_model_ready_store model_specific_preprocessing run_model postprocess_results}"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" \
    symfluence workflow steps "${STEPS[@]}" --config "$CONFIG"

echo "Finished:  $(date --iso-8601=seconds)"
