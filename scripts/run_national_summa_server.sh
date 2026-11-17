#!/usr/bin/env bash
# National SUMMA 1991-2025 on the landslíkan server (ADR 0015): 5-year spin-up (WY1992-1996 from the cold state),
# then WY1992-2025 from the spin-up's end state. Run from the Mac (the CARRA archive is here):
#   nohup bash scripts/run_national_summa_server.sh > <log> 2>&1 &
set -euo pipefail
PROCS_ICE=${PROCS_ICE:-5}; PROCS_LAND=${PROCS_LAND:-7}; NICE=${NICE:-10}
PY=/Users/darri/repos/SYMFLUENCE/.venv/bin/python
CODE=/Users/darri/repos/SYMFLUENCE-national/src/symfluence/models/summa
FABRIC=/Users/darri/repos/landslikan/data/build/v1.1_25m
DOMAIN=/Users/darri/repos/SYMFLUENCE_data/domain_landslikan
H3=/Users/darri/repos/landslikan/data/raw/carra/3h
R=/home/darey/landslikan_models
common=(--host darey@100.85.106.89 --wsl Ubuntu-24.04 --stage-win Downloads/landslikan_build/stage
        --stage-wsl /mnt/c/Users/darey/Downloads/landslikan_build/stage --remote-domain $R/run/domain_landslikan
        --remote-code $R/run/code --exe $R/summa-glacier/bin/summa.exe --procs-ice $PROCS_ICE --procs-land $PROCS_LAND
        --nice $NICE)
cd /Users/darri/repos/landslikan
$PY $CODE/national_remote.py $FABRIC $DOMAIN $H3 summa_spinup 1992 1996 "${common[@]}"
$PY $CODE/national_remote.py $FABRIC $DOMAIN $H3 summa_v1 1992 2025 "${common[@]}" \
    --init-ice restart_ice_WY1992-1996.nc --init-land restart_land_WY1992-1996.nc
echo FINISHED
