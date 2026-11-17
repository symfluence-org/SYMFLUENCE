# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Run national SUMMA on a remote Linux host, feeding it forcing from this machine (ADR 0015).

The 3-h CARRA archive (18 GB) stays here; the host has the SUMMA build and the settings. For each chunk of water
years: write the chunk's forcing here, copy it over and delete it here, run ``national_run.py chunk`` there
(ice and land side by side, restarts merged), copy the daily output back and delete it there. The next chunk's
forcing is written and copied while the current chunk runs. Progress is kept in ``<sim>/remote_state.json``, so a
rerun resumes after the last finished chunk.

    python national_remote.py <fabric_dir> <domain_dir> <carra_3h_dir> <name> <wy0> <wy1> \\
        --host user@host --remote-domain /home/u/landslikan_models/run/domain_landslikan \\
        --remote-code /home/u/landslikan_models/run/code --exe /home/u/landslikan_models/summa-glacier/bin/summa.exe

The host is reached as ``ssh <host> "wsl -d <distro> -- bash -lc '<cmd>'"`` when --wsl is given (a Windows host
with a PowerShell login), else ``ssh <host> <cmd>``; files go by scp to --stage (a Windows path the WSL side sees
under /mnt/c) or straight to the remote domain.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import national as N  # noqa: E402

logger = logging.getLogger(__name__)


class Host:
    def __init__(self, a):
        self.a = a

    def sh(self, cmd: str, check: bool = True) -> str:
        if self.a.wsl:
            full = ['ssh', '-o', 'ConnectTimeout=30', self.a.host,
                    f'wsl -d {self.a.wsl} -- bash -c {shlex.quote(cmd)}']
        else:
            full = ['ssh', '-o', 'ConnectTimeout=30', self.a.host, cmd]
        for attempt in range(5):
            r = subprocess.run(full, capture_output=True, text=True)
            if r.returncode != 255:          # 255: ssh itself failed (network); retry
                break
            time.sleep(30 * (attempt + 1))
        if check and r.returncode != 0:
            raise RuntimeError(f'remote command failed ({r.returncode}): {cmd}\n{r.stdout}\n{r.stderr}')
        return r.stdout

    def put(self, files: list, remote_dir: str) -> None:
        """Copy local files into remote_dir (via the Windows stage directory when --wsl)."""
        if self.a.wsl:
            subprocess.run(['scp', '-q'] + [str(f) for f in files] + [f'{self.a.host}:{self.a.stage_win}/'],
                           check=True)
            names = ' '.join(shlex.quote(Path(f).name) for f in files)
            self.sh(f'mkdir -p {remote_dir} && cd {self.a.stage_wsl} && mv {names} {remote_dir}/')
        else:
            self.sh(f'mkdir -p {remote_dir}')
            subprocess.run(['scp', '-q'] + [str(f) for f in files] + [f'{self.a.host}:{remote_dir}/'], check=True)

    def get(self, remote_glob: str, local_dir: Path) -> None:
        """Copy remote files back (through the stage directory when --wsl)."""
        local_dir.mkdir(parents=True, exist_ok=True)
        if self.a.wsl:
            names = self.sh(f'ls {remote_glob} 2>/dev/null || true').split()
            if not names:
                return
            self.sh(f'cp {" ".join(names)} {self.a.stage_wsl}/')
            for n in names:
                b = Path(n).name
                subprocess.run(['scp', '-q', f'{self.a.host}:{self.a.stage_win}/{b}', str(local_dir / b)],
                               check=True)
            self.sh(f'cd {self.a.stage_wsl} && rm -f {" ".join(Path(n).name for n in names)}')
        else:
            subprocess.run(['scp', '-q', f'{self.a.host}:{remote_glob}', str(local_dir)], check=True)


def chunks(wy0: int, wy1: int, size: int) -> list:
    return [(c, min(c + size - 1, wy1)) for c in range(wy0, wy1 + 1, size)]


def stage_forcing(a, host: Host, c0: int, c1: int, runs: tuple) -> None:
    """Write the chunk's forcing here, copy it to the host, delete it here."""
    fdir = a.domain / 'forcing' / 'SUMMA'
    for wy in range(c0, c1 + 1):
        N.write_forcing(a.fabric, a.domain, a.h3, wy, runs=runs)
        files = [fdir / f'{r}_WY{wy}.nc' for r in runs]
        host.put(files, f'{a.remote_domain}/forcing/SUMMA')
        for f in files:
            f.unlink()
        logger.info('forcing WY%d staged', wy)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('fabric', type=Path)
    ap.add_argument('domain', type=Path)
    ap.add_argument('h3', type=Path)
    ap.add_argument('name')
    ap.add_argument('wy0', type=int)
    ap.add_argument('wy1', type=int)
    ap.add_argument('--host', required=True)
    ap.add_argument('--wsl', default='')
    ap.add_argument('--stage-win', default='Downloads/landslikan_build/stage')
    ap.add_argument('--stage-wsl', default='')
    ap.add_argument('--remote-domain', required=True)
    ap.add_argument('--remote-code', required=True)
    ap.add_argument('--exe', required=True)
    ap.add_argument('--chunk', type=int, default=5)
    ap.add_argument('--procs-ice', type=int, default=10)
    ap.add_argument('--procs-land', type=int, default=13)
    ap.add_argument('--poll', type=int, default=300)
    a = ap.parse_args()
    if a.wsl and not a.stage_wsl:
        ap.error('--wsl needs --stage-wsl (the --stage-win directory as WSL sees it, e.g. /mnt/c/Users/u/...)')
    runs = ('ice', 'land')
    host = Host(a)
    if a.wsl:
        host.sh(f'mkdir -p {a.stage_wsl}')
    local = a.domain / 'simulations' / a.name / 'SUMMA'
    local.mkdir(parents=True, exist_ok=True)
    state_f = local / 'remote_state.json'
    state = json.loads(state_f.read_text()) if state_f.exists() else {'done': [], 'init': {r: 'coldState.nc'
                                                                                           for r in runs}}
    rsim = f'{a.remote_domain}/simulations/{a.name}/SUMMA'
    todo = [c for c in chunks(a.wy0, a.wy1, a.chunk) if f'WY{c[0]}-{c[1]}' not in state['done']]
    pool = ThreadPoolExecutor(1)
    staged = pool.submit(stage_forcing, a, host, *todo[0], runs) if todo else None
    for i, (c0, c1) in enumerate(todo):
        tag = f'WY{c0}-{c1}'
        staged.result()
        cmd = (f'cd {a.remote_code} && setsid nohup python3 national_run.py chunk - {a.remote_domain} {a.name} '
               f'{c0} {c1} --exe {a.exe} --init-ice {state["init"]["ice"]} --init-land {state["init"]["land"]} '
               f'--procs-ice {a.procs_ice} --procs-land {a.procs_land} > {rsim}/chunk_{tag}.log 2>&1 '
               f'< /dev/null & disown')
        host.sh(f'mkdir -p {rsim} && rm -f {rsim}/done_{tag}.json {rsim}/failed_{tag}.txt')
        host.sh(cmd)
        t = time.time()
        logger.info('%s started on %s', tag, a.host)
        # the next chunk's forcing goes over while this one runs
        staged = pool.submit(stage_forcing, a, host, *todo[i + 1], runs) if i + 1 < len(todo) else None
        while True:
            time.sleep(a.poll)
            st = host.sh(f'ls {rsim}/done_{tag}.json {rsim}/failed_{tag}.txt 2>/dev/null || true')
            if 'failed_' in st:
                raise RuntimeError(f'{tag} failed: ' + host.sh(f'cat {rsim}/failed_{tag}.txt; tail -5 '
                                                              f'{rsim}/chunk_{tag}.log'))
            if 'done_' in st:
                break
        res = json.loads(host.sh(f'cat {rsim}/done_{tag}.json'))
        for r in runs:
            host.get(f'{rsim}/{r}/{r}_{tag}_G*_day.nc', local / r)
            host.sh(f'rm -f {rsim}/{r}/{r}_{tag}_G*_day.nc {rsim}/{r}/{r}_{tag}_G*_timestep.nc '
                    f'{rsim}/{r}/{r}_{tag}_restart_*.nc')
            state['init'][r] = res[r]['restart']
        state['done'].append(tag)
        state.setdefault('timing', {})[tag] = dict(res, wall_s=round(time.time() - t))
        state_f.write_text(json.dumps(state, indent=1))
        logger.info('%s done in %.1f h', tag, (time.time() - t) / 3600)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
