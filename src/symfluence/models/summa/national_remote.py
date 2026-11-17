# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Run national SUMMA on a remote Linux host, feeding it forcing from this machine (ADR 0015).

The 3-h CARRA archive (18 GB) stays here; the host has the SUMMA build and the settings. For each chunk of water
years: write the chunk's forcing here, copy it over and delete it here, run ``national_run.py chunk`` there
for ice and for land, each with its own restart chain), copy the daily output back and delete it there. Ice and land
are independent queues: each starts its next chunk as soon as its last one is done (up to --lead chunks ahead of the
other), sized to the cores free on the host at that moment (between --min-procs and --procs-ice/--procs-land). The
next chunk's forcing is written and copied while chunks run. Progress is kept in ``<sim>/remote_state.json``; a rerun
resumes after the last finished chunk and adopts a chunk that is still running (or finished) on the host instead of
starting it again.

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
    """Write the chunk's forcing here, copy it to the host, delete it here (WYs already there are skipped)."""
    fdir = a.domain / 'forcing' / 'SUMMA'
    there = set(host.sh(f'ls {a.remote_domain}/forcing/SUMMA 2>/dev/null || true').split())
    for wy in range(c0, c1 + 1):
        need = tuple(r for r in runs if f'{r}_WY{wy}.nc' not in there)
        if not need:
            continue
        N.write_forcing(a.fabric, a.domain, a.h3, wy, runs=need)
        files = [fdir / f'{r}_WY{wy}.nc' for r in need]
        host.put(files, f'{a.remote_domain}/forcing/SUMMA')
        for f in files:
            f.unlink()
        logger.info('forcing WY%d staged (%s)', wy, ','.join(need))


def host_status(a, host: Host) -> dict:
    return json.loads(host.sh(f'cd {a.remote_code} && python3 national_run.py hoststatus - -'))


def job_runs(job: list) -> tuple:
    """(name, c0, c1, runs, procs) of a running `chunk - <domain> <name> <c0> <c1> ...` command line."""
    opt = {job[k]: job[k + 1] for k in range(6, len(job) - 1) if job[k].startswith('--')}
    runs = tuple(opt.get('--runs', 'ice,land').split(','))
    return job[3], int(job[4]), int(job[5]), runs, {r: int(opt.get(f'--procs-{r}', 0)) for r in runs}


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
    ap.add_argument('--procs-ice', type=int, default=10, help='most processes for an ice chunk')
    ap.add_argument('--procs-land', type=int, default=13, help='most processes for a land chunk')
    ap.add_argument('--min-procs', type=int, default=3)
    ap.add_argument('--reserve', type=int, default=1, help='cores left free on the host')
    ap.add_argument('--lead', type=int, default=2, help='chunks one run may get ahead of the other')
    ap.add_argument('--poll', type=int, default=300)
    ap.add_argument('--nice', type=int, default=10, help='niceness of the SUMMA processes on the host')
    ap.add_argument('--init-ice', default='coldState.nc', help='initial state of the first chunk (settings file)')
    ap.add_argument('--init-land', default='coldState.nc')
    a = ap.parse_args()
    if a.wsl and not a.stage_wsl:
        ap.error('--wsl needs --stage-wsl (the --stage-win directory as WSL sees it, e.g. /mnt/c/Users/u/...)')
    runs = ('ice', 'land')
    pmax = {'ice': a.procs_ice, 'land': a.procs_land}
    host = Host(a)
    if a.wsl:
        host.sh(f'mkdir -p {a.stage_wsl}')
    local = a.domain / 'simulations' / a.name / 'SUMMA'
    local.mkdir(parents=True, exist_ok=True)
    state_f = local / 'remote_state.json'
    state = json.loads(state_f.read_text()) if state_f.exists() else {'init': {'ice': a.init_ice, 'land': a.init_land}}
    done = state.setdefault('done_run', {r: list(state.get('done', [])) for r in runs})   # 'done': both runs
    rsim = f'{a.remote_domain}/simulations/{a.name}/SUMMA'
    host.sh(f'mkdir -p {rsim}')
    cl = chunks(a.wy0, a.wy1, a.chunk)
    tags = [f'WY{c0}-{c1}' for c0, c1 in cl]
    n = len(cl)

    def nxt(r):
        return next((i for i, t in enumerate(tags) if t not in done[r]), n)

    def save():
        state_f.write_text(json.dumps(state, indent=1))

    # adopt chunks already running on the host, or finished there while nobody was watching
    inflight = {}                       # run -> {i, procs, t}
    marks = set(host.sh(f'ls {rsim}').split())
    for job in host_status(a, host)['jobs']:
        name, c0, c1, jr, procs = job_runs(job)
        if name == a.name and (c0, c1) in cl:
            for r in jr:
                if nxt(r) == cl.index((c0, c1)):
                    inflight[r] = dict(i=cl.index((c0, c1)), procs=procs[r], t=time.time())
                    logger.info('%s %s is running on the host: adopted', r, tags[inflight[r]['i']])
    for r in runs:
        i = nxt(r)
        if r not in inflight and i < n and ({f'done_{tags[i]}.json', f'done_{r}_{tags[i]}.json'} & marks):
            inflight[r] = dict(i=i, procs=0, t=time.time())
            logger.info('%s %s finished on the host: collecting', r, tags[i])
    staged = {v['i'] for v in inflight.values()}
    pool = ThreadPoolExecutor(1)
    stage_job = None                    # (index, future)

    while any(nxt(r) < n for r in runs):
        # -- forcing: stage the next chunk some run will need
        if stage_job and stage_job[1].done():
            stage_job[1].result()
            staged.add(stage_job[0])
            stage_job = None
        live = [nxt(r) for r in runs if nxt(r) < n]
        if not stage_job:
            want = [i for i in range(min(live), min(max(live) + 2, n))
                    if i not in staged and any(nxt(r) <= i for r in runs)]
            if want:
                i = want[0]
                stage_job = (i, pool.submit(stage_forcing, a, host, *cl[i],
                                            tuple(r for r in runs if tags[i] not in done[r])))
        # -- launch: each idle run starts its next chunk if staged and not too far ahead
        go = [r for r in runs if r not in inflight and nxt(r) < n and nxt(r) in staged and nxt(r) - min(live) <= a.lead]
        hs = host_status(a, host) if go else None
        for r in go:
            i = nxt(r)
            free = hs['ncpu'] - a.reserve - hs['busy_other'] - sum(v['procs'] for v in inflight.values())
            share = free * pmax[r] / sum(pmax[q] for q in go if q not in inflight)    # runs starting together
            procs = max(a.min_procs, min(pmax[r], int(share)))
            tag = tags[i]
            c0, c1 = cl[i]
            host.sh(f'rm -f {rsim}/done_{r}_{tag}.json {rsim}/failed_{r}_{tag}.txt')
            host.sh(f'cd {a.remote_code} && setsid -f nohup python3 national_run.py chunk - {a.remote_domain} '
                    f'{a.name} {c0} {c1} --runs {r} --exe {a.exe} --init-{r} {state["init"][r]} '
                    f'--procs-{r} {procs} --nice {a.nice} > {rsim}/chunk_{r}_{tag}.log 2>&1 < /dev/null; sleep 5')
            # setsid -f: detached before the WSL session ends (plain & was killed)
            inflight[r] = dict(i=i, procs=procs, t=time.time())
            logger.info('%s %s started on %s with %d processes (%d busy others)', r, tag, a.host, procs,
                        hs['busy_other'])
        # -- poll
        time.sleep(a.poll)
        marks = set(host.sh(f'ls {rsim}').split())
        for r in list(inflight):
            tag = tags[inflight[r]['i']]
            fail = {f'failed_{tag}.txt', f'failed_{r}_{tag}.txt'} & marks
            if fail:
                f = fail.pop()
                raise RuntimeError(f'{r} {tag} failed: ' + host.sh(f'cat {rsim}/{f}'))
            ok = sorted({f'done_{tag}.json', f'done_{r}_{tag}.json'} & marks)
            if not ok:
                continue
            res = json.loads(host.sh(f'cat {rsim}/{ok[0]}'))[r]
            host.get(f'{rsim}/{r}/{r}_{tag}_G*_day.nc', local / r)
            host.sh(f'rm -f {rsim}/{r}/{r}_{tag}_G*_day.nc {rsim}/{r}/{r}_{tag}_G*_timestep.nc '
                    f'{rsim}/{r}/{r}_{tag}_restart_*.nc')
            state['init'][r] = res['restart']
            done[r].append(tag)
            state.setdefault('timing', {})[f'{r}_{tag}'] = dict(res, wall_s=round(time.time() - inflight[r]['t']))
            save()
            logger.info('%s %s done in %.1f h', r, tag, (time.time() - inflight[r]['t']) / 3600)
            del inflight[r]
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
