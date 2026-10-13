"""Apply the version-pinned PRMS 5.2.1.1 glacier patch to a source tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from symfluence.core.exceptions import ValidationError


def apply(source: Path):
    here=Path(__file__).resolve().parent
    manifest=json.loads((here/'prms5211_glacier_conservation.json').read_text())
    states=[]
    for relative,expected in manifest.items():
        content=(source/relative).read_text()
        digest=hashlib.sha256(content.encode()).hexdigest()
        state=next((name for name in ['before','after'] if digest==expected[name+'_normalized_sha256']),None)
        if state is None:
            raise ValidationError(f'Unexpected source version or local modifications: {relative}')
        states.append(state)
    if all(state=='after' for state in states):
        print('Glacier conservation patch already applied');return
    if not all(state=='before' for state in states):
        raise ValidationError('Partially patched source tree; restore a complete source snapshot first')
    for relative in manifest:
        p=source/relative;p.write_text(p.read_text())  # normalize upstream CRLF
    patch=here/'prms5211_glacier_conservation.patch'
    subprocess.run(['patch','--dry-run','-p1','-i',str(patch)],cwd=source,check=True)
    subprocess.run(['patch','-p1','-i',str(patch)],cwd=source,check=True)
    for relative,expected in manifest.items():
        assert hashlib.sha256((source/relative).read_text().encode()).hexdigest()==expected['after_normalized_sha256']
    print('Applied glacier conservation / optional fixed-geometry patch; rebuild mmf and prms')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path)
    apply(parser.parse_args().source.resolve())
