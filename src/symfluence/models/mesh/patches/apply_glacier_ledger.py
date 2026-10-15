"""Apply the versioned experimental ledger to a private MESH source checkout."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1])
here = Path(__file__).resolve().parent
if (root / "Driver/MESH_Driver/glacier_ledger.f90").exists():
    raise SystemExit("Patch already applied")


def edit(p, old, new):
    p = root / p
    s = p.read_text()
    assert s.count(old) == 1, (p, old, s.count(old))
    p.write_text(s.replace(old, new))


shutil.copy2(here / "glacier_ledger.f90", root / "Driver/MESH_Driver/glacier_ledger.f90")
edit(Path("makefile.def"), "OBJECTS+=\tmpi_flags.o", "OBJECTS+= glacier_ledger.o\nOBJECTS+=\tmpi_flags.o")
base = Path("LSS_Model/CLASS/3.6")
p = base / "src/ICEBAL.f"
edit(p, "      IMPLICIT NONE", "      USE glacier_ledger, ONLY: ledger_flux\n      IMPLICIT NONE")
edit(
    p,
    "                      ZPOND(I)=0.0",
    "                      CALL ledger_flux(I,4,FI(I)*ZFREZ*RHOICE)\n                      ZPOND(I)=0.0",
)
edit(
    p,
    "                  ZMOVE (I)=ZMELT+EVAP(I)*DELT*RHOW/RHOICE",
    """                  CALL ledger_flux(I,1,FI(I)*ZMELT*RHOICE)
                  CALL ledger_flux(I,2,FI(I)*EVAP(I)*DELT*RHOW)
                  CALL ledger_flux(I,5,FI(I)*QMELT(I)*DELT)
                  ZMOVE (I)=ZMELT+EVAP(I)*DELT*RHOW/RHOICE""",
)
edit(
    p,
    "              ICE(I)=ICE(I)+SNOCONV",
    "              CALL ledger_flux(I,3,FI(I)*SNOCONV)\n              ICE(I)=ICE(I)+SNOCONV",
)
p = base / "sa_mesh_process/RUNCLASS36_config.f90"
edit(p, "module RUNCLASS36_config", "module RUNCLASS36_config\n    use glacier_ledger, only: ledger_init")
edit(p, "        catv%ZDH = 2.0", "        catv%ZDH = 2.0\n        call ledger_init(csfv%MID,csfv%SAND(:,1),il1,il2)")
p = base / "sa_mesh_process/RUNCLASS36_module.f90"
edit(
    p, "module RUNCLASS36_module", "module RUNCLASS36_module\n    use glacier_ledger, only: ledger_begin, ledger_commit"
)
edit(p, "            cdv%ICE = 0.0", "            call ledger_begin()\n            cdv%ICE = 0.0")
edit(
    p,
    "            !> Distribution of blowing snow mass",
    "            call ledger_commit()\n            !> Distribution of blowing snow mass",
)
for mode, call in [("save", "ledger_save"), ("read", "ledger_read")]:
    p = base / f"sa_mesh_process/{mode}_init_prog_variables_class.f90"
    edit(
        p,
        "    use RUNCLASS36_variables",
        f"    use RUNCLASS36_variables\n    use model_dates, only: ic\n    use glacier_ledger, only: {call}",
    )
    end = "end subroutine !" + mode + "_init_prog_variables_class"
    clock = "now" if mode == "save" else "start"
    stamp = ", [" + ", ".join(f"ic%{clock}%{x}" for x in ("year", "jday", "hour", "mins")) + "]"
    edit(p, end, f"    call {call}(trim(resume_fn)//'.glacier-v1'{stamp})\n\n" + end)
print(root)
