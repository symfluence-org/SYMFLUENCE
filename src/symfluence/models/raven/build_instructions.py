"""Reproducible Raven build with an opt-in finite-glacier storage switch."""

from __future__ import annotations

from pathlib import Path

from symfluence.core.registries import R


@R.build_instructions.add("raven")
def get_raven_build_instructions():
    patch = Path(__file__).with_name("patches") / "finite_glacier_storage.patch"
    return dict(
        description="Raven HBV-EC with opt-in finite glacier storage",
        config_path_key="RAVEN_INSTALL_PATH",
        config_exe_key="RAVEN_EXE",
        default_path_suffix="installs/raven/bin",
        default_exe="Raven",
        repository="https://github.com/CSHS-CWRA/RavenHydroFramework.git",
        branch="main",
        install_dir="raven",
        build_commands=[
            f'''set -e
git checkout e7e28688c72a71726853e6a0a24ccda0d266b55f
if ! git apply --reverse --check "{patch}" 2>/dev/null; then
  git apply "{patch}"
fi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 2
mkdir -p bin
cp build/Raven bin/Raven
'''
        ],
        dependencies=[],
        test_command=None,
        verify_install={"file_paths": ["bin/Raven"], "check_type": "exists"},
        order=20,
        optional=True,
    )
