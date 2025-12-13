#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ci_artifacts

echo "== Python =="
python -V | tee ci_artifacts/python_version.log

echo "== Virtualenv =="
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ ! -d .venv ]]; then
    python -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
python -V

echo "== Dependencies =="
python -m pip install --upgrade pip setuptools wheel | tee ci_artifacts/pip_upgrade.log
python -m pip install -r requirements.txt | tee ci_artifacts/pip_install.log

# Ensure cmake is available (Codespaces containers sometimes omit it)
if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found; installing via pip" | tee -a ci_artifacts/build.log
  python -m pip install cmake | tee -a ci_artifacts/pip_install.log
fi

echo "== Build C++ extension =="
mkdir -p cpp/build
(
  cd cpp/build
  cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tee "$ROOT_DIR/ci_artifacts/cmake_configure.log"
  make -j2 2>&1 | tee "$ROOT_DIR/ci_artifacts/cmake_build.log"
)

echo "== Verify otrep_core import =="
python -c "import otrep_core; print(otrep_core.get_version())" 2>&1 | tee ci_artifacts/otrep_core_version.log

echo "== Pytest =="
python -m pytest -q 2>&1 | tee ci_artifacts/pytest.log

echo "== Dry run (no secrets required) =="
python mvt_trader_live.py --dry-run 2>&1 | tee ci_artifacts/dry_run.log

echo "CI OK" | tee ci_artifacts/status.log
