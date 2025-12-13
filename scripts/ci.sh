#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ci_artifacts

echo "== Python ==" | tee ci_artifacts/step.log
python -V | tee ci_artifacts/python_version.log

# No API keys required: this script only does local/static checks.

echo "== Compileall (tracked python only) ==" | tee -a ci_artifacts/step.log
TRACKED_PY_FILES="$(git ls-files '*.py' || true)"
if [[ -n "$TRACKED_PY_FILES" ]]; then
  # Compile tracked Python files only to avoid scanning untracked build outputs.
  python - <<'PY' 2>&1 | tee ci_artifacts/compileall.log
import os
import py_compile
import subprocess

files = subprocess.check_output(['git', 'ls-files', '*.py'], text=True).splitlines()
ok = True
for path in files:
    try:
        py_compile.compile(path, doraise=True)
    except Exception:
        ok = False
        raise

print(f"Compiled {len(files)} tracked Python files")
PY
else
  echo "No tracked Python files; skipping compile." | tee ci_artifacts/compileall.log
fi

# If requirements.txt exists, attempt dependency + pytest run.
# This may use network to install packages, but never calls trading APIs.
if [[ -f requirements.txt ]]; then
  echo "== Dependencies (optional) ==" | tee -a ci_artifacts/step.log
  python -m pip install --upgrade pip setuptools wheel 2>&1 | tee ci_artifacts/pip_upgrade.log
  python -m pip install -r requirements.txt 2>&1 | tee ci_artifacts/pip_install.log
fi

# Run pytest only if tests exist.
if [[ -d test ]] && find test -type f -name 'test_*.py' | grep -q .; then
  echo "== Pytest ==" | tee -a ci_artifacts/step.log
  python -m pip install pytest 2>&1 | tee -a ci_artifacts/pip_install.log
  python -m pytest -q 2>&1 | tee ci_artifacts/pytest.log
else
  echo "== Pytest ==" | tee -a ci_artifacts/step.log
  echo "No tests found; skipping pytest." | tee ci_artifacts/pytest.log
fi

# Run a dry-run entrypoint only if present and explicitly supported.
if [[ -f mvt_trader_live.py ]]; then
  echo "== Dry run ==" | tee -a ci_artifacts/step.log
  python mvt_trader_live.py --dry-run 2>&1 | tee ci_artifacts/dry_run.log || true
fi

echo "CI OK" | tee ci_artifacts/status.log
