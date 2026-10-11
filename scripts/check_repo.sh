#!/usr/bin/env bash
# Repo preflight: the one command to run before (and after) any pipeline rerun.
#
#   bash scripts/check_repo.sh
#
# Checks, in order:
#   0. the runtime dependencies import
#   1. every Python file under src/, scripts/ and tests/ compiles
#   2. every shell script parses (bash -n)
#   3. the unit test suite passes (torch-free; numpy is enough)
#   4. the handover bundle verifies (skipped when handover/ is absent)
#   5. every path, module and glob README.md points at actually resolves
#
# Exit status is non-zero if anything fails.  Nothing here trains or evaluates a
# model -- this is a static preflight, so it is safe to run on a login node.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PY="${PYTHON:-python3}"
FAILURES=0

pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }
detail() { sed 's/^/         /'; }

echo "== StonkBench preflight =="

# ---------------------------------------------------------------- 0. deps
echo "[0] runtime dependencies"
if ! "${PY}" -c 'import numpy' >/dev/null 2>&1; then
  fail "numpy is not importable (install the environment first)"
  echo "         bash scripts/install_stonkbench.sh    # builds ~/.venvs/stonkbench"
  echo "         source ~/.venvs/stonkbench/bin/activate"
  echo "         (or: PYTHONPATH=<dir-with-numpy> bash scripts/check_repo.sh)"
  echo
  echo "PREFLIGHT FAILED: ${FAILURES} check(s)"
  exit 1
fi
pass "numpy is importable"

# ---------------------------------------------------------------- 1. python
echo "[1] python compiles"
bad="$("${PY}" - <<'PY'
import pathlib, py_compile

skip = ("src/models/deep_learning",)  # vendored third-party repos
bad = []
for root in ("src", "scripts", "tests"):
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        if any(str(path).startswith(s) for s in skip):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            bad.append(f"{path}: {str(exc).splitlines()[-1]}")
print("\n".join(bad))
PY
)"
if [[ -z "${bad}" ]]; then
  pass "all python files compile"
else
  fail "python compile errors"
  printf '%s\n' "${bad}" | detail
fi

# ------------------------------------------------------------------ 2. bash
echo "[2] shell scripts parse"
sh_bad=0
while IFS= read -r script; do
  if ! bash -n "${script}" 2>/tmp/stonkbench_bash_n.err; then
    fail "bash -n ${script}"
    detail </tmp/stonkbench_bash_n.err
    sh_bad=$((sh_bad + 1))
  fi
done < <(find scripts -name '*.sh' | sort)
if [[ ${sh_bad} -eq 0 ]]; then
  pass "all $(find scripts -name '*.sh' | wc -l) shell scripts parse"
fi

# --------------------------------------------------------------- 3. tests
echo "[3] unit tests"
if "${PY}" -m unittest discover -s tests >/tmp/stonkbench_tests.log 2>&1; then
  pass "python3 -m unittest discover -s tests  ($(grep -oE 'Ran [0-9]+ tests?' /tmp/stonkbench_tests.log | tail -1))"
else
  fail "unit tests"
  tail -25 /tmp/stonkbench_tests.log | detail
fi

# --------------------------------------------------------------- 4. bundle
echo "[4] handover bundle"
if [[ -f handover/MANIFEST.json ]]; then
  if "${PY}" scripts/handover/verify_bundle.py >/tmp/stonkbench_bundle.log 2>&1; then
    pass "handover/ verifies"
  else
    fail "handover/ verification"
    tail -25 /tmp/stonkbench_bundle.log | detail
  fi
else
  pass "handover/ absent - skipped (regenerate with scripts/handover/)"
fi

# --------------------------------------------------------------- 5. docs
echo "[5] paths, modules and globs documented in README.md resolve"
doc_bad=0
while IFS= read -r path; do
  # Trim leading ./ and at most one trailing punctuation character.
  clean="${path#./}"
  clean="${clean%,}"; clean="${clean%:}"; clean="${clean%)}"; clean="${clean%;}"; clean="${clean%.}"

  if [[ "${clean}" == *[*?]* ]]; then
    # A glob (e.g. scripts/slurm/final_training*.sh) must match at least one file.
    compgen -G "${clean}" >/dev/null || { fail "README.md glob matches nothing: ${clean}"; doc_bad=$((doc_bad + 1)); }
    continue
  fi
  if [[ "${clean}" != */* && "${clean}" == *.* ]]; then
    # A dotted name with no slash is a `python -m` module reference:
    # src.a.b -> src/a/b.py.  Anything with a slash is an ordinary path.
    mod_path="${clean//./\/}"
    [[ -e "${mod_path}.py" || -e "${mod_path}/__init__.py" ]] || {
      fail "README.md references missing module: ${clean}"; doc_bad=$((doc_bad + 1));
    }
    continue
  fi
  [[ -e "${clean}" ]] || { fail "README.md references missing path: ${clean}"; doc_bad=$((doc_bad + 1)); }
done < <(grep -oE '(scripts|src|tests|handover)[A-Za-z0-9_./*-]*' README.md | sort -u)
if [[ ${doc_bad} -eq 0 ]]; then
  pass "every path, module and glob named in README.md resolves"
fi

# -------------------------------------------------------------- 6. docker
echo "[6] docker setup is self-consistent"
docker_bad=0
if ! docker_bad_out="$("${PY}" - <<'PY' 2>&1
import json, pathlib, re

bad = []
# Dockerfile.base must keep a valid JSON CMD so the image has a sane default.
df = pathlib.Path("Dockerfile.base").read_text()
m = re.search(r"^CMD (\[.*\])$", df, re.M | re.S)
if not m:
    bad.append("Dockerfile.base has no CMD")
else:
    try:
        json.loads(m.group(1))
    except Exception as exc:  # noqa: BLE001
        bad.append(f"Dockerfile.base CMD is not valid JSON: {exc}")
if "PYTHONPATH=/app" not in df:
    bad.append("Dockerfile.base does not set PYTHONPATH=/app")

# Every service the README tells the user to run must exist in docker-compose.yml.
compose = pathlib.Path("docker-compose.yml").read_text()
services = set(re.findall(r"^  ([a-z0-9-]+):\s*$", compose, re.M))
for name in sorted(set(re.findall(r"docker compose (?:up|run|build) ([a-z0-9 -]+)", pathlib.Path("README.md").read_text()))):
    for svc in name.split():
        if svc not in services:
            bad.append(f"README.md references missing compose service: {svc}")
if bad:
    print("\n".join(bad))
PY
)"; then
  fail "docker check errored"
  printf '%s\n' "${docker_bad_out}" | detail
  docker_bad=$((docker_bad + 1))
elif [[ -n "${docker_bad_out}" ]]; then
  fail "docker setup inconsistent"
  printf '%s\n' "${docker_bad_out}" | detail
  docker_bad=$((docker_bad + 1))
fi
if [[ ${docker_bad} -eq 0 ]]; then
  pass "Dockerfile.base CMD is valid JSON, PYTHONPATH is set, compose services resolve"
fi

# -------------------------------------------------------------- summary
echo
if [[ ${FAILURES} -gt 0 ]]; then
  echo "PREFLIGHT FAILED: ${FAILURES} check(s)"
  exit 1
fi
echo "PREFLIGHT PASSED"
