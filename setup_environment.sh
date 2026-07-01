#!/usr/bin/env bash
# One-click environment setup for PonziSense.
# Usage: bash setup_environment.sh [--check-only] [--skip-torch] [--preprocess]
set -euo pipefail

usage() {
  cat <<'EOF'
PonziSense environment setup

Usage:
  bash setup_environment.sh [options]

Options:
  --check-only             Validate the current environment without installing packages.
  --skip-torch             Do not reinstall PyTorch. Useful when system CUDA PyTorch is already provided.
  --system-site-packages   Create the virtual environment with access to system site packages.
  --preprocess             Run python preprocess_dataset.py after installing dependencies.
  -h, --help               Show this help message.

Environment variables:
  VENV_DIR                 Virtual environment path. Default: .venv_linux
  PYTHON_BIN               Python executable used to create the venv. Default: python3
  PYTORCH_INDEX_URL        PyTorch wheel index. Default: https://download.pytorch.org/whl/cu121
  INSTALL_TORCH            Set to 0 to skip PyTorch reinstall. Default: 1
  PIP_EXTRA_ARGS           Extra arguments appended to pip install commands.

Examples:
  bash setup_environment.sh
  bash setup_environment.sh --check-only
  INSTALL_TORCH=0 bash setup_environment.sh
  VENV_DIR=.venv_cpu PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup_environment.sh
EOF
}

log() {
  printf '[PonziSense setup] %s\n' "$*"
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${VENV_DIR:-.venv_linux}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
CHECK_ONLY=0
PREPROCESS=0
SYSTEM_SITE_PACKAGES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      ;;
    --skip-torch)
      INSTALL_TORCH=0
      ;;
    --system-site-packages)
      SYSTEM_SITE_PACKAGES=1
      ;;
    --preprocess)
      PREPROCESS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -f /etc/profile.d/a100_cuda.sh ]]; then
  # Load server-level CUDA compatibility libraries when they exist.
  # This is required on the current A100 code-server image.
  # shellcheck source=/dev/null
  source /etc/profile.d/a100_cuda.sh
  log 'loaded /etc/profile.d/a100_cuda.sh'
fi

if [[ "$CHECK_ONLY" == "0" ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating virtual environment at $VENV_DIR"
    if [[ "$SYSTEM_SITE_PACKAGES" == "1" ]]; then
      "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    else
      "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
  else
    log "using existing virtual environment at $VENV_DIR"
  fi
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  printf 'Virtual environment not found at %s. Run without --check-only first.\n' "$VENV_DIR" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
log "python: $(python --version 2>&1)"
log "pip: $(python -m pip --version)"

if [[ "$CHECK_ONLY" == "0" ]]; then
  log 'upgrading pip'
  python -m pip install --upgrade pip ${PIP_EXTRA_ARGS}

  if [[ "$INSTALL_TORCH" != "0" ]]; then
    log "installing CUDA-compatible PyTorch from $PYTORCH_INDEX_URL"
    python -m pip install --force-reinstall torch --index-url "$PYTORCH_INDEX_URL" ${PIP_EXTRA_ARGS}
  else
    log 'skipping PyTorch reinstall because INSTALL_TORCH=0 or --skip-torch was provided'
  fi

  log 'installing project requirements'
  python -m pip install -r requirements.txt ${PIP_EXTRA_ARGS}

  log 'enforcing the paper-tested transformers version range'
  python -m pip install 'transformers>=4.38.0,<5.0.0' ${PIP_EXTRA_ARGS}

  if [[ "$PREPROCESS" == "1" ]]; then
    log 'running dataset preprocessing'
    python preprocess_dataset.py
  fi
fi

log 'validating Python imports and CUDA visibility'
python - <<'PY'
import importlib
import sys

checks = [
    ('torch', 'torch'),
    ('torch_geometric', 'torch_geometric'),
    ('transformers', 'transformers'),
    ('pandas', 'pandas'),
    ('numpy', 'numpy'),
    ('sklearn', 'sklearn'),
    ('networkx', 'networkx'),
    ('yaml', 'yaml'),
    ('flask', 'flask'),
]
missing = []
for label, module in checks:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((label, str(exc)))

try:
    import torch
    print('torch:', torch.__version__)
    print('cuda available:', torch.cuda.is_available())
    print('cuda count:', torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        print(f'cuda device {idx}:', torch.cuda.get_device_name(idx))
except Exception as exc:
    missing.append(('torch cuda check', str(exc)))

if missing:
    print('Missing or failing dependencies:')
    for label, exc in missing:
        print(f'- {label}: {exc}')
    sys.exit(1)

print('dependency check: ok')
PY

cat <<EOF

Environment ready.
Activate it with:
  cd "$REPO_ROOT"
  source "$VENV_DIR/bin/activate"

Recommended GPU runs on this server should also source:
  source /etc/profile.d/a100_cuda.sh
EOF