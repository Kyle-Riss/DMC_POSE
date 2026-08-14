#!/usr/bin/env bash
# pose 레포용 conda 환경: PyTorch(CUDA) + Ultralytics(YOLO) + Keras(TensorFlow) 등
set -euo pipefail

ENV_NAME="${POSE_CUDA_ENV_NAME:-pose-cuda}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQ="${REPO_ROOT}/requirements-pose-cuda.txt"

# PyTorch CUDA 빌드 (드라이버에 맞는 cu128 휠; 필요 시 https://pytorch.org 에서 인덱스 확인)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

if [[ ! -f "${REQ}" ]]; then
  echo "[error] requirements not found: ${REQ}" >&2
  exit 1
fi

if ! command -v conda &>/dev/null; then
  echo "[error] conda 가 PATH 에 없습니다. Miniconda/Anaconda 를 설치하거나 PATH 를 설정하세요." >&2
  exit 1
fi

echo "[info] creating conda env: ${ENV_NAME} (python 3.11)"
conda create -n "${ENV_NAME}" python=3.11 -y

# conda activate in script
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate pose-cuda로 활성화
conda activate "${ENV_NAME}"

echo "[info] upgrading pip"
python -m pip install --upgrade pip

echo "[info] installing PyTorch (CUDA) from ${TORCH_INDEX_URL}"
python -m pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

echo "[info] installing project requirements from ${REQ}"
python -m pip install -r "${REQ}"

echo ""
echo "[ok] environment '${ENV_NAME}' ready."
echo "     activate:  conda activate ${ENV_NAME}"
echo "     verify:    python -c \"import torch; print('cuda:', torch.cuda.is_available())\""
echo ""

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
