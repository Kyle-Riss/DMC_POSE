#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
manifest="${2:-}"
run_name="${3:-}"
epochs="${4:-3}"
source_root=/home/dmc/AI/DMC_POSE_source
python_bin=/home/dmc/anaconda3/envs/pose-cuda/bin/python

usage() {
    echo "Usage:" >&2
    echo "  ./site-finetune.sh check <manifest.json> [run-name]" >&2
    echo "  ./site-finetune.sh train <manifest.json> <run-name> [epochs]" >&2
}

if [[ -z "$mode" || -z "$manifest" ]]; then
    usage
    exit 2
fi

case "$mode" in
    check)
        run_name="${run_name:-site_candidate_check}"
        exec "$python_bin" "$source_root/scripts/finetune_swin3d_site.py" \
            --manifest "$manifest" \
            --out-dir "$source_root/runs/video_verifier/sites/$run_name" \
            --validate-only
        ;;
    train)
        if [[ -z "$run_name" ]]; then
            usage
            exit 2
        fi
        exec "$python_bin" "$source_root/scripts/finetune_swin3d_site.py" \
            --manifest "$manifest" \
            --out-dir "$source_root/runs/video_verifier/sites/$run_name" \
            --epochs "$epochs" \
            --device cuda
        ;;
    *)
        usage
        exit 2
        ;;
esac
