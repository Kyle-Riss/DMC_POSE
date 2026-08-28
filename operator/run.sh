#!/usr/bin/env bash
set -euo pipefail

action="${1:-start}"
services=(company-core.service company-edge.service company-gateway.service)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'HELP'
Usage: ./run.sh <command>

Read-only:
  help                 Show this help
  viewer               Print the configured Viewer URL
  health               Check the public health endpoint
  doctor               Check service health and all six camera RTSP TCP ports
  status               Check whether the service is ready
  shadow-status        Show detailed central shadow telemetry (sudo)
  shadow-plan          Validate the 20 Hz / 80-row deployment inputs
  shadow-plan-10hz     Validate the 10 Hz / 40-row deployment inputs

State-changing (sudo):
  start | stop | restart
  deploy-shadow        Deploy the 20 Hz telemetry-only GRU shadow
  deploy-shadow-10hz   Deploy the 10 Hz telemetry-only GRU shadow
HELP
}

ready() {
    curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8020/health/live >/dev/null 2>&1 \
        && curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:8030/health >/dev/null 2>&1 \
        && curl --noproxy '*' -fsS --max-time 20 http://127.0.0.1:8030/api/cameras >/dev/null 2>&1
}

wait_until_ready() {
    for _ in $(seq 1 120); do
        if ready; then
            echo "Service ready: http://$(hostname -I | awk '{print $1}'):8030/viewer"
            return 0
        fi
        sleep 1
    done
    echo "Service is not ready. Check again with: ./run.sh status" >&2
    return 1
}

case "$action" in
    help|-h|--help)
        usage
        ;;
    viewer)
        /usr/bin/python3 "$script_dir/tools/viewer_url.py"
        ;;
    health)
        /usr/bin/python3 "$script_dir/tools/health_check.py"
        ;;
    doctor)
        failed=0
        /usr/bin/python3 "$script_dir/tools/health_check.py" || failed=1
        for host in 192.168.0.161 192.168.0.162 192.168.0.174 192.168.0.175 192.168.0.178 192.168.0.179; do
            /usr/bin/python3 "$script_dir/tools/network_probe.py" "$host" 8554 || failed=1
        done
        /usr/bin/python3 "$script_dir/tools/viewer_url.py"
        exit "$failed"
        ;;
    start)
        sudo systemctl start "${services[@]}"
        wait_until_ready
        ;;
    stop)
        sudo systemctl stop "${services[@]}"
        echo "Service stopped."
        ;;
    restart)
        sudo systemctl restart "${services[@]}"
        wait_until_ready
        ;;
    status)
        if ready; then
            echo "Service ready: http://192.168.0.108:8030/viewer"
        else
            echo "Service not ready."
            exit 1
        fi
        ;;
    deploy-shadow)
        sudo /usr/bin/python3 "$script_dir/tools/deploy_central_gru_shadow.py"
        wait_until_ready
        ;;
    deploy-shadow-10hz)
        sudo /usr/bin/python3 "$script_dir/tools/deploy_central_gru_shadow_10hz.py"
        wait_until_ready
        ;;
    shadow-plan)
        /usr/bin/python3 "$script_dir/tools/deploy_central_gru_shadow.py" --plan
        ;;
    shadow-plan-10hz)
        /usr/bin/python3 "$script_dir/tools/deploy_central_gru_shadow_10hz.py" --plan
        ;;
    shadow-status)
        sudo /usr/bin/python3 "$script_dir/tools/check_central_gru_shadow.py"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
