#!/usr/bin/env bash
# Safe SSH attempt to 192.168.0.161 — never hang forever.
# Usage: bash scripts/ssh_161_try.sh          # test only
#        bash scripts/ssh_161_try.sh login    # interactive if test OK
set -euo pipefail

HOST="${SSH_161_HOST:-192.168.0.161}"
USER="${SSH_161_USER:-dmc}"
WAIT_SEC="${SSH_161_WAIT_SEC:-0}"
TIMEOUT_SEC="${SSH_161_TIMEOUT_SEC:-15}"

if [[ "$WAIT_SEC" -gt 0 ]]; then
  echo "waiting ${WAIT_SEC}s (let .161 clear stale SSH slots)..."
  sleep "$WAIT_SEC"
fi

echo "=== step 1: ping $HOST ==="
ping -c 2 -W 2 "$HOST" | tail -2 || { echo "FAIL: no ping"; exit 1; }

echo
echo "=== step 2: local hung ssh (must be empty) ==="
if pgrep -af "[s]sh .*${HOST}" >/dev/null 2>&1; then
  echo "FAIL: hung ssh on THIS pc — Ctrl+C those terminals first:"
  pgrep -af "[s]sh .*${HOST}" || true
  exit 1
fi
echo "OK"

echo
echo "=== step 3: ssh probe (${TIMEOUT_SEC}s max) ==="
log=$(mktemp)
set +e
timeout "$TIMEOUT_SEC" ssh \
  -o ConnectTimeout=8 \
  -o GSSAPIAuthentication=no \
  -o PreferredAuthentications=password,keyboard-interactive \
  -o PubkeyAuthentication=no \
  -o BatchMode=yes \
  -vv "${USER}@${HOST}" exit 2>"$log"
rc=$?
set -e

if grep -q 'Exceeded MaxStartups' "$log"; then
  echo "RESULT: MaxStartups (ssh slots full on .161)"
  echo "NEXT:"
  echo "  1) do NOT retry ssh for 5-10 minutes"
  echo "  2) on .161 physically: sudo systemctl restart ssh"
  echo "  3) then: SSH_161_WAIT_SEC=300 bash scripts/ssh_161_try.sh"
  rm -f "$log"
  exit 2
fi

if grep -q 'SSH2_MSG_KEXINIT sent' "$log" && [[ $rc -eq 124 ]]; then
  echo "RESULT: KEX hang (sshd stuck during key exchange)"
  echo "NEXT:"
  echo "  1) Ctrl+C if you started ssh manually"
  echo "  2) on .161 physically: sudo systemctl restart ssh  (or reboot)"
  echo "  3) wait 2 min, single retry"
  rm -f "$log"
  exit 3
fi

if grep -qE 'Permission denied|Authenticated to' "$log"; then
  echo "RESULT: sshd reachable — password auth works"
  rm -f "$log"
  if [[ "${1:-}" == "login" ]]; then
    echo "=== step 4: interactive login ==="
    exec ssh -o ConnectTimeout=10 -o GSSAPIAuthentication=no "${USER}@${HOST}"
  fi
  echo "run: bash scripts/ssh_161_try.sh login"
  exit 0
fi

echo "RESULT: unknown (rc=$rc)"
tail -8 "$log"
rm -f "$log"
exit 4
