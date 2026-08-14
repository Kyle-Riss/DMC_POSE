#!/usr/bin/env python3
"""Run monitor once, evaluate thresholds, log alerts, and optionally POST to webhook.
Adds: retries, exponential backoff, and simple rate-limited deduplication to avoid alert storms.
Environment:
  MONITOR_WEBHOOK=http://example.com/webhook (optional)
  MONITOR_FPS_THRESHOLD=5
  MONITOR_FRAME_AGE_MS=1000
  MONITOR_WEBHOOK_RETRIES=3
  MONITOR_WEBHOOK_TIMEOUT=3
  MONITOR_WEBHOOK_RATE_LIMIT_SEC=60  # suppress same (camera,type) alerts during this window
"""
import os, subprocess, csv, time, json, requests
from datetime import datetime
BASE = os.path.dirname(__file__)
MONITOR_SCRIPT = os.path.join(BASE, 'monitor_cameras.py')
LOG = os.path.join(BASE, 'log.csv')
ALERTS = os.path.join(BASE, 'alerts.log')
STATE_FILE = os.path.join(BASE, 'last_alerts.json')

WEBHOOK = os.environ.get('MONITOR_WEBHOOK')
FPS_THRESH = float(os.environ.get('MONITOR_FPS_THRESHOLD', '5'))
AGE_THRESH = float(os.environ.get('MONITOR_FRAME_AGE_MS', '1000'))
WEBHOOK_RETRIES = int(os.environ.get('MONITOR_WEBHOOK_RETRIES', '3'))
WEBHOOK_TIMEOUT = float(os.environ.get('MONITOR_WEBHOOK_TIMEOUT', '3'))
RATE_LIMIT_SEC = int(os.environ.get('MONITOR_WEBHOOK_RATE_LIMIT_SEC', '60'))


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_state(st):
    try:
        with open(STATE_FILE, 'w') as fh:
            json.dump(st, fh)
    except Exception as e:
        print('Failed to save state', e)


# Run monitor script (one-shot)
ret = subprocess.run([MONITOR_SCRIPT], cwd=BASE)
# Read last rows from log.csv
if not os.path.exists(LOG):
    print('No log found')
    raise SystemExit(0)

# read CSV and get latest per camera
latest = {}
with open(LOG, 'r') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        cid = row.get('camera_id')
        if not cid:
            continue
        latest[cid] = row

alerts = []
for cid, row in latest.items():
    try:
        fps = float(row['pipeline_fps']) if row.get('pipeline_fps') not in (None,'','None') else None
    except Exception:
        fps = None
    try:
        age = float(row['frame_age_ms']) if row.get('frame_age_ms') not in (None,'','None') else None
    except Exception:
        age = None
    tcp = row.get('tcp_connect_ms','')
    note = row.get('note','')
    if tcp=='' or str(tcp).lower()=='none':
        alerts.append((cid,'tcp_fail',row))
    if fps is not None and fps < FPS_THRESH:
        alerts.append((cid,'low_fps',row))
    if age is not None and age > AGE_THRESH:
        alerts.append((cid,'old_frame',row))
    if note:
        alerts.append((cid,'note',row))

# Load last-alert timestamps to deduplicate
state = load_state()
now_ts = int(time.time())
filtered_alerts = []
updated = False
for a in alerts:
    key = f"{a[0]}::{a[1]}"  # camera::type
    last = int(state.get(key, 0))
    if now_ts - last >= RATE_LIMIT_SEC:
        filtered_alerts.append(a)
        state[key] = now_ts
        updated = True
    else:
        # suppressed by rate limit
        print(f"Suppressed alert {key}; last sent {now_ts-last}s ago")

if filtered_alerts:
    ts = datetime.utcnow().isoformat()+'Z'
    with open(ALERTS, 'a') as af:
        for a in filtered_alerts:
            line = f"{ts}\t{a[0]}\t{a[1]}\t{json.dumps(a[2])}\n"
            af.write(line)
    print(f"{len(filtered_alerts)} alerts logged to {ALERTS}")

    # persist state if updated
    if updated:
        save_state(state)

    # Send webhook with retries/backoff
    if WEBHOOK:
        payload = {'ts': datetime.utcnow().isoformat()+'Z', 'alerts': []}
        for a in filtered_alerts:
            payload['alerts'].append({'camera': a[0], 'type': a[1], 'row': a[2]})

        attempt = 0
        while attempt < WEBHOOK_RETRIES:
            attempt += 1
            try:
                r = requests.post(WEBHOOK, json=payload, timeout=WEBHOOK_TIMEOUT)
                print('Webhook posted, status', r.status_code)
                if r.status_code >= 200 and r.status_code < 300:
                    break
                else:
                    print('Non-2xx response, retrying...')
            except Exception as e:
                print('Webhook attempt failed', e)
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)
        else:
            print('Webhook failed after retries')
else:
    print('No alerts or all alerts suppressed by rate-limit')
