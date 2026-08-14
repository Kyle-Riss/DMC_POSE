#!/usr/bin/env python3
"""Simple camera connectivity and latency monitor.
Writes CSV lines to monitor/log.csv and prints a table.
"""
import socket, time, json, csv, requests, os
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
CONFIG = os.path.join(BASE_DIR, 'cameras.json')
LOG = os.path.join(BASE_DIR, 'log.csv')
SERVER_STATUS_URL = 'http://localhost:8000/status'

with open(CONFIG, 'r') as f:
    cameras = json.load(f)

# Ensure log header
if not os.path.exists(LOG):
    with open(LOG, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['ts','camera_id','host','port','tcp_connect_ms','server_online','pipeline_fps','frame_age_ms','note'])


def tcp_connect_ms(host, port, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    t0 = time.perf_counter()
    try:
        s.connect((host, port))
        dt = (time.perf_counter() - t0) * 1000.0
        s.close()
        return round(dt, 1)
    except Exception as e:
        return None


def fetch_server_status():
    try:
        r = requests.get(SERVER_STATUS_URL, timeout=1.0)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def sample_once():
    ts = datetime.utcnow().isoformat() + 'Z'
    status = fetch_server_status()
    rows = []
    for cam in cameras:
        cid = cam['id']
        host = cam['host']
        port = cam.get('port', 8554)
        tcp_ms = tcp_connect_ms(host, port, timeout=1.0)
        server_online = False
        pipeline_fps = None
        frame_age_ms = None
        note = ''
        if status is not None and cid in status:
            server_online = True
            cinfo = status[cid]
            pipeline_fps = cinfo.get('pipeline_fps')
            ts_cam = cinfo.get('timestamp')
            try:
                if ts_cam:
                    # compute age in ms — accept Z suffix or naive ISO by normalizing to UTC
                    try:
                        ts_norm = ts_cam
                        if ts_norm.endswith('Z'):
                            ts_norm = ts_norm.replace('Z', '+00:00')
                        # if no offset present, assume UTC
                        if ts_norm[-6] not in ['+','-']:
                            ts_norm = ts_norm + '+00:00'
                        age = (datetime.utcnow() - datetime.fromisoformat(ts_norm)).total_seconds() * 1000.0
                        frame_age_ms = round(age, 1)
                    except Exception:
                        frame_age_ms = None
            except Exception:
                frame_age_ms = None
        if tcp_ms is None:
            note = 'tcp-conn-failed'
        rows.append((ts,cid,host,port,tcp_ms,server_online,pipeline_fps,frame_age_ms,note))
    # append to CSV
    with open(LOG, 'a', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)
    # print summary
    print(f"Sample @ {ts}")
    print("cam\tconnect_ms\tserver_on\tfps\tframe_age_ms\tnote")
    for r in rows:
        print(f"{r[1]}\t{r[4]}\t{r[5]}\t{r[6]}\t{r[7]}\t{r[8]}")


if __name__ == '__main__':
    print('Running one-shot monitor sample...')
    sample_once()
    print('Done. log ->', LOG)
