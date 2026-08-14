# Pilot Deployment Readiness — 2026-08-14

## Decision

The current system is approved for a supervised pilot or shadow-monitoring
deployment.  It is not approved as a standalone patient-safety device and must
not replace staff observation, existing alarms, or an established emergency
response process.

## Verified in the current six-camera runtime

| Check | Result | Evidence |
|---|---|---|
| Six camera endpoints | PASS | All six `/image/{camera_id}` requests returned HTTP 200 JPEG |
| Live video path | PASS | Public gateway received a continuous multipart video stream |
| Public viewer | PASS | LAN `/viewer` returned HTTP 200 |
| Minimal public status | PASS | Only `camera_id`, `state`, and `updated_at` are returned |
| Public API documentation disabled | PASS | `/docs` and `/openapi.json` return HTTP 404 |
| Core network isolation | PASS | Core binds `127.0.0.1:18000`; gateway binds `0.0.0.0:8000` |
| Managed startup | PASS | Monitoring target and core/gateway/control services are active |
| Fall-event processing path | DEMONSTRATED | Staged live event path was exercised; population-level performance is not established |

## Required before standalone operational use

| Gate | Required evidence |
|---|---|
| Long-run stability | At least 72 hours per deployment profile with reconnect and resource metrics |
| Camera failure behavior | Offline/stale camera must create an explicit operational alarm |
| Event performance | Subject- and session-disjoint event recall, precision, latency, and false alerts per valid bed-hour |
| Hard-negative coverage | Sitting, crouching, object pickup, fast lying, blanket occlusion, staff assistance, and multiple people |
| Alert delivery | End-to-end alert receipt, acknowledgement, escalation, and audit trail |
| Model/data drift | Per-room acceptance test and periodic review after camera or furniture changes |
| Privacy/access | Authentication, transport protection, least privilege, retention policy, and access audit |
| Safety process | Written response procedure and a named human responder for every alert |

## Permitted use now

```text
camera monitoring
  -> protected inference
  -> shadow/supervised alert
  -> human verification
  -> existing response procedure
```

The output is decision support, not a guaranteed diagnosis or a substitute for
human supervision.
