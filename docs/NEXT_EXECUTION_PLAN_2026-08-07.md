# DMC_POSE 실행 마스터 플랜

기준일: 2026-08-07  
대상 저장소: `/home/dmc/AI/DMC_POSE`  
최종 목표: 6개 병상에서 영상 모니터링을 방해하지 않으면서, 침대 자동 ROI·사람 추적·프레임 시계열·하이브리드 fusion을 이용해 낙상 후보를 검출하고 Raspberry Pi edge와 중앙 GPU 서버로 운영한다.

## 진행 상태 (2026-08-07 현장 검증)

- Phase 0: 완료
- Phase 1: 완료
- Phase 2: 진행 중 (시나리오 1~7 기본 전이 검증 완료)
- Phase 3 이후: 대기

최근 검증 결과:

- Bed Seg는 원본 카메라 방향에서 실행하고 ROI를 분석 좌표로 회전하도록 교정
- MobileSAM 3-point 저빈도 refinement 적용; 6대 매트리스 ROI 수치/육안 검증 통과
- `bed_162`의 베개 조각 ROI 문제 해결
- 침대 밖 서기: overlap `0.00~0.05`, `in_bed=NO`, fusion `SAFE`
- 가장자리 착석: seated joint geometry + mattress proximity로 `partial`, `in_bed=YES`
- 정상 눕기: overlap `0.83~0.91`, fusion `SAFE`
- 정상 눕기에서 legacy TCN shadow가 `0.997~0.999` false candidate를 냄; production 승격 보류 근거 확인
- 정상 이탈: fusion `VERIFY`까지만 진입 후 경보 없이 `SAFE/NO_PERSON` 복귀
- 단위 테스트 `124/124` 통과

## 1. 현재 기준선

### 구현 완료

- 6대 RTSP latest-frame capture와 독립 MJPEG viewer
- 분석 프레임 90도 회전과 원본 viewer 분리
- 자동 침대 segmentation consensus/cache
- 빈 병실 `EMPTY/idle` 저빈도 Pose probe
- motion watcher와 10초 pre-event ring
- 중앙 latest-only inference scheduler
- primary person tracking
- observed-only 10Hz, `30 x 109` TCN 입력
- track/gap reset 및 synthetic/copy row 금지
- 단일 프레임 모델 + TCN + bed relation hybrid fusion shadow
- 빈 병실 소형 장비 Pose 오탐 면적 gate 2.5%
- 단위 테스트 124개 통과

### 아직 승인되지 않은 항목

- TCN production 승격
- fusion 외부 경보 활성화
- 168 bed-hour 장시간 shadow gate
- 충분한 staged/actual fall sensitivity 측정
- FallVision temporal annotation과 안전한 subject/session split
- Raspberry Pi event uplink 구현
- systemd, readiness, event history, 인증을 포함한 운영 배포

## 2. 작업 원칙

1. 현재 동작하는 중앙 서버 경로를 기준선으로 보존한다.
2. 모델 학습과 라이브 입력은 동일한 observed-only 계약을 사용한다.
3. window 무작위 분할을 금지하고 subject/video/camera 누수를 막는다.
4. TCN 미준비와 TCN 오분류를 별도 지표로 계산한다.
5. 각 Phase의 산출물과 합격 기준을 충족하기 전 production alert를 켜지 않는다.
6. RPi 전환 전 중앙 서버에서 사건 계약과 현장 임계값을 먼저 동결한다.

## 3. 순차 실행 계획

## Phase 0 — 기준선 동결과 저장소 정리

목적: 현재 동작 상태를 잃지 않고 이후 결과를 재현할 수 있게 한다.

작업:

- 변경/신규 파일을 기능별로 분류한다.
- 모델 파일, 설정, 데이터 manifest, 런타임 플래그의 경로와 SHA-256을 기록한다.
- 데이터 다운로드 상태 검사는 read-only와 resume 명령으로 분리한다.
- 비밀정보가 포함된 RTSP URL과 계정정보의 추적 여부를 검사한다.
- 테스트를 다시 실행하고 기준 결과를 저장한다.
- 코드 스냅샷/커밋 후보 목록을 만든다. 실제 commit은 검토 가능한 단위로 나눈다.

산출물:

- `docs/runtime_artifact.json` 갱신
- 저장소 변경 분류표
- 모델/설정 체크섬 목록
- read-only 데이터 인벤토리
- 전체 테스트 리포트

합격 기준:

- 실행 파일과 사용 모델이 모호하지 않음
- 다운로드 확인 명령이 데이터를 변경하지 않음
- 테스트 실패 0
- 현재 기준선을 복구할 수 있음

## Phase 1 — 중앙 서버 자동 Smoke 검증

목적: 현장 시험 전에 6대 capture와 AI 구성요소가 정상인지 자동 판정한다.

작업:

- `run_all_cameras.sh`로 서버 기동
- 8000 listen, `/viewer`, `/status` 확인
- 6대별 capture/ROI/watcher/scheduler 상태 확인
- 빈 병실에서 10초 이상 다음 조건 검사
  - `person_count=0`
  - `runtime_mode=EMPTY`
  - `analysis_state=idle`
  - `tcn_samples=0`
  - scheduler error 0
- H.264 decode 오류 빈도와 frame age 기록

산출물:

- timestamp가 있는 smoke JSON/Markdown 리포트

합격 기준:

- 6대 연결 정상
- ROI 6대 모두 READY 또는 승인된 cache fallback
- 빈 방 Pose 오탐 0
- viewer와 status 갱신 정상

## Phase 2 — 현장 시나리오 Acceptance

목적: 단순 빈 방이 아니라 사람 행동 전체 상태 전이를 검증한다.

시나리오:

1. 빈 방 유지
2. 사람이 방에 들어옴
3. 침대 옆에 서 있음
4. 침대 가장자리에 앉음
5. 침대 중앙에 누움
6. 정상적으로 침대에서 나옴
7. 카메라 밖으로 퇴장
8. 빠르게 눕기
9. 물건 줍기/무릎 꿇기
10. 천천히 미끄러짐
11. 빠른 주저앉기
12. 모의 낙상 후 누워 있음
13. 모의 낙상 후 스스로 일어남
14. 의료진 부축 및 2인 동시 등장
15. 이불/가림 상태

각 시나리오 기록:

- capture timestamp와 camera ID
- person/track/bed overlap
- TCN samples/ready/probability
- motion/replay 상태
- fusion phase/risk/evidence
- 기대값과 실제값, PASS/FAIL

합격 기준:

- 사람 입장과 퇴장 상태 전이가 반복 재현됨
- 실제 사람이 3~4초 연속 관측되면 TCN ready
- 침대 밖 사람을 `없음`으로 처리하지 않음
- 사람이 사라지면 grace 후 EMPTY 복귀
- 정상 ADL이 경보로 승격되지 않음

## Phase 3 — 168 bed-hour Shadow 운영 계측

목적: 실제 오탐률과 사건 민감도를 frame accuracy가 아닌 event 단위로 측정한다.

작업:

- feature-only shadow recorder 상시 실행
- candidate ledger와 actual/staged event ledger 유지
- 모든 후보를 `true_fall/false_alarm/staged_fall/uncertain`으로 검토
- 카메라별 bed-hour와 장애 시간을 분리
- false alert, sensitivity, latency, TCN coverage를 매일 집계

시작용 gate:

- 누적 168 bed-hours 이상
- false alarms/bed-hour <= 0.01
- staged/actual fall sensitivity >= 0.90
- pending/uncertain 후보 0

## Phase 4 — FallVision 학습 자산 확정

목적: 영상 단위 라벨을 event-TCN 학습 가능한 temporal 데이터로 바꾼다.

작업:

- 모든 archive의 크기/체크섬/압축 테스트
- raw/mask/CSV 대응 canonical inventory 확정
- decode 불가/누락/filename mismatch 격리
- pilot annotation 검수
- fall onset/impact/post-fall stable/end 확정
- subject ID가 불명확하면 가장 보수적인 session/archive group 사용
- pre-onset context와 event-evaluable 여부 계산

합격 기준:

- annotation 완료 영상만 `temporal_tcn_eligible=true`
- split group 미해결 영상은 평가에서 제외
- train/val/test provenance 중복 0

## Phase 5 — 혼합 재학습과 고정 평가

데이터:

- GMDCSA observed-only
- FallVision temporal annotated
- 자체 병실 staged fall
- 자체 병실 hard-negative

실험:

- legacy checkpoint compatibility
- observed-only baseline
- mixed random initialization
- mixed warm start

고정 지표:

- pre-onset-ready coverage
- event-evaluable coverage
- conditional event recall
- end-to-end event recall
- event precision/F1
- false events per video-hour와 Poisson 95% CI
- latency from onset/impact

합격 기준:

- validation에서 operating point 선택 후 test 1회 적용
- 기존 baseline보다 event 수준에서 명확히 개선
- 현장 hard-negative 오탐이 허용 기준 이내

## Phase 6 — TCN/Fusion 승격

작업:

- checkpoint, mean/std, threshold, persistence, schema version 동결
- TCN 단독과 전체 fusion 성능 분리
- shadow A/B 비교
- rollback flag와 이전 checkpoint 보존

합격 기준:

- Phase 3과 Phase 5 gate 모두 통과
- 모델 입력 계약 불일치 0
- production 후보 버전이 하나로 특정됨

## Phase 7 — Raspberry Pi Edge 구현

RPi 역할:

- RTSP/local camera decode
- 축소 grayscale watcher
- 8~10초 pre-event ring
- trigger candidate/state machine
- heartbeat
- 이벤트 pre/live/post 프레임 업로드
- 서버 단절 시 로컬 spool

중앙 서버 역할:

- bed ROI, Pose, tracking, TCN, fusion
- event session 관리
- 결과/설정 하향 전송

필수 계약:

- `POST /edge/heartbeat`
- `POST /events/start`
- `WS /events/{event_id}/frames`
- `POST /events/end`
- event/camera/frame sequence와 capture timestamp

합격 기준:

- 평시 원본 영상 상시 업링크 없음
- trigger 이전 문맥 손실 없음
- 중복/역순 프레임 안전 처리
- 서버 단절 후 재전송 가능

## Phase 8 — 장애와 부하 시험

시험:

- 카메라별 RTSP 30초 차단/복구
- H.264 손상과 frame drop
- GPU 고의 지연
- ROI 검출 실패
- TCN 미로드/입력 gap
- 6개 병상 동시 이벤트
- RPi 네트워크 단절/복구
- 디스크 부족과 오래된 spool 정리

합격 기준:

- 한 병상 장애가 다른 병상에 전파되지 않음
- stale queue 누적 없음
- 데이터 유실/중복 정책이 문서와 일치
- 자동 복구 후 정상 상태 복귀

## Phase 9 — 운영 배포

작업:

- systemd 서비스와 자동 재시작
- `/health/live`, `/health/ready`
- event history/SSE
- Prometheus 지표
- 인증, 권한, RTSP secret 분리
- 로그 rotation과 저장 기간
- 배포/rollback/runbook

합격 기준:

- 재부팅 후 자동 복구
- readiness가 카메라/모델 장애를 정확히 구분
- 이벤트 근거와 model/ROI version 추적 가능

## Phase 10 — 최종 승인과 인수

작업:

- 현장 KPI 최종 검토
- production alert flag 활성화 승인
- 운영자 교육 자료와 장애 대응 문서
- 소스/모델/설정/체크섬 handoff ZIP

완료 조건:

- 기술 gate와 ML gate 모두 PASS
- 경보 전달 end-to-end 시험 PASS
- rollback 검증 PASS
- 미해결 P0/P1 결함 0

## 4. 즉시 실행 순서

```text
Phase 0 기준선 감사
  -> Phase 1 중앙 서버 smoke
  -> Phase 2 현장 acceptance
  -> Phase 3 shadow 수집 시작

Phase 3과 병행:
  Phase 4 FallVision 정리
  -> Phase 5 재학습
  -> Phase 6 승격 판정

승격 계약 동결 후:
  Phase 7 RPi
  -> Phase 8 장애/부하
  -> Phase 9 운영화
  -> Phase 10 승인
```

## 5. 현재 다음 행동

Phase 0을 시작한다.

1. Git 변경을 기능별로 분류한다.
2. 현재 실행 모델과 설정 체크섬을 기록한다.
3. 데이터 인벤토리를 read-only로 다시 만든다.
4. 전체 테스트를 실행한다.
5. Phase 0 기준선 리포트를 생성한다.

## 6. 2026-08-07 라이브 저자세 검증 결과

`bed_161`에서 서기, 통제된 쪼그리기, 바닥 앉기/눕기, 다시 일어나기를
충격 없이 수행했다. 검증 구간은 `2026-08-07T02:31:20.171526Z`부터
`2026-08-07T02:31:57.879328Z`까지다.

결과:

- 사람 관측 37.7초 연속 유지
- primary/TCN track ID 모두 1로 유지
- 관측 중 primary 누락 0회
- track switch 0회, TCN track reset 0회
- 실제 관측 30개 확보 후 약 31.8초간 TCN ready 유지
- 분석 프레임 age 최댓값 약 127ms
- TCN 단독 최대 낙상 확률 0.9833, candidate 기록 12개
- hybrid fusion은 모든 candidate를 SAFE로 억제하여 운영 경보 0개

판정:

```text
저자세 사람 검출: PASS
자세 전환 중 identity continuity: PASS
TCN observed-only ready 유지: PASS (초기 warming gap reset 1회는 별도 개선)
하드 네거티브에 대한 TCN 단독 분류: FAIL/재학습 필요
구조적 hybrid fusion 억제: PASS
```

근거 주석은
`runtime_data/annotations/hard_negative_sessions.jsonl`에 세션 단위
`NO_FALL`로 저장한다. 세부 행동별 시간 경계는 기록하지 않았으므로 이 세션을
frame-level 행동 라벨로 사용하지 않는다.

다음 구현 우선순위:

1. [구현 완료] synchronous pre-event replay를 카메라 live analysis thread 밖으로 이동한다.
2. [보강 완료] replay 작업을 카메라당 하나로 제한하고 P3 우선순위와 10초 cooldown을 적용한다.
3. [검증 대기] 실제 motion replay 중에도 live 10Hz Pose/TCN cadence가 유지되는지 확인한다.
4. 본 세션과 추가 저자세 세션을 hard-negative 재학습/threshold calibration에 사용한다.

비동기 replay 구현 검증:

- `AsyncReplayWorker`: 카메라당 single-flight daemon 작업
- shared Keras predictor: 모든 카메라/replay predict 호출 직렬화
- 최초 라이브 시험에서 P2 replay가 live Pose와 경쟁하여 TCN ready reset 재현
- replay scheduler priority: P3_EMPTY_PROBE로 추가 하향
- motion/presence replay 공통 카메라별 10초 cooldown 적용
- P3 재시험에서 shared Keras batch lock으로 최대 431ms live 지연과 gap reset 재현
- live 1-frame Keras와 replay batch Keras 모델/lock을 별도 인스턴스로 분리
- 단일 서버 재시험에서도 최대 분석 지연 약 277ms가 남아 운영 기본값은 replay OFF로 결정
- replay 구현은 기능 플래그로 보존하며 더 강한 서버/별도 프로세스에서만 재검증
- `tcn_gap_reset_total`은 live-only로 고정하고 replay counter를 별도 노출
- graceful shutdown 중 구 서버가 분석을 지속해 이중 서버가 된 사례를 재현
- `run_all_cameras.sh`에 수명 전체를 유지하는 non-blocking `flock` 단일 인스턴스 가드 추가
- 문법 검사 통과
- 전체 단위 테스트 132개 통과
- 재시작 후 6대 capture 연결/자동 ROI READY, scheduler pending 0
