# 낙상 시계열 데이터 통합 실행 기준

기준일: 2026-08-03  
Canonical runtime contract: architecture_v2/11_FIXED_CAMERA_MODULE_CONTRACT.md

## 고정된 경계

- 공개 영상은 모두 yolo11m-pose.pt로 다시 추출한다.
- 기존 제공 landmark는 메타데이터와 라벨 대응 확인에만 사용한다.
- TCN v1에는 실제로 관측된 동일 primary Pose만 들어간다.
- 사람 전체 미검출 시 행을 삽입하지 않고 이전 skeleton도 복사하지 않는다.
- 관측 간격 70~150ms만 이어 붙이며 초과 gap과 track change에서 reset한다.
- 일부 관절 미검출은 joint visibility 0으로 표현할 수 있다.
- BED_EXIT와 BED_EXIT_FALL은 TCN label이 아니라 Bed ROI 기반 fusion 사건이다.

## 실행 순서

1. config/temporal_contract_v2.json과 모델 hash를 검증한다.
2. FallVision fall/non-fall raw archive를 각각 하나씩 smoke 해제한다.
3. recording ID, subject ID, 파생본 중복, event interval 출처를 감사한다.
4. dataset별 temporal_manifest_v2를 만든다.
5. offline extractor와 live runner의 feature/window parity test를 통과시킨다.
6. GMDCSA-24를 다시 추출해 GMDCSA-only v2 baseline을 학습한다.
7. 학습에 사용하지 않은 FallVision frozen subset으로 외부평가한다.
8. FallVision train subset을 섞은 scratch/warm-start 모델을 비교한다.
9. 자체 병실 hard-negative와 staged fall replay로 fusion을 검증한다.

## FallVision 현재 감사 상태

- 공식 파일 60개와 checksum 60개가 일치한다.
- 실제 원본 영상 archive는 20개이며 나머지 40개는 mask/keypoint 관련이다.
- 전체 archive 목록 감사 결과 60/60개를 열었고 오류는 0개였다. 총 17,656개
  엔트리는 MP4 11,731개, CSV 5,864개, 보조 Python 1개다.
- repository 경로와 archive 이름에서 `fall/no-fall` 및 `bed/chair/stand` 영상 단위
  라벨을 확정할 수 있다.
- keypoint CSV 헤더는 `Frame,Keypoint,X,Y,Confidence`이며 17개 COCO keypoint의
  프레임별 좌표와 confidence를 제공한다.
- 배포본 수량은 raw MP4 5,866개, mask MP4 5,865개, keypoint CSV 5,864개다. 논문의
  총 MP4 11,732개 주장과 비교하면 mask MP4가 1개 부족하다.
- `C_M_223`은 keypoint CSV가 없고, `S_D_058`은 mask/keypoint가 없으며,
  `S_D_00462`(raw) 대 `S_D_0046`(mask/CSV)은 파일명 오타 후보라 자동 pairing에서
  제외하고 수동 확인한다.
- onset, impact, fall end, subject/participant mapping 또는 별도 ground-truth 이름의
  파일은 전체 archive 목록에서 발견되지 않았다.
- smoke archive는 f_raw_b_3.rar, nf_raw_b_2.rar다.
- 파일명의 숫자는 recording 번호이며 58명의 명시적 subject-ID 매핑으로 확인되지
  않았으므로 검증 없이 subject ID로 쓰지 않는다.
- resized, anonymized 파생본은 같은 recording group으로 묶어 split 누수를 막는다.
- `video_classification_eligible=true`, `temporal_tcn_eligible=false`,
  `subject_disjoint_split_ready=false`로 능력을 분리한다. 기존
  `training_eligible=false`는 temporal TCN aggregate gate로만 유지한다.
- 재현 가능한 목록 감사 결과는
  `external_datasets/manifests/fallvision_archive_audit.json`에 저장한다.

## FallVision canonical pairing 및 annotation pilot

- 원본 RAR은 수정하지 않고 raw/mask/CSV member 경로를 canonical inventory에 보존한다.
- union recording 5,869개 중 `complete` 5,861개, filename mismatch candidate 6개,
  `missing_csv` 1개, `raw_only` 1개다. mismatch 6개는
  `B_N_336 - Copy_resized`↔`B_N_541`, `B_N_337 - Copy_resized`↔`B_N_542`,
  `S_D_00462`↔`S_D_0046`의 3개 후보 쌍이며 사람이 확인하기 전에는 병합하지 않는다.
- alias table은 `activity_label, scene_id, chunk_id, raw_id`를 복합 키로 사용해 다른
  class의 동일 recording 이름이 잘못 연결되지 않게 한다.
- provenance 단위 RAR group은 `provisional_split_group`으로만 기록한다. 동일 참가자가
  다른 archive에 있을 가능성을 배제할 수 없어 `split_group_resolved=false`다.
- 영상별 temporal eligibility는 raw/decode, complete annotation, resolved split group,
  observed Pose, sufficient pre-context가 모두 true일 때만 true가 된다.
- annotation pilot은 complete pair에서 bed/chair/stand 각각 8개, 총 24개를 뽑았다.
  24개 영상 모두 추출과 decode에 성공했으며 원본 SHA-256을 annotation CSV에
  기록했다.
- 24개 pilot은 모두 수동 검수 완료됐다. bed/chair/stand 각 8개이며 annotation
  ordering 및 frame bounds 오류는 0개다. 완성본은
  `external_datasets/annotations/fallvision_pilot_v1_complete.csv`에 동결했다.
- 자동 후보는 global frame motion, 골반 중심 이동, 몸통/박스 형태 변화를 결합한다.
  후보 프레임만 제안하며 ground truth에는 사람 확인이 필요하다.
  annotation schema는 `config/fallvision_temporal_annotation_v1.schema.json`이다.
- 자동 후보는 수동 CSV와 별도 파일에 저장하고 UI에서 읽기 전용으로 병합한다.
  `Apply proposal`은 입력란에 복사만 하며 자동 저장하거나 complete로 승격하지 않는다.

```bash
/home/dmc/anaconda3/envs/pose-cuda/bin/python \
  scripts/serve_fallvision_annotation.py --port 8010
```

브라우저에서 서버 로컬은 `http://127.0.0.1:8010`, LAN에서는
`http://192.168.0.108:8010`을 열어 프레임을 지정한다. `complete` 저장은
onset, impact, stable, end가 모두 있고 시간 순서가 맞을 때만 허용된다.

## FallVision 자동 후보 pilot 결과

24개 수동 라벨을 대상으로 한 작은 pilot 결과이므로 일반화 성능으로 해석하지 않는다.
각 held-out 영상은 나머지 23개에서 파라미터를 선택하는 leave-one-video-out 방식으로
측정했다.

| signal | onset MAE | impact MAE | stable MAE | 사용 판정 |
| --- | ---: | ---: | ---: | --- |
| frame motion + 현재 YOLO11m Pose | 0.452초 | 0.455초 | 0.185초 | 검토 후보로 사용 |
| FallVision 제공 keypoint CSV | 0.638초 | 0.703초 | 0.347초 | 무결성/선별 보조만 |

- 현재 YOLO 결합 후보는 `fallvision_pilot_v1_proposals.csv`, 상세 보고서는
  `fallvision_pilot_v1_proposals_report.json`에 저장한다.
- 제공 keypoint archive 20개를 모두 해제했고 CSV 5,864개를 확인했다. 추출 보고서는
  `external_datasets/fallvision/provided_keypoints_extraction_report.json`이다.
- 제공 keypoint는 현재 extractor와 정의·confidence 분포가 다르고 impact 오차도 더
  크므로 전체 데이터의 temporal ground truth로 자동 승격하지 않는다.
- 다음 확장은 자동 후보로 검수 시간을 줄인 추가 층화 표본을 만든 뒤, 독립 표본에서
  오차를 다시 측정하는 active-review 방식으로 진행한다.

## 평가 분리

legacy_checkpoint_on_v2_features는 입력 호환성 실험이다. 깨끗한 외부 일반화
baseline은 새 계약으로 다시 추출한 GMDCSA-only v2 모델을 frozen FallVision에
평가한 결과다. 혼합 학습에서는 frozen external test recording을 제외한다.

## 2026-08-03 구현 및 검증 결과

- `temporal_sequence.decide_observation`을 라이브/오프라인 공통 cadence 판정기로 고정했다.
- 라이브 primary track 전환 시 카메라의 TCN runner cache 전체를 비운다.
- GMDCSA manifest를 `temporal_manifest_v2`로 재생성했다: 영상 160개, 오류 0개.
- 새 extractor는 sequential decode, 실제 Pose-only, 동일 track, 70~150ms 계약으로 동작한다.
- 한 영상 smoke 결과: 203 decode frames, 68 Pose probes, 실제 관측 55행, 8 sequences, synthetic missing rows 0개.
- 이 smoke CSV에서 경계를 넘지 않는 30행/109-feature window 생성까지 확인했다.
- 전체 unit test 86개 및 `git diff --check`를 통과했다.

FallVision의 영상 분류 라벨과 제공 keypoint 시계열은 즉시 활용 가능하다. 다만
subject mapping과 낙상 event interval이 확보되기 전까지 temporal TCN gate인
`temporal_tcn_eligible=false`를 유지하며, 영상 단위 fall label을 30행 window
label로 확대 사용하지 않는다.

## GMDCSA v2 baseline 결과와 승격 결정

- 재추출: 160/160 성공, 실제 Pose 관측 11,156행, 계약 오류 0개.
- 전체 Pose probes 12,901회 중 primary 미관측 506회는 행을 만들지 않았다.
- gap reset 1,254회, track reset 9회이며 40개 영상은 연속 30행 구간이 없었다.
- 유효 window: train 313, validation 106, test 176.
- window test: AUROC 0.8645, recall 0.7297, F1 0.5684.
- 원래 operating point의 event test: precision 0.4667, end-to-end recall 0.4667,
  false event 8건/0.080295시간, 99.63건/시간(95% Poisson CI 43.01~196.32).
- event-evaluable coverage는 validation 0.3333, test 0.7333이었다.
- 더 엄격한 pre-onset-ready coverage는 validation 0.0952, test 0.0이었다. 즉 Test에는
  낙상 onset 전에 30개 관측을 확보하고 onset까지 track이 살아 있던 사건이 없다.

따라서 `gmdcsa24_tcn_v2_observed_only`는 계약 검증용 baseline으로 보존하되
현재 라이브 기본 가중치로 승격하지 않는다. 기존 기본 경로도 이번 단계에서는
바꾸지 않았다. 다음 승격 후보는 낙상 전 3초 이상 문맥이 보장되는 데이터와
독립 event annotation을 포함해 다시 학습·평가해야 한다.

## Event 지표 정의

- `pre_onset_ready_coverage`: onset 이전까지 동일 sequence의 유효 30행을 확보했고,
  마지막 관측이 onset에서 150ms 이내인 사건 비율이다.
- `event_evaluable_coverage`: GT fall interval 안에서 끝나는 유효 30행 window가 하나
  이상 존재하는 사건 비율이다.
- `conditional_event_recall`: event-evaluable 사건 중 검출한 사건 비율이다.
- `end_to_end_event_recall`: 전체 GT 사건 중 검출한 사건 비율이다.
- latency는 onset과 impact 기준을 분리한다. GMDCSA에는 독립 impact annotation이
  없으므로 impact latency는 null이다.
- 오탐은 raw count, 정확한 평가 시간, 시간당 비율, Poisson 95% 신뢰구간을 함께
  기록한다.

## Legacy 대 v2 observed-only 호환성 비교

두 checkpoint에 완전히 같은 Phase 10 raw window를 주고 각 checkpoint의 자체
mean/std를 적용했다. Validation에서 threshold와 persistence를 선택한 뒤 Test에는
고정값을 한 번만 적용했다.

| checkpoint | frozen threshold / persistence | Test precision | Test E2E recall | Test conditional recall | Test FP / hours | Test FP/hour (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy | 0.27 / 2 | 0.5294 | 0.6000 | 0.8182 | 8 / 0.080295 | 99.63 (43.01~196.32) |
| v2 observed-only | 0.40 / 2 | 0.5000 | 0.5333 | 0.7273 | 8 / 0.080295 | 99.63 (43.01~196.32) |

Legacy가 이 Test에서는 조금 우세하지만 이는 새 운영 입력에 대한 checkpoint
호환성 비교다. Legacy의 과거 training preprocessing 계약은 확인되지 않았으며,
두 모델 모두 운영 불가능한 오탐률을 보였다. 특히 Test의 pre-onset-ready GT가
0건이므로 실제 live-ready 상태의 recall을 검증하지 못했다. 따라서 어느
checkpoint도 승격하지 않고 현재 라이브 기본 경로도 변경하지 않는다.

재현 아티팩트는 `runs/temporal_tcn/observed_only_checkpoint_comparison.json`이며,
다음 gate는 행동 전 8~10초가 포함된 자체 병실 자료와 subject/temporal annotation이
완성된 FallVision subset으로 동일 평가를 반복하는 것이다.

## FallVision weak-label train-only 실험

자동 후보는 ground truth로 승격하지 않고 training augmentation으로만 제한했다.
낙상 시작 후보부터 impact 직전까지는 `ignore`, impact부터 제안 종료까지는 `fall`로
두어 애매한 transition이 `non_fall` 오답으로 들어가지 않게 했다. `ignore`로 끝나는
window는 학습에서 제외하며, context 내부의 ignore row는 실제 Pose 관측이므로 유지한다.

- weak fall: chunk 1, bed/chair/stand 각 24개, 총 72개 영상. 72/72 추출 성공.
- official non-fall: chunk 1, 각 12개, 총 36개 영상. 36/36 추출 성공.
- 3초 window: fall 65개/32개 영상, non-fall 111개/30개 영상.
- 혼합 train: GMDCSA 313 + FallVision 176 = 489 windows
  (`non_fall=323`, `fall=166`).
- GMDCSA validation 106개와 test 176개 NPZ SHA-256은 혼합 전후 동일하다.

| model | GMDCSA Test event precision | E2E recall | conditional recall | false events/hour |
| --- | ---: | ---: | ---: | ---: |
| v2 baseline | 0.5000 | 0.5333 | 0.7273 | 99.63 |
| mixed scratch | 0.3750 | 0.6000 | 0.8182 | 186.81 |
| mixed warm-start | 0.4000 | 0.5333 | 0.7273 | 149.45 |

Scratch는 전체 낙상 15건 중 한 건을 더 찾았지만 false event가 8건에서 15건으로
늘었다. Warm-start도 baseline보다 false event가 많다. 따라서 두 혼합 checkpoint는
모두 승격하지 않고 실험 아티팩트로만 보존한다.

재현 경로:

- `external_datasets/manifests/fallvision_weak_train_v1.json`
- `external_datasets/manifests/fallvision_non_fall_train_v1.json`
- `external_datasets/windows/tcn_109_v2_no_missing/gmdcsa24_fallvision_weak_v1_3s`
- `runs/temporal_tcn/fallvision_weak_v1_baseline_vs_scratch_event.json`
- `runs/temporal_tcn/fallvision_weak_v1_scratch_vs_warm_event.json`

## FallVision 수동 pilot 인도메인 진단

수동 완료 fall 24개와 학습에 쓰지 않은 chunk 2 official non-fall 24개를 합쳐
diagnostic-only test를 만들었다. subject identity가 해결되지 않았고 양성 pilot은
자동 후보 calibration에도 쓰였으므로 promotion 지표가 아니다. 운영점은 GMDCSA
validation에서 선택한 값을 그대로 고정했다.

- 48개 영상 모두 동일 YOLO11m Pose로 재추출 성공.
- 3초 window는 64개(`fall=16`, `non_fall=48`), window가 있는 영상은 25개.
- fall 24건 중 event-evaluable은 9건(37.5%), pre-onset-ready는 0건.
- baseline, mixed scratch, mixed warm-start 모두 evaluable 9건 중 2건만 검출했다.
- false event는 각각 1, 4, 2건이었다.

이 결과는 현재 병목이 단순 checkpoint가 아니라 **낙상 전에 30개 실제 Pose 관측을
확보하지 못하는 짧은 clip/context 구조**임을 확인한다. 다음 학습은 더 많은 짧은
FallVision clip을 자동 확대하는 방향이 아니라 다음 조건을 먼저 만족해야 한다.

1. 자체 병실 또는 연속 녹화에서 행동 전 8~10초를 포함한다.
2. motion watcher는 짧은 급격 움직임을 즉시 포착하고 TCN은 ready일 때 확인기로 쓴다.
3. TCN 미준비 상태를 정상 판정으로 취급하지 않고 fusion의 별도 상태로 유지한다.
4. 독립 subject/session split과 긴 hard-negative bed-hour로 FP/hour를 다시 측정한다.

진단 보고서는
`runs/temporal_tcn/fallvision_pilot_balanced_diagnostic_v1_report.json`에 있다.
