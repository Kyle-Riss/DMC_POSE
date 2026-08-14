import {
  Button,
  Callout,
  Divider,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Table,
  Text,
  computeDAGLayout,
  mergeStyle,
  useCanvasState,
  useHostTheme,
  type CanvasHostTheme,
} from "cursor/canvas";

type ViewId = "setup" | "runtime" | "rawbatch" | "impl";
type NodeStatus = "done" | "current" | "planned" | "partial";

type FlowNodeDef = {
  id: string;
  title: string;
  subtitle?: string;
  status: NodeStatus;
};

type FlowDef = {
  nodes: FlowNodeDef[];
  edges: Array<{ from: string; to: string }>;
  direction?: "vertical" | "horizontal";
  nodeWidth?: number;
  nodeHeight?: number;
};

const VIEW_OPTIONS: Array<{ id: ViewId; label: string }> = [
  { id: "setup", label: "현장 셋업" },
  { id: "runtime", label: "프레임 파이프라인 (A→G)" },
  { id: "rawbatch", label: "Raw 배치 시계열" },
  { id: "impl", label: "코드 Phase 0→5" },
];

const RAW_BATCH_FLOW: FlowDef = {
  direction: "vertical",
  nodeWidth: 208,
  nodeHeight: 52,
  nodes: [
    { id: "mp4", title: "Raw MP4 ×8", subtitle: "Raw_data/video/", status: "done" },
    { id: "ts", title: "frame_timestamps", subtitle: "meta/frame_timestamps/", status: "done" },
    { id: "sample", title: "1Hz 샘플링", subtitle: "sample_hz=1.0", status: "done" },
    { id: "yolo", title: "YOLO pose", subtitle: "GPU · yolo11m-pose", status: "done" },
    { id: "cls", title: "6-class Keras", subtitle: "CPU · my_model_six", status: "done" },
    { id: "out", title: "timeseries/", subtitle: "CSV + JSON / 영상", status: "done" },
    { id: "idx", title: "timeseries_index", subtitle: "~31k rows 합계", status: "done" },
  ],
  edges: [
    { from: "mp4", to: "ts" },
    { from: "ts", to: "sample" },
    { from: "sample", to: "yolo" },
    { from: "yolo", to: "cls" },
    { from: "cls", to: "out" },
    { from: "out", to: "idx" },
  ],
};

const SETUP_FLOW: FlowDef = {
  direction: "vertical",
  nodeWidth: 200,
  nodeHeight: 56,
  nodes: [
    { id: "camera", title: "카메라 설치", subtitle: "RTSP 640×360", status: "done" },
    { id: "refs", title: "침대 참고 이미지", subtitle: "labeling/ vlcsnap ×16", status: "done" },
    { id: "preset", title: "room / preset JSON", subtitle: "설치 프로파일 등록", status: "planned" },
    { id: "calib", title: "4점 H 보정", subtitle: "침대 상면 모서리", status: "planned" },
    { id: "server", title: "server.py 실행", subtitle: "pose-cuda env", status: "partial" },
    { id: "api", title: "API 확인", subtitle: "/status · /video", status: "planned" },
  ],
  edges: [
    { from: "camera", to: "refs" },
    { from: "refs", to: "preset" },
    { from: "preset", to: "calib" },
    { from: "calib", to: "server" },
    { from: "server", to: "api" },
  ],
};

const RUNTIME_FLOW: FlowDef = {
  direction: "vertical",
  nodeWidth: 196,
  nodeHeight: 52,
  nodes: [
    { id: "in", title: "RTSP / MP4", status: "done" },
    { id: "pre", title: "Preprocess", subtitle: "resize W=800", status: "done" },
    { id: "a", title: "A · Bed Seg", subtitle: "yolo11n-seg · class 59", status: "done" },
    { id: "c", title: "C · Pose", subtitle: "yolo11m-pose", status: "done" },
    { id: "b", title: "B · Homography", subtitle: "H · bed_norm", status: "planned" },
    { id: "d", title: "D · 6-class", subtitle: "my_model_six_check.keras", status: "done" },
    { id: "e", title: "E · Features", subtitle: "out_bed_ratio · motion", status: "planned" },
    { id: "f", title: "F · Temporal", subtitle: "3~5s buffer", status: "planned" },
    { id: "g", title: "G · Rule Score", subtitle: "risk_score 0~10", status: "planned" },
    { id: "out", title: "출력", subtitle: "/status · CSV · JSON", status: "partial" },
  ],
  edges: [
    { from: "in", to: "pre" },
    { from: "pre", to: "a" },
    { from: "pre", to: "c" },
    { from: "a", to: "b" },
    { from: "a", to: "e" },
    { from: "c", to: "d" },
    { from: "c", to: "e" },
    { from: "d", to: "e" },
    { from: "b", to: "e" },
    { from: "e", to: "f" },
    { from: "f", to: "g" },
    { from: "g", to: "out" },
  ],
};

const IMPL_FLOW: FlowDef = {
  direction: "vertical",
  nodeWidth: 210,
  nodeHeight: 52,
  nodes: [
    { id: "p0", title: "Phase 0", subtitle: "bed_monitor 모듈 분리", status: "planned" },
    { id: "p1", title: "Phase 1", subtitle: "preset / room 로더", status: "planned" },
    { id: "p2", title: "Phase 2", subtitle: "bed_bbox · overflow", status: "planned" },
    { id: "p3", title: "Phase 3", subtitle: "H · out_bed_ratio · 캘리브", status: "planned" },
    { id: "p4", title: "Phase 4", subtitle: "temporal · risk_score", status: "planned" },
    { id: "p5", title: "Phase 5", subtitle: "MP4 → CSV · 라벨링", status: "planned" },
  ],
  edges: [
    { from: "p0", to: "p1" },
    { from: "p1", to: "p2" },
    { from: "p2", to: "p3" },
    { from: "p3", to: "p4" },
    { from: "p4", to: "p5" },
  ],
};

const FLOWS: Record<ViewId, FlowDef> = {
  setup: SETUP_FLOW,
  runtime: RUNTIME_FLOW,
  rawbatch: RAW_BATCH_FLOW,
  impl: IMPL_FLOW,
};

const VIEW_CAPTION: Record<ViewId, string> = {
  setup: "고정 카메라 · 침대 거의 안 밀림 전제. labeling/ 이미지는 Phase 3 캘리브 직전 자료.",
  runtime: "S3 파이프라인 (FALL_RISK_SYSTEM_DESIGN). server.py는 A·C·D + 중심점 in_bed만 구현.",
  rawbatch:
    "extract_raw_timeseries.py · Dataset/Raw_data. 8개 MP4 → 1Hz CSV/JSON (2026-06-01 완료).",
  impl: "IMPLEMENTATION_PLAN.md Phase 0→5. 권장 PR: P0 → P1+P2 → P3 → P4 → P5.",
};

function statusColors(theme: CanvasHostTheme, status: NodeStatus) {
  switch (status) {
    case "done":
      return {
        fill: theme.fill.secondary,
        stroke: theme.stroke.secondary,
        title: theme.text.primary,
        sub: theme.text.secondary,
        badge: "success" as const,
      };
    case "current":
      return {
        fill: theme.fill.tertiary,
        stroke: theme.accent.primary,
        title: theme.text.primary,
        sub: theme.text.secondary,
        badge: "info" as const,
      };
    case "partial":
      return {
        fill: theme.fill.tertiary,
        stroke: theme.stroke.primary,
        title: theme.text.primary,
        sub: theme.text.secondary,
        badge: "warning" as const,
      };
    default:
      return {
        fill: theme.bg.editor,
        stroke: theme.stroke.tertiary,
        title: theme.text.secondary,
        sub: theme.text.tertiary,
        badge: "neutral" as const,
      };
  }
}

function statusLabel(status: NodeStatus): string {
  switch (status) {
    case "done":
      return "완료";
    case "current":
      return "현재";
    case "partial":
      return "일부";
    default:
      return "예정";
  }
}

function FlowDiagram({ flow }: { flow: FlowDef }) {
  const theme = useHostTheme();
  const byId = new Map(flow.nodes.map((n) => [n.id, n]));

  const layout = computeDAGLayout({
    nodes: flow.nodes.map((n) => ({ id: n.id })),
    edges: flow.edges,
    direction: flow.direction ?? "vertical",
    nodeWidth: flow.nodeWidth ?? 180,
    nodeHeight: flow.nodeHeight ?? 48,
    rankGap: 56,
    nodeGap: 32,
    padding: 28,
  });

  const nw = flow.nodeWidth ?? 180;
  const nh = flow.nodeHeight ?? 48;

  return (
    <FlowSvg width={layout.width} height={layout.height} theme={theme}>
      {layout.edges.map((edge) => {
        const dashed = edge.isBackEdge;
        return (
          <line
            key={`${edge.from}-${edge.to}`}
            x1={edge.sourceX}
            y1={edge.sourceY}
            x2={edge.targetX}
            y2={edge.targetY}
            stroke={theme.stroke.secondary}
            strokeWidth={1.5}
            strokeDasharray={dashed ? "5 4" : undefined}
            markerEnd="url(#arrow)"
          />
        );
      })}
      {layout.nodes.map((pos) => {
        const def = byId.get(pos.id);
        if (!def) return null;
        const colors = statusColors(theme, def.status);
        return (
          <g key={pos.id} transform={`translate(${pos.x}, ${pos.y})`}>
            <rect
              width={nw}
              height={nh}
              rx={6}
              fill={colors.fill}
              stroke={colors.stroke}
              strokeWidth={def.status === "current" ? 2 : 1}
            />
            <text
              x={12}
              y={22}
              fill={colors.title}
              fontSize={13}
              fontWeight={600}
              fontFamily="system-ui, sans-serif"
            >
              {def.title}
            </text>
            {def.subtitle ? (
              <text
                x={12}
                y={40}
                fill={colors.sub}
                fontSize={11}
                fontFamily="system-ui, sans-serif"
              >
                {def.subtitle}
              </text>
            ) : null}
            <rect x={nw - 52} y={8} width={44} height={18} rx={4} fill={theme.fill.quaternary} />
            <text
              x={nw - 30}
              y={20}
              textAnchor="middle"
              fill={colors.sub}
              fontSize={9}
              fontFamily="system-ui, sans-serif"
            >
              {statusLabel(def.status)}
            </text>
          </g>
        );
      })}
    </FlowSvg>
  );
}

function FlowSvg({
  width,
  height,
  theme,
  children,
}: {
  width: number;
  height: number;
  theme: CanvasHostTheme;
  children?: unknown;
}) {
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block", maxWidth: "100%", height: "auto" }}
    >
      <defs>
        <marker
          id="arrow"
          markerWidth="8"
          markerHeight="8"
          refX="6"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 Z" fill={theme.stroke.secondary} />
        </marker>
      </defs>
      {children}
    </svg>
  );
}

export default function PosePipelineFlow() {
  const theme = useHostTheme();
  const [view, setView] = useCanvasState<ViewId>("flow-view", "setup");
  const flow = FLOWS[view];

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>pose-sixclass · 파이프라인 순서</H1>
        <Text tone="secondary">
          현장 셋업 → 프레임 처리 (A~G) → 코드 이식 Phase. 채팅 옆에서 탭으로 전환하세요.
        </Text>
      </Stack>

      <Row gap={8} wrap>
        {VIEW_OPTIONS.map((opt) => (
          <Button
            key={opt.id}
            variant={view === opt.id ? "primary" : "ghost"}
            onClick={() => setView(opt.id)}
          >
            {opt.label}
          </Button>
        ))}
      </Row>

      <Callout tone="info">{VIEW_CAPTION[view]}</Callout>

      <Row gap={12} wrap>
        <Pill tone="success" size="small">
          완료
        </Pill>
        <Pill tone="warning" size="small">
          일부
        </Pill>
        <Pill tone="neutral" size="small">
          예정
        </Pill>
      </Row>

      <div
        style={mergeStyle({
          overflowX: "auto",
          padding: 8,
          borderRadius: 8,
          border: `1px solid ${theme.stroke.tertiary}`,
        })}
      >
        <FlowDiagram flow={flow} />
      </div>

      <Divider />

      <H2>현재 상태 요약</H2>
      <Table
        headers={["구분", "항목", "상태", "비고"]}
        rows={
          view === "setup"
            ? [
                ["자료", "labeling/ vlcsnap", "완료", "16장 · 640×360 · PNG 정상"],
                ["네트워크", "RTSP", "완료", "192.168.0.157:8554"],
                ["설정", "preset / H", "예정", "Phase 1·3"],
                ["서비스", "server.py", "일부", "코드만 — 수동 기동 필요"],
              ]
            : view === "runtime"
              ? [
                  ["L1", "A Bed Seg", "완료", "server.py"],
                  ["L1", "C Pose + D 6-class", "완료", "server.py"],
                  ["L1", "in_bed", "일부", "bbox 중심 1점 (out_ratio 없음)"],
                  ["L2", "B H · E features", "예정", "Phase 3"],
                  ["L3", "F · G risk_score", "예정", "Phase 4"],
                ]
              : view === "rawbatch"
                ? [
                    ["입력", "Raw MP4", "완료", "8 files"],
                    ["스크립트", "extract_raw_timeseries", "완료", "YOLO GPU + Keras CPU"],
                    ["출력", "timeseries/*.csv", "완료", "8 files · ~31k rows"],
                    ["메타", "rotation_bucket", "일부", "기본 unknown"],
                    ["다음", "F·G risk_score", "예정", "CSV 입력 rule"],
                  ]
                : [
                  ["코드", "bed_monitor/", "예정", "Phase 0"],
                  ["기하", "overflow · zone", "예정", "Phase 2"],
                  ["캘리브", "4점 + labeling 참고", "예정", "Phase 3"],
                  ["데이터", "MP4 → CSV", "예정", "Phase 5"],
                ]
        }
        rowTone={[undefined, undefined, "success", undefined]}
      />

      <Divider />

      <H3>server.py 지금 하는 일 (런타임 축약)</H3>
      <Text size="small" tone="secondary">
        RTSP → resize 800 → YOLO-seg(침대) + YOLO-pose → Keras 6-class → 중심점 in_bed → /status
      </Text>
    </Stack>
  );
}
