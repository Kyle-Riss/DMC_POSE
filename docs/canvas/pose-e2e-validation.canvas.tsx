import {
  BarChart,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  PieChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const labels = ["정면_누움", "엎드림_등", "옆누움_가까움", "옆누움_멀음", "앉음_중앙", "앉음_가장자리"];

const summary = {
  root: "/home/dmc/pose/extracted_frames",
  dataset: "/home/dmc/pose-sixclass/pose_dataset_six.csv",
  model: "/home/dmc/pose-sixclass/my_model_six.keras",
  weights: "yolo11m-pose.pt",
  sampled: 300,
  used: 295,
  skippedNoPose: 5,
  correct: 291,
  accuracy: 0.986441,
};

const classMetrics = [
  { label: "정면_누움", precision: 0.98, recall: 0.98, f1: 0.98, support: 50 },
  { label: "엎드림_등", precision: 0.9846, recall: 0.9846, f1: 0.9846, support: 65 },
  { label: "옆누움_가까움", precision: 0.9722, recall: 1, f1: 0.9859, support: 35 },
  { label: "옆누움_멀음", precision: 1, recall: 0.9459, f1: 0.9722, support: 37 },
  { label: "앉음_중앙", precision: 1, recall: 1, f1: 1, support: 51 },
  { label: "앉음_가장자리", precision: 0.9828, recall: 1, f1: 0.9913, support: 57 },
];

const confusionMatrix = [
  [49, 0, 1, 0, 0, 0],
  [1, 64, 0, 0, 0, 0],
  [0, 0, 35, 0, 0, 0],
  [0, 1, 0, 35, 0, 1],
  [0, 0, 0, 0, 51, 0],
  [0, 0, 0, 0, 0, 57],
];

const gtCounts = [50, 68, 35, 37, 53, 57];
const predCounts = [50, 65, 36, 35, 51, 58];

const misclassified = [
  {
    src: "/home/dmc/pose/extracted_frames/P07,08/7/15/frame_0124_124s.jpg",
    gt: "옆누움_멀음",
    pred: "앉음_가장자리",
    conf: 0.9746,
  },
  {
    src: "/home/dmc/pose/extracted_frames/P01,02/P2/8/frame_0006_6s.jpg",
    gt: "정면_누움",
    pred: "옆누움_가까움",
    conf: 0.7015,
  },
  {
    src: "/home/dmc/pose/extracted_frames/P03,04/P3/9/frame_0000_0s.jpg",
    gt: "엎드림_등",
    pred: "정면_누움",
    conf: 0.9151,
  },
  {
    src: "/home/dmc/pose/extracted_frames/P07,08/8/4/frame_0001_1s.jpg",
    gt: "옆누움_멀음",
    pred: "엎드림_등",
    conf: 0.5537,
  },
];

function pct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function shortPath(path: string) {
  return path.replace("/home/dmc/pose/extracted_frames/", "");
}

function MetricCell({ value, best }: { value: number; best?: boolean }) {
  return (
    <Row gap={8} align="center">
      <Text as="span" weight={best ? "semibold" : "normal"}>
        {pct(value)}
      </Text>
      {best ? (
        <Pill tone="success" size="sm" active>
          max
        </Pill>
      ) : null}
    </Row>
  );
}

function ConfusionMatrix() {
  const theme = useHostTheme();
  const maxValue = Math.max(...confusionMatrix.flat());

  return (
    <Stack gap={8}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "132px repeat(6, minmax(64px, 1fr))",
          gap: 4,
          alignItems: "stretch",
        }}
      >
        <div />
        {labels.map((label) => (
          <Text key={`pred-${label}`} size="small" tone="secondary" truncate style={{ textAlign: "center" }}>
            {label}
          </Text>
        ))}
        {confusionMatrix.map((row, rowIndex) => (
          <>
            <Text key={`row-label-${labels[rowIndex]}`} size="small" tone="secondary" truncate style={{ alignSelf: "center" }}>
              {labels[rowIndex]}
            </Text>
            {row.map((value, colIndex) => {
              const isCorrect = rowIndex === colIndex;
              const isMistake = value > 0 && !isCorrect;
              const strength = value === 0 ? 0 : 0.22 + (value / maxValue) * 0.58;
              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  title={`${labels[rowIndex]} -> ${labels[colIndex]}: ${value}`}
                  style={{
                    minHeight: 38,
                    borderRadius: 6,
                    border: `1px solid ${theme.stroke.tertiary}`,
                    background: value === 0 ? theme.fill.quaternary : isMistake ? theme.fill.secondary : theme.accent.control,
                    color: isCorrect && value > 0 ? theme.text.onAccent : theme.text.primary,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: value > 0 ? 600 : 400,
                    opacity: value === 0 ? 0.55 : strength,
                  }}
                >
                  {value}
                </div>
              );
            })}
          </>
        ))}
      </div>
      <Text size="small" tone="secondary">
        행은 실제 라벨, 열은 예측 라벨입니다. 대각선 밖의 값 4개가 오분류입니다.
      </Text>
    </Stack>
  );
}

export default function PoseE2EValidationReport() {
  const totalErrors = summary.used - summary.correct;
  const detectionCoverage = summary.used / summary.sampled;

  return (
    <Stack gap={22}>
      <Stack gap={6}>
        <H1>Pose Six-Class E2E Validation</H1>
        <Text tone="secondary">
          <Code>runs/e2e_validate/report.txt</Code> 기준 모델 검증 결과입니다.
        </Text>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value={pct(summary.accuracy)} label="Accuracy" tone="success" />
        <Stat value={`${summary.correct}/${summary.used}`} label="Correct / Evaluated" />
        <Stat value={`${totalErrors}`} label="Misclassified" tone="warning" />
        <Stat value={`${summary.skippedNoPose}`} label="Skipped No Pose" tone="info" />
      </Grid>

      <Grid columns="1.05fr 1.4fr" gap={18}>
        <Card size="lg">
          <CardHeader trailing={<Pill tone="success" active>{pct(summary.accuracy)}</Pill>}>Evaluation Split</CardHeader>
          <CardBody>
            <Stack gap={14}>
              <PieChart
                donut
                size={210}
                data={[
                  { label: "Correct", value: summary.correct, tone: "success" },
                  { label: "Wrong", value: totalErrors, tone: "warning" },
                  { label: "Skipped", value: summary.skippedNoPose, tone: "info" },
                ]}
              />
              <Grid columns={2} gap={10}>
                <Stat value={summary.sampled} label="Sampled" />
                <Stat value={summary.used} label="Used For Eval" />
                <Stat value={pct(detectionCoverage)} label="Pose Detection Coverage" />
                <Stat value={labels.length} label="Classes" />
              </Grid>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={10}>
          <H2>Class-Level F1</H2>
          <BarChart
            horizontal
            height={260}
            categories={classMetrics.map((item) => item.label)}
            valueSuffix="%"
            series={[{ name: "F1-score", data: classMetrics.map((item) => Number((item.f1 * 100).toFixed(2))), tone: "success" }]}
          />
          <Text size="small" tone="secondary">
            최저 F1은 <Code>옆누움_멀음</Code>의 97.22%이고, <Code>앉음_중앙</Code>은 precision/recall/F1 모두 100%입니다.
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <Grid columns="1fr 1fr" gap={18}>
        <Stack gap={10}>
          <H2>Ground Truth vs Prediction Counts</H2>
          <BarChart
            height={260}
            categories={labels}
            series={[
              { name: "GT", data: gtCounts, tone: "neutral" },
              { name: "Pred", data: predCounts, tone: "info" },
            ]}
          />
        </Stack>

        <Stack gap={10}>
          <H2>Precision / Recall / F1</H2>
          <Table
            headers={["Class", "Precision", "Recall", "F1", "Support"]}
            columnAlign={["left", "right", "right", "right", "right"]}
            striped
            rows={classMetrics.map((item) => [
              item.label,
              <MetricCell value={item.precision} best={item.precision === 1} />,
              <MetricCell value={item.recall} best={item.recall === 1} />,
              <MetricCell value={item.f1} best={item.f1 === 1} />,
              item.support,
            ])}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>Confusion Matrix</H2>
        <ConfusionMatrix />
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>Misclassified Samples</H2>
        <Text tone="secondary">
          총 4건의 오분류가 있으며, 이 중 2건은 실제 <Code>옆누움_멀음</Code> 샘플입니다.
        </Text>
        <Table
          headers={["#", "Frame", "GT", "Pred", "Confidence"]}
          columnAlign={["right", "left", "left", "left", "right"]}
          rowTone={["warning", "warning", "warning", "warning"]}
          rows={misclassified.map((item, index) => [
            index + 1,
            <Text size="small" truncate="start">{shortPath(item.src)}</Text>,
            item.gt,
            item.pred,
            pct(item.conf),
          ])}
        />
      </Stack>

      <Card>
        <CardHeader>Run Inputs</CardHeader>
        <CardBody>
          <Stack gap={8}>
            <Text><Text as="span" weight="semibold">Root:</Text> <Code>{summary.root}</Code></Text>
            <Text><Text as="span" weight="semibold">Dataset:</Text> <Code>{summary.dataset}</Code></Text>
            <Text><Text as="span" weight="semibold">Model:</Text> <Code>{summary.model}</Code></Text>
            <Text><Text as="span" weight="semibold">Pose weights:</Text> <Code>{summary.weights}</Code></Text>
          </Stack>
        </CardBody>
      </Card>

      <H3>Readout</H3>
      <Text>
        전체 성능은 매우 높고, 주요 약점은 <Code>옆누움_멀음</Code> recall 94.59%입니다. 실제 <Code>옆누움_멀음</Code> 37개 중 2개가 각각
        <Code>엎드림_등</Code>, <Code>앉음_가장자리</Code>로 분류되었습니다.
      </Text>
    </Stack>
  );
}
