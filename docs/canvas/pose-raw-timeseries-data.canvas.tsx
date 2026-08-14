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
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

const videos = [
  { name: "Rail-Raw0 (1).mp4", rows: 5634, hours: 1.6 },
  { name: "Raw0 (1).mp4", rows: 5123, hours: 1.4 },
  { name: "Raw0 (2).mp4", rows: 647, hours: 0.2 },
  { name: "Raw0 (3).mp4", rows: 14256, hours: 4.0 },
  { name: "Raw0 (4).mp4", rows: 1169, hours: 0.3 },
  { name: "Raw0 (5).mp4", rows: 1673, hours: 0.5 },
  { name: "Raw0 (6).mp4", rows: 2452, hours: 0.7 },
  { name: "Raw0 (7).mp4", rows: 628, hours: 0.2 },
];

const totalRows = videos.reduce((s, v) => s + v.rows, 0);

const csvColumns = [
  "video_file",
  "frame_idx",
  "timestamp_sec",
  "fps",
  "rotation_bucket",
  "pose_class",
  "pose_class_id",
  "pose_conf",
  "person_detected",
  "kpt_0 … kpt_33",
];

export default function PoseRawTimeseriesData() {
  const theme = useHostTheme();

  return (
    <Stack gap={20}>
      <Stack gap={6}>
        <H1>Raw_data 시계열 산출물</H1>
        <Text tone="secondary">
          Source: extract_raw_timeseries.py · sample_hz=1.0 · 2026-06-01 배치 완료
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat label="영상 수" value="8" tone="info" />
        <Stat label="총 샘플 행" value={totalRows.toLocaleString()} tone="success" />
        <Stat label="샘플 밀도" value="1 Hz" />
        <Stat label="출력 경로" value="Raw_data/timeseries/" />
      </Grid>

      <Card>
        <CardHeader title="영상별 행 수 (1Hz)" trailing={`합계 ${totalRows.toLocaleString()} rows`} />
        <CardBody>
          <BarChart
            title="Timeseries rows per video"
            xLabel="Video"
            yLabel="Row count"
            data={videos.map((v) => ({
              label: v.name.replace(".mp4", "").replace("Raw0 ", "R"),
              value: v.rows,
            }))}
            height={220}
          />
          <Text size="small" tone="tertiary">
            Source: timeseries_index.json · longest: Raw0 (3).mp4
          </Text>
        </CardBody>
      </Card>

      <Card>
        <CardHeader title="CSV 스키마" />
        <CardBody>
          <Code block>{csvColumns.join(", ")}</Code>
          <Text size="small" tone="secondary">
            pose_class_id = -1 when person_detected=false. rotation_bucket default unknown.
          </Text>
        </CardBody>
      </Card>

      <H2>영상 목록</H2>
      <Table
        headers={["파일", "rows (1Hz)", "대략 길이"]}
        rows={videos.map((v) => [v.name, String(v.rows), `~${v.hours} h`])}
      />

      <Divider />

      <H3>폴더 구조 (Dataset/Raw_data)</H3>
      <Code block>{`Raw_data/
  video/              # 8× MP4
  meta/frame_timestamps/
  timeseries/         # 8× .csv + .json
  timeseries_index.json
  README.md`}</Code>

      <Row gap={8} wrap>
        <Text size="small" tone="tertiary">
          Cursor canvas: pose-pipeline-flow (탭 Raw 배치 시계열)
        </Text>
      </Row>
    </Stack>
  );
}
