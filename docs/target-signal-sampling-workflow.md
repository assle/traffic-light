# 站点分组与关键帧标注工作流

本工作流在人工标注前固定站点的训练/验证归属，再从连续帧和可选目标提示中生成关键帧、插值区间和人工抽查清单。

## 工作顺序

1. 数据负责人编写站点计划，将每个连续视频区段分配到 `train` 或 `val`。
2. 采集流程生成逐帧清单，并可附带已有模型或粗审产生的目标提示。
3. 校验器确认站点、区段和帧没有跨集合重叠。
4. 抽帧器生成关键帧候选、Issue #2兼容的采集元数据和人工标注模板。
5. 标注人员确认当前目标、显示配置和观测灯色，并抽查插值区间。

站点归属必须先于抽帧，不能先生成图片再随机划分训练和验证集合。

## 站点计划

站点计划是一个 JSON 对象：

```json
{
  "station_groups": [
    {
      "station_group_id": "station-01",
      "station_id": "station-01",
      "split": "train",
      "sections": [
        {
          "section_id": "station-01-a",
          "camera": "A",
          "video_id": "a-trip-01",
          "start_frame": 0,
          "end_frame": 1500
        },
        {
          "section_id": "station-01-b",
          "camera": "B",
          "video_id": "b-trip-01",
          "start_frame": 0,
          "end_frame": 1200
        }
      ]
    }
  ]
}
```

`start_frame` 和 `end_frame` 都包含在区段范围内。

| 字段 | 含义 |
| --- | --- |
| `station_group_id` | 同一站点或需要共同分配的数据组 |
| `station_id` | 站点身份；同一站点的所有组必须属于同一集合 |
| `split` | `train` 或 `val` |
| `section_id` | 一个连续、不重叠的视频区段 |
| `camera` | `A` 或 `B` |
| `video_id` | 帧清单中的视频标识 |
| `start_frame` | 区段首帧 |
| `end_frame` | 区段末帧 |

同一站点的 A系和 B系区段使用相同 `station_id`，从而保证它们进入同一集合。

## 帧清单

帧清单使用 JSONL。每行描述一帧：

```json
{"frame_id":"a-trip-01-000120","image_path":"frames/a-trip-01-000120.jpg","camera":"A","video_id":"a-trip-01","frame_index":120,"timestamp_ms":12000,"image_width":1920,"image_height":1080,"target_hint":null}
```

`target_hint` 可以为空。没有目标提示的帧仍参与无目标区间抽样。

目标提示存在时使用以下结构：

```json
{
  "track_id": "candidate-01",
  "bbox_xyxy": [120, 80, 420, 620],
  "state_hint": "L",
  "occlusion_score": 0.1,
  "exposure_score": 0.2
}
```

| 字段 | 含义 |
| --- | --- |
| `track_id` | 连续候选轨迹标识，用于排序、插值和交接候选 |
| `bbox_xyxy` | 候选信号机构框 |
| `state_hint` | 可选的 U/L/H/UNLIT/UNREADABLE提示 |
| `occlusion_score` | 0到1的遮挡程度提示 |
| `exposure_score` | 0到1的异常曝光程度提示 |

目标提示只用于安排标注帧，不是人工真值。人工标注人员仍需确认当前目标、显示配置和观测灯色。

## 校验

运行：

```bash
python3 target_signal_sampling.py validate \
  --plan station-plan.json \
  --inventory frame-inventory.jsonl
```

校验会拒绝：

- 同一 `station_id` 同时属于 `train` 和 `val`；
- 重复 `station_group_id` 或 `section_id`；
- 同一相机和视频中的连续区段范围重叠；
- 同一帧范围被分配到多个集合；
- 重复的 `frame_id` 或相机/视频/帧序号组合；
- 没有归属任何区段的帧；
- 越界候选框、非法提示状态和超出0到1的质量分数。

合法输入输出帧数、训练/验证帧数、A/B帧数、站点组数、区段数和预计关键帧数。

## 关键帧原因

抽帧器为候选帧记录一个或多个原因：

| 原因 | 含义 |
| --- | --- |
| `TRACK_START` | 目标轨迹初见 |
| `DISTANCE_CHANGE` | 候选框面积相对上一个距离样本发生显著变化 |
| `NEAR` | 轨迹中候选框面积最大的近处帧 |
| `TRACK_END` | 目标轨迹离开 |
| `STATE_CHANGE` | 灯色提示发生变化 |
| `HANDOVER` | 连续目标提示的轨迹身份发生交接 |
| `OCCLUSION` | 遮挡高风险连续区间的峰值帧 |
| `EXPOSURE` | 异常曝光连续区间的峰值帧 |
| `STABLE_INTERVAL` | 稳定轨迹中为控制插值跨度而选取的周期帧 |
| `NO_TARGET_INTERVAL` | 没有目标提示区间的边界或周期帧 |

可配置参数：

- `--max-gap-frames`：稳定区间两个标注关键帧之间允许的最大帧差；
- `--distance-change-ratio`：触发距离变化候选的框面积相对变化；
- `--occlusion-threshold`：遮挡候选阈值；
- `--exposure-threshold`：异常曝光候选阈值。

## 构建标注包

运行：

```bash
python3 target_signal_sampling.py build \
  --plan station-plan.json \
  --inventory frame-inventory.jsonl \
  --output annotation-package
```

输出目录必须为空。标注包包含：

| 文件 | 内容 |
| --- | --- |
| `data-manifest.jsonl` | 全量帧的来源、站点组、集合、相机、视频、帧序号和是否入选 |
| `keyframes.jsonl` | 入选关键帧、目标提示和选择原因 |
| `acquisition-metadata.jsonl` | 可直接交给主标注合同的采集元数据 |
| `manual-annotations-template.jsonl` | 待人工填写的最小标注模板 |
| `interpolation-segments.jsonl` | 同一轨迹相邻关键帧之间的有序中间帧 |
| `review-checklist.jsonl` | 插值中点、状态变化、交接、遮挡和曝光风险抽查项 |
| `sampling-summary.json` | 集合、相机、站点、候选原因和抽查数量汇总 |

人工标注模板中的 `targets` 初始值为 `null`，表示尚未标注；它不能当作无目标事实提交给主标注校验器。标注人员确认后必须将其改为一个目标列表或空列表。

## 插值与抽查

插值区间只连接同一 `section_id`、同一 `track_id` 的相邻关键帧，不跨站点、区段或轨迹身份。每个包含中间帧的插值区间至少抽查一个中点帧；状态变化、目标交接、遮挡和异常曝光帧始终进入抽查清单。

框插值结果只是标注加速手段。当前目标身份、显示配置和观测灯色仍由人工确认。
