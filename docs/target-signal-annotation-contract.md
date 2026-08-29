# 当前目标信号机标注合同

本合同定义当前目标信号机的唯一标注来源，以及当前目标检测数据和 ROI 有效灯色数据的确定性导出规则。

## 输入文件

合同使用两个 JSONL 输入。每行都是一个 JSON 对象，两个文件通过 `frame_id` 一一对应。

### 采集元数据

采集工具负责生成元数据，标注人员无需重复填写。

```json
{"frame_id":"a-st01-000123","image_path":"frames/a-st01-000123.jpg","camera":"A","video_id":"a-trip-01","frame_index":123,"timestamp_ms":12300,"section_id":"station-01","split":"train","image_width":1920,"image_height":1080}
```

| 字段 | 含义 |
| --- | --- |
| `frame_id` | 帧的稳定标识，只能包含字母、数字、点、下划线和连字符 |
| `image_path` | 图像路径 |
| `camera` | 相机来源，取 `A` 或 `B` |
| `video_id` | 原始视频或行程标识 |
| `frame_index` | 从零开始的非负帧序号 |
| `timestamp_ms` | 非负毫秒时间戳 |
| `section_id` | 站点或连续区段标识 |
| `split` | 数据集合，取 `train` 或 `val` |
| `image_width` | 原图宽度，单位为像素 |
| `image_height` | 原图高度，单位为像素 |

### 人工标注

有当前目标信号机时，`targets` 包含一个目标：

```json
{"frame_id":"a-st01-000123","targets":[{"bbox_xyxy":[120,80,420,620],"configuration":"ULH","observation":"L"}]}
```

没有当前目标信号机时，`targets` 为空：

```json
{"frame_id":"a-st01-000124","targets":[]}
```

每帧最多一个当前目标。目标字段为：

| 字段 | 含义 |
| --- | --- |
| `bbox_xyxy` | 完整目标信号机构的像素框，顺序为左、上、右、下 |
| `configuration` | 显示配置，取 `ULH`、`LH` 或 `空H` |
| `observation` | 观测灯色，取 `U`、`L`、`H`、`UNLIT` 或 `UNREADABLE` |

`UNLIT` 表示信号机构可见，但所有已安装发光盘均未点亮。`UNREADABLE` 表示图像证据不足，不能把它猜测成未亮或任一灯色。

## 合法组合

| 显示配置 | 合法观测 |
| --- | --- |
| `ULH` | `U`、`L`、`H`、`UNLIT`、`UNREADABLE` |
| `LH` | `L`、`H`、`UNLIT`、`UNREADABLE` |
| `空H` | `H`、`UNLIT`、`UNREADABLE` |

U 不能标给 LH 或空H，L 不能标给空H。空H中的“空”是没有安装发光盘的空物理灯位，不是颜色类别。

## 校验

校验要求：

- 每条采集元数据恰好对应一条人工标注；
- 人工标注引用的 `frame_id` 必须存在于采集元数据；
- 一个帧内至多一个当前目标；
- 框必须包含四个有限数值，并完全位于原图范围内；
- 显示配置与观测必须属于合法组合；
- 输入中不能出现重复 `frame_id`；
- `split` 只能是 `train` 或 `val`。

运行校验：

```bash
python3 target_signal_annotations.py validate \
  --metadata acquisition-metadata.jsonl \
  --annotations manual-annotations.jsonl
```

合法输入输出记录数、当前目标数、无目标帧数、灯色样本数和看不清样本数。非法输入以退出码 `2` 结束，并指出对应帧和字段。

## 派生数据

运行导出：

```bash
python3 target_signal_annotations.py export \
  --metadata acquisition-metadata.jsonl \
  --annotations manual-annotations.jsonl \
  --output exported-dataset
```

输出目录必须为空，避免旧标签残留。

### 当前目标检测数据

检测类别固定为：

| 类别编号 | 类别名称 |
| ---: | --- |
| 0 | `TARGET_ULH` |
| 1 | `TARGET_LH` |
| 2 | `TARGET_EMPTY_H` |

每帧生成一个 YOLO 格式标签文件。无目标帧生成空标签文件。检测清单保留采集元数据、目标存在性和标签路径。

### ROI 有效灯色数据

灯色目标向量顺序固定为 `[U, L, H]`：

| 观测 | 目标向量 |
| --- | --- |
| `U` | `[1, 0, 0]` |
| `L` | `[0, 1, 0]` |
| `H` | `[0, 0, 1]` |
| `UNLIT` | `[0, 0, 0]` |

无目标帧和 `UNREADABLE` 不进入灯色清单。灯色清单记录源图、目标框、显示配置、观测、目标向量和采集元数据；后续灯色数据加载器根据目标框裁剪 ROI，不维护第二份人工框。

### 一致性产物

导出同时生成：

- 合并后的主标注清单；
- 检测类别表和逐帧检测清单；
- ROI 灯色清单；
- 导出数量汇总。

两个模型的训练数据都从同一组已校验主记录产生，派生文件不是独立标注来源。
