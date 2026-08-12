# CoMMA Configuration Options

> [!NOTE]
> This document serves as a complete reference of all configuration options for CoMMA developers and advanced users. Many of these options are internal or experimental. For the list of officially supported and stable options for Google Cloud Users, please refer to the [Google Cloud configure CoMMA documentation](https://cloud.google.com/ai-hypercomputer/docs/nccl/configure-comma).

CoMMA is configured primarily through environment variables. This document lists all available configuration options, categorized by their functionality.

Unless otherwise specified, all environment variables are prefixed with `NCCL_PROFILER_`.

---

## 1. Profiling Granularity

These options control what level of detail CoMMA collects. Enabling more detailed profiling may introduce higher overhead.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_TRACK_NCCLOP` | Boolean | `true` | Enable tracking of NCCL operations (collectives and P2P calls). |
| `NCCL_PROFILER_TRACK_GROUP` | Boolean | `false` | Enable tracking of NCCL group operations (`ncclGroupStart`/`ncclGroupEnd`). |
| `NCCL_PROFILER_TRACK_PROXYOP` | Boolean | `false` | Enable tracking of NCCL proxy operations (helper threads handling communication). |
| `NCCL_PROFILER_TRACK_INTERPROCESS_PROXYOP` | Boolean | `true` | Enable tracking of proxy operations that involve inter-process communication. |
| `NCCL_PROFILER_TRACK_STEPS` | Boolean | `false` | Enable tracking of individual network transfer steps within proxy operations. |
| `NCCL_PROFILER_TRACK_RECV_STEPS` | Boolean | `false` | Enable tracking of receive steps in addition to send steps. |
| `NCCL_PROFILER_TRACK_STEP_FIFO_WAIT` | Boolean | `true` | Enable tracking of time spent by steps waiting in the FIFO queue. |
| `NCCL_PROFILER_AGGREGATE_STEPS` | Boolean | `true` | Enable aggregation of step telemetry to reduce overhead when detailed step tracking is disabled. |
| `NCCL_PROFILER_TRACK_KERNEL_CH` | Boolean | `false` | Enable tracking of GPU kernel channel information. |
| `NCCL_PROFILER_NCCLOP_COMPLETION_DELAY` | Duration | `2s` | Delay duration before considering an asynchronous NCCL operation as complete. |
| `NCCL_PROFILER_COMM_HASH_IPC_TIMEOUT` | Duration | `1s` | Timeout duration for IPC communication to resolve communicator hashes. |

---

## 2. Performance & Sampling

These options allow tuning CoMMA's performance and overhead, primarily through sampling and size thresholds.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_FIFO_BATCH_SIZE` | Integer | `1024` | Batch size for reading events from the shared memory FIFO queue. |
| `NCCL_PROFILER_NCCLOP_TIMEOUT` | Duration | `10s` | Timeout after which a NCCL operation is considered hung if no progress is detected. |
| `NCCL_PROFILER_MAX_TRACKED_NCCLOP` | Integer | `65536` | Maximum number of concurrent NCCL operations tracked by the daemon. |
| `NCCL_PROFILER_SMALL_MSG_THRESHOLD` | Integer | `65536` | Message size threshold (in bytes) below which operations may be treated differently (e.g., skipped or aggregated). |
| `NCCL_PROFILER_SKIP_NVLS` | Boolean | `true` | Skip profiling NVLink SHARP (NVLS) operations to reduce overhead. |
| `NCCL_PROFILER_SKIP_SMALL_COLLECTIVE` | Boolean | `true` | Skip profiling collective operations with message sizes below `SMALL_MSG_THRESHOLD`. |
| `NCCL_PROFILER_SKIP_SMALL_COLLECTIVE_STEPS` | Boolean | `true` | Skip profiling steps of collective operations with message sizes below `SMALL_MSG_THRESHOLD`. |
| `NCCL_PROFILER_P2P_SAMPLE_RATE` | Float | `1.0` | Sample rate for Point-to-Point (P2P) operations (range `0.0` to `1.0`). `1.0` means profile all. |
| `NCCL_PROFILER_P2P_RECV_SAMPLE_RATE` | Float | `0.1` | Base sample rate for P2P receive operations. Adjusted dynamically based on `P2P_SAMPLE_RATE`. |
| `NCCL_PROFILER_USE_CACHED_CLOCK` | Boolean | `false` | Use a cached clock for timestamps to reduce CPU overhead from clock read system calls. |

---

## 3. Local File Export

CoMMA can export raw telemetry data to local files in JSON format.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_LATENCY_FILE` | String | *None* | Path to the local file where raw event telemetry will be written (e.g., `/tmp/latency-%p.json`). Supports `%p` for PID. |
| `NCCL_PROFILER_LATENCY_SOCK` | String | *None* | Unix stream socket for live raw event telemetry. Sends the same newline-delimited JSON as `LATENCY_FILE`; supports `%p` for PID but the monitor usually uses one shared socket per host. |
| `NCCL_PROFILER_SUMMARY_FILE` | String | *None* | Path to the local file where periodic summaries will be written. |
| `NCCL_PROFILER_SUMMARY_INTERVAL` | Duration | `60s` | Interval at which periodic summaries are calculated and written to `SUMMARY_FILE`. |

For details on the exported JSON format, see [CoMMA JSON export format](json-format.md).

---

## 4. Heartbeat Telemetry (GCP Specific)

These options configure the periodic heartbeat telemetry exported to GCP services.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_HEARTBEAT` | Boolean | `true` | Enable periodic heartbeat telemetry. |
| `NCCL_PROFILER_HEARTBEAT_UPLOAD_INTERVAL` | Duration | `1s` | Interval at which heartbeat telemetry is uploaded. |
| `NCCL_PROFILER_HEARTBEAT_COLLECTIVE_PROGRESS` | Boolean | `false` | Include detailed collective progress information in the heartbeat. Requires `HEARTBEAT` to be `true`. |

---

## 5. OpenTelemetry (OTel)

Options to configure OTel exporter. For detailed explanation, see [OpenTelemetry Support in CoMMA](otel-support.md).

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_OTEL_ENABLE` | Boolean | `false` | Enables OpenTelemetry support. |
| `NCCL_PROFILER_OTEL_TRACE_NCCLOP` | Boolean | `false` | (Experimental) Enables tracing for NCCL operations. |
| `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` | Integer | `0` | Max cardinality for metrics. |
| `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL` | Duration | `3600s` | Interval to update priority of metrics. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SIZE` | Integer | `160` | OTel Histogram max size. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SCALE` | Integer | `20` | OTel Histogram max scale. |

---

## 6. General & Legacy Telemetry

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_USE_GPUVIZ` | Boolean | `true` | Enable integration with GPUViz (legacy telemetry system). |
| `NCCL_PROFILER_GPUVIZ_LOG_LEVEL` | Integer | `0` | Log level mapping for GPUViz logs (0: warn/error to info, 1: error to warn, 2: as-is). |
| `NCCL_PROFILER_GPUVIZ_LIB` | String | *Varies* | Path to the GPUViz library. |
| `NCCL_TELEMETRY_MODE` | Integer | `3` | Configures the telemetry mode. Default is `3` (upload enabled). |

*Note: `NCCL_TELEMETRY_MODE` does not use the `PROFILER_` infix.*

---

## Duration Format
Duration fields support values with units:
- `d`: Days
- `h`: Hours
- `m`: Minutes
- `s`: Seconds
- `ms`: Milliseconds
- `us`: Microseconds
- `ns`: Nanoseconds

Example: `2s`, `10m`, `1h30m`.

## Boolean Format
Boolean fields are case-insensitive and accept multiple representations:
- **True / Enabled**: `true`, `yes`, `y`, `1`
- **False / Disabled**: `false`, `no`, `n`, `0`
