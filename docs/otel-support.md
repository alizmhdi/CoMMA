# OpenTelemetry Support in CoMMA

CoMMA (Collective coMMunication Analyzer) profiler supports exporting telemetry data via OpenTelemetry (OTel). It supports both metrics (latency histograms and status gauges) and tracing (NCCL operation spans).

## Configuration

OTel support is configured via environment variables.

| Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `NCCL_PROFILER_OTEL_ENABLE` | Boolean | `false` | Enables OpenTelemetry support. |
| `NCCL_PROFILER_OTEL_TRACE_NCCLOP` | Boolean | `false` | (Experimental) Enables tracing for NCCL operations. Requires `NCCL_PROFILER_OTEL_ENABLE` to be `true`. |
| `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` | Integer | `0` | Maximum number of unique metric streams (cardinality limit) for high-fidelity tracking. If set to `0`, all metrics use low-fidelity aggregation. |
| `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL` | Duration | `3600s` | Interval at which CoMMA updates the priority ("Top K") of metrics for cardinality management. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SIZE` | Integer | `160` | Maximum size parameter for OTel Base2 Exponential Histogram. |
| `NCCL_PROFILER_OTEL_LATENCY_HISTOGRAM_MAX_SCALE` | Integer | `20` | Maximum scale parameter for OTel Base2 Exponential Histogram. |

### Duration Format
Duration fields (like `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL`) support values with units:
- `d`: Days
- `h`: Hours
- `m`: Minutes
- `s`: Seconds
- `ms`: Milliseconds
- `us`: Microseconds
- `ns`: Nanoseconds

Example: `1h30m` or `10s`.

## Metrics

CoMMA registers a meter provider under the service name `CoMMA`. It exports the following metrics:

### `nccl.net.send.latency` (Histogram, Unit: `ns`)

This metric records the latency of network send operations (also known as `isend()` or "proxy step").

To prevent high cardinality issues, CoMMA uses a "Top K" cardinality management strategy if `NCCL_PROFILER_OTEL_METRICS_MAX_CARDINALITY` is configured to a value greater than 0:

- **High-Fidelity Streams (Top K)**: The most frequent network send operations (up to the configured limit) are recorded with full attributes:
  - `nccl.communicator.hash`: Hexadecimal string identifying the NCCL communicator.
  - `nccl.source.rank`: Source rank of the transfer.
  - `nccl.destination.rank`: Destination rank of the transfer.
  - `nccl.hostname`: Hostname of the node.
- **Aggregated Streams**: Streams exceeding the cardinality limit are aggregated together to save memory and export bandwidth. They are recorded with:
  - `nccl.metric.aggregated`: Set to `true`.
  - `nccl.hostname`: Hostname of the node.

The "Top K" list is dynamically updated at the interval defined by `NCCL_PROFILER_OTEL_METRICS_CARDINALITY_GROUPING_INTERVAL`.

### `nccl.collective.seq_num` (Gauge)

This metric records the sequence number of the last collective operation executed.

Attributes:
- `nccl.comm.hash`: Hexadecimal string identifying the NCCL communicator.
- `nccl.collective.name`: Name of the collective operation (e.g., `ncclAllReduce`, `ncclBroadcast`).
- `nccl.rank`: Rank of the process.
- `nccl.hostname`: Hostname of the node.

## Tracing (Experimental)

> [!WARNING]
> Tracing support in CoMMA is currently experimental and may be subject to future changes.

When tracing is enabled (via `NCCL_PROFILER_OTEL_TRACE_NCCLOP`), CoMMA exports spans for NCCL operations.

### Span Correlation Across Ranks

For collective operations, CoMMA attempts to correlate spans across different ranks participating in the same collective:

- A deterministic `TraceId` is generated using the communicator hash.
- A deterministic `SpanId` is generated using the collective operation type and its sequence number.
- Rank 0 of the communicator creates the parent span (Server span) with the duration of the operation on rank 0.
- Other ranks create child spans linked to this parent span using the remote span context.

This allows visualization of the collective operation as a single distributed trace.

### Span Attributes

Spans contain the following attributes:

- **Common Attributes**:
  - `nccl.comm.hash`: Communicator hash.
  - `nccl.rank`: Rank of the process.
  - `nccl.size.bytes`: Size of the data transferred.

- **Collective Operation Attributes** (e.g., `ncclAllReduce`, `ncclBroadcast`):
  - `nccl.collective.algo`: NCCL algorithm used (e.g., Tree, Ring).
  - `nccl.collective.proto`: NCCL protocol used (e.g., LL, LL128, Simple).
  - `nccl.collective.n_max_channel`: Number of channels used.

- **P2P Operation Attributes** (`ncclSend`, `ncclRecv`):
  - `nccl.p2p.peer.rank`: Peer rank involved in the transfer.
