# CoMMA (Collective coMMunication Analyzer)
This is the repository for CoMMA, a library that collects collective telemetry
through NCCL's profiler plugin API.

## Overview
CoMMA (Collective coMMunication Analyzer) is a library designed for use with NVIDIA’s NCCL profiler plugin API to collect NCCL telemetry  for use by Google Cloud services. It uses NCCL's profiler API, available since NCCL version 2.23, to extract and export detailed collective and network events from NCCL, including timelines of operations, data transfer sizes, and algorithm choices.

CoMMA is designed for low-overhead tracing, making it ideal for performance-sensitive and long-running machine learning workloads such as large language model (LLM) training.
For more information, see the [AI Hypercomputer documentation](https://cloud.google.com/ai-hypercomputer/docs/nccl/comma).

## Getting Started
CoMMA is often pre-installed as part of most GCP AI Hypercomputer container and OS images and is enabled by default. For more information, see the [AI Hypercomputer documentation](https://cloud.google.com/ai-hypercomputer/docs/nccl/comma).

If you use any of these OS or container images and want to disable CoMMA from collecting NCCL telemetry, see [Disable CoMMA](https://cloud.google.com/ai-hypercomputer/docs/nccl/configure-comma#disable-plugin).

### Installation
If you don't use any of these images and want to install CoMMA, use one of the following methods. For full installation instructions, see [AI Hypercomputer documentation](https://cloud.google.com/ai-hypercomputer/docs/nccl/comma).

| Installation method | Supported machine types |
|---------------------|-------------------------|
| Use NCCL gIB image (Recommended for newer machine types) | A4X, A4 High, and A3 Ultra |
| Use CoMMA installer image |  A4X, A4 High, and A3 Ultra |
| Build from source (Required for older machine types) | A3 Mega, A3 High, A3 Edge, A2 Ultra, A2 Standard, and N1 with attached GPUs |

### Using CoMMA outside of GCP
CoMMA could also be used on non-GCP environments with NCCL version >= v2.23. We recommend either using a CoMMA installer image or building from source for those use cases.

Note that CoMMA integrates with GCP services to enable GCP-specific features. When running on non-GCP environment, those integrations need to be disabled by setting the following environment variables:
- NCCL_PROFILER_USE_GPUVIZ=false

### Understanding CoMMA Output
You can view the raw data collected by exporting the CoMMA output to a local file. This could be done by setting the following environment varaibles:
- NCCL_PROFILER_LATENCY_FILE=/tmp/latency-%p.txt

For details on the JSON format, see [CoMMA JSON export format](docs/json-format.md).

For a complete list of configuration options, see [CoMMA Configuration Options](docs/configuration.md).

For detailed instructions on configuraing the granularity of telemetry export, see [AI Hypercomputer documentation](https://cloud.google.com/ai-hypercomputer/docs/nccl/comma).

### OpenTelemetry Support
CoMMA supports exporting metrics and (experimentally) traces via OpenTelemetry. For details on how to configure and use OTel support, see [OpenTelemetry Support in CoMMA](docs/otel-support.md).


## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## Licensing
CoMMA is licensed under the terms of the Apache license. See [LICENSE](LICENSE) for more information.
