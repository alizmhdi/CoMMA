// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::daemon::*;
use crate::event;
use crate::event::ProfilerEvent as _;
use crate::gcp_acs_proto;
use crate::gpuviz;
use crate::histogram::{Histogram1D, Histogram2D};
use crate::live_ring::{
    self, pack_str, RingSlot, RingWriter, SlotKind, DEFAULT_RING_CAPACITY, STEP_SOURCE_KERNEL,
    STEP_SOURCE_KERNEL_CH, STEP_SOURCE_PROXY, STEP_STAGE_COMPLETE, STEP_STAGE_POSTED,
};
use crate::nccl_metadata;
use crate::nccl_metadata::NcclOpKey;
use crate::otel_utils;
use crate::profiler::{Communicator, Profiler};
use crate::step_tracker::EventStep;

use gcp_acs_proto::ntc::ActiveCommunicator;
use gcp_acs_proto::ntc::ClosedCommunicator;

use log::{error, warn};
use opentelemetry::global::BoxedTracer as OtelTracer;
use opentelemetry::metrics::Gauge as OtelGauge;
use serde_json::json;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
pub struct CloudDaemon {
    rt: tokio::runtime::Runtime,
    worker: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    stop_signal: Option<oneshot::Sender<()>>,
}

impl Daemon for CloudDaemon {
    fn new(profiler: &'static Profiler) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (tx, rx) = oneshot::channel::<()>();
        let handle = rt.spawn(main_loop(profiler, rx));
        Self {
            rt,
            worker: Some(handle),
            stop_signal: Some(tx),
        }
    }
}

impl std::ops::Drop for CloudDaemon {
    fn drop(&mut self) {
        let stop_signal = self.stop_signal.take().unwrap();
        stop_signal.send(()).unwrap();
        let handle = self.worker.take().unwrap();
        let _ = self.rt.block_on(async { handle.await.unwrap() });
    }
}

impl Telemetry {
    async fn write_to_file<W, F>(&self, mut file: W, mut time_to_num: F) -> std::io::Result<()>
    where
        W: AsyncWriteExt + Unpin,
        F: FnMut(Instant) -> u64,
    {
        let mut buf: Vec<u8> = Vec::new();
        match self {
            Telemetry::Group(group) => {
                writeln!(buf, "{}", group.trace_record(&mut time_to_num))?;
            }
            Telemetry::NcclOpIssued(ncclop) => {
                writeln!(buf, "{}", ncclop.issued_trace_record(&mut time_to_num))?;
            }
            Telemetry::NcclOpComplete(ncclop) => {
                writeln!(
                    buf,
                    "{}",
                    ncclop.online_complete_trace_record(&mut time_to_num)
                )?;
            }
            Telemetry::NcclOp(ncclop) => {
                writeln!(buf, "{}", ncclop.trace_record(&mut time_to_num))?;
            }
            Telemetry::P2pParentStart(parent) => {
                writeln!(buf, "{}", parent.issued_trace_record(&mut time_to_num))?;
            }
            Telemetry::P2pGroupSeal(seal) => {
                writeln!(buf, "{}", seal.trace_record(&mut time_to_num))?;
            }
            Telemetry::P2pParent(parent) => {
                writeln!(buf, "{}", parent.trace_record(&mut time_to_num))?;
            }
            Telemetry::StepProgress(progress) => {
                writeln!(buf, "{}", progress.trace_record(&mut time_to_num))?;
            }
            Telemetry::ProxyOp(proxyop) => {
                writeln!(buf, "{}", proxyop.trace_record(&mut time_to_num))?;
            }
            Telemetry::CommInit(membership) => {
                writeln!(buf, "{}", membership.trace_record(&mut time_to_num))?;
            }
            _ => {}
        }
        file.write_all(&buf).await
    }
}

#[derive(Clone)]
struct ActiveComm {
    active_comm: ActiveCommunicator,
    total_net_bytes: Option<Arc<AtomicUsize>>,
}

struct CollectiveSummary {
    count: HashMap<NcclOpKey, Histogram1D>,
    latency: HashMap<NcclOpKey, Histogram2D>,

    active_comm: HashMap<u64, ActiveComm>,
    closed_comm: HashMap<u64, ClosedCommunicator>,

    plugin_major: Option<i32>,
    plugin_minor: Option<i32>,
}

impl CollectiveSummary {
    fn new() -> Self {
        Self {
            count: HashMap::new(),
            latency: HashMap::new(),

            active_comm: HashMap::new(),
            closed_comm: HashMap::new(),

            plugin_major: Some(env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0)),
            plugin_minor: Some(env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0)),
        }
    }

    fn record_op_issue(&mut self, op: &event::NcclOp) {
        if !op.is_p2p() {
            let key = op.op_key();
            let comm_hash = op.comm_hash();
            let comm = self
                .active_comm
                .entry(comm_hash)
                .or_insert_with(|| ActiveComm {
                    active_comm: ActiveCommunicator {
                        comm_hash: Some(comm_hash),
                        coll_seq: HashMap::new(),
                        ..Default::default()
                    },
                    total_net_bytes: None,
                });
            let active_comm = &mut comm.active_comm;
            active_comm.rank = Some(op.basic_info().rank() as _);
            let name = key.get_coll_op_type().name();
            let descr = op.get_coll_descr().unwrap();
            let algo = nccl_metadata::algo::name(descr.algo());
            active_comm
                .coll_seq
                .insert(String::from(name), descr.seq_num());
            active_comm
                .coll_last_algo
                .insert(String::from(name), String::from(algo));
            active_comm
                .coll_last_op_size
                .insert(String::from(name), op.byte_count() as _);
        }
    }

    fn record_comm_open(&mut self, comm: Communicator) {
        let comm_hash = comm.comm_hash;
        if comm_hash.is_none() {
            return;
        }
        let active_comm = self
            .active_comm
            .entry(comm_hash.unwrap())
            .or_insert_with(|| ActiveComm {
                active_comm: ActiveCommunicator {
                    comm_hash,
                    coll_seq: HashMap::new(),
                    ..Default::default()
                },
                total_net_bytes: None,
            });
        if active_comm.total_net_bytes.is_none() {
            active_comm.total_net_bytes = Some(comm.total_net_bytes);
        }
    }

    fn record_comm_close(&mut self, comm_hash: u64) {
        if let Some(active_comm) = self.active_comm.remove(&comm_hash) {
            self.closed_comm.insert(
                comm_hash,
                ClosedCommunicator {
                    comm_hash: Some(comm_hash),
                    rank: active_comm.active_comm.rank,
                    close_time: Some(std::time::SystemTime::now().into()),
                },
            );
        }
    }

    fn add_op(&mut self, op: &event::NcclOp) {
        let key = op.op_key();
        let byte_count = op.byte_count();
        let count_histo = self.count.entry(key.clone()).or_default();
        count_histo.add(byte_count as _, 1);

        if let Some(latency) = op.child_duration() {
            let latency_histo = self.latency.entry(key.clone()).or_default();
            latency_histo.add((byte_count as _, latency.as_nanos() as u64), 1);
        }
    }

    async fn write_to_file<W>(&mut self, file: &mut W) -> std::io::Result<()>
    where
        W: AsyncWriteExt + std::marker::Unpin,
    {
        let now = std::time::SystemTime::now();
        let mut buf: Vec<u8> = Vec::new();
        let mut coll_counts = Vec::new();
        for (key, histogram) in self.count.iter() {
            let entry = json!({
                "metadata": key.to_json(),
                "message_sizes": histogram.to_json(),
            });
            coll_counts.push(entry);
        }
        let coll_counts = json!({
            "name": "collective_counts",
            "time": now,
            "entries": coll_counts,
        });
        writeln!(
            buf,
            "{}",
            serde_json::to_string_pretty(&coll_counts).unwrap()
        )?;

        let mut latency_distro = Vec::new();
        for (key, histogram) in self.latency.iter() {
            let entry = json!({
                "metadata": key.to_json(),
                "latency_distribution": histogram.to_json(),
            });
            latency_distro.push(entry);
        }
        let latency_distro = json!({
            "name": "collective_latency",
            "time": now,
            "entries": latency_distro,
        });
        writeln!(
            buf,
            "{}",
            serde_json::to_string_pretty(&latency_distro).unwrap()
        )?;

        file.write_all(&buf).await
    }

    fn generate_heartbeat(
        &mut self,
        remote_comm_bytes: &Option<Arc<AtomicUsize>>,
    ) -> gcp_acs_proto::ntc::Event {
        use gcp_acs_proto::ntc::event::Body;

        let mut heartbeat = gcp_acs_proto::ntc::Heartbeat {
            active_comm: self
                .active_comm
                .values()
                .map(|comm| {
                    let mut active_comm = comm.active_comm.clone();
                    if let Some(size) = comm.total_net_bytes.as_ref() {
                        active_comm
                            .coll_net_bytes
                            .insert(String::from("total"), load_usize_as_u64(size));
                    }
                    active_comm
                })
                .collect(),
            closed_comm: std::mem::take(&mut self.closed_comm)
                .values()
                .cloned()
                .collect(),
            remote_comm_bytes: None,
        };

        if let Some(remote_comm_bytes) = remote_comm_bytes {
            heartbeat.remote_comm_bytes = Some(load_usize_as_u64(remote_comm_bytes));
        }

        gcp_acs_proto::ntc::Event {
            telemetry_type: Some("heartbeat".into()),
            event_creation_ts: Some(std::time::SystemTime::now().into()),
            plugin_version: Some(gcp_acs_proto::ntc::PluginVersion {
                major: self.plugin_major,
                minor: self.plugin_minor,
            }),
            body: Some(Body::Heartbeat(heartbeat)),
            ..Default::default()
        }
    }
}

fn load_usize_as_u64(val: &Arc<AtomicUsize>) -> u64 {
    let val = val.load(Ordering::Relaxed);
    match val.try_into() {
        Ok(total) => total,
        Err(e) => {
            error!("Failed to convert usize {} to u64. {}", val, e);
            0
        }
    }
}

async fn open_trace_file(path: impl AsRef<std::path::Path>) -> std::io::Result<tokio::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    // The container writes as root but the monitor typically runs as an
    // unprivileged user on the host. Without world-readable mode the monitor
    // cannot tail the NDJSON latency file. Force 0666; also fchmod after
    // open so the process umask cannot strip world/other read/write bits.
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o666)
        .open(path.as_ref())
        .await?;
    unsafe {
        use std::os::unix::io::AsRawFd;
        let _ = libc::fchmod(file.as_raw_fd(), 0o666);
    }
    Ok(file)
}

async fn build_bufwriter(
    path: impl AsRef<std::path::Path>,
) -> Option<tokio::io::BufWriter<tokio::fs::File>> {
    let r = open_trace_file(&path).await;
    match r {
        Ok(file) => Some(tokio::io::BufWriter::new(file)),
        Err(e) => {
            error!(
                "Failed to open file {:?} for logging telemetry: {}.",
                path.as_ref().as_os_str(),
                e
            );
            None
        }
    }
}

/// Create (or truncate) the per-pid live-telemetry ring file. Returns
/// `None` if the writer could not be created; the caller then continues
/// without the ring and only the optional NDJSON `latency_file` is written.
fn open_live_ring(dir: &std::path::Path, pid: u32) -> Option<RingWriter> {
    if let Err(e) = std::fs::create_dir_all(dir) {
        error!(
            "Failed to prepare live-ring dir {:?}: {}. Live telemetry will be dropped.",
            dir.as_os_str(),
            e
        );
        return None;
    }
    let path: PathBuf = dir.join(format!("ring-{pid}.ring"));
    match RingWriter::create(&path, DEFAULT_RING_CAPACITY, pid) {
        Ok(w) => {
            log::info!(
                "live-ring open: pid={pid} capacity={} path={}",
                DEFAULT_RING_CAPACITY,
                path.display()
            );
            Some(w)
        }
        Err(e) => {
            error!(
                "Failed to create live-ring file {:?}: {}. Live telemetry will be dropped.",
                path.as_os_str(),
                e
            );
            None
        }
    }
}

/// Try-publish a telemetry record on the ring. Returns the number of slots
/// consumed (0 when the record is not a live-lane event). Never blocks: a
/// full ring bumps the ring's `dropped` counter and the record is discarded.
fn ring_publish(
    ring: &mut RingWriter,
    telemetry: &Telemetry,
    profiler: &'static Profiler,
) -> usize {
    let to_us = |t: Instant| -> i64 { profiler.instant_to_timestamp(t).as_micros() as i64 };
    let attempt = |ring: &mut RingWriter, slot: &RingSlot| -> usize {
        match ring.push(slot) {
            Ok(()) => 1,
            Err(()) => 0,
        }
    };
    match telemetry {
        Telemetry::NcclOpIssued(op) => {
            let slot = encode_ncclop_issue(op, to_us);
            attempt(ring, &slot)
        }
        Telemetry::NcclOpComplete(op) => {
            let slot = encode_ncclop_complete(op, to_us);
            attempt(ring, &slot)
        }
        Telemetry::P2pParentStart(parent) => {
            let slot = encode_p2p_group_start(parent, to_us);
            attempt(ring, &slot)
        }
        Telemetry::P2pGroupSeal(seal) => {
            let slot = encode_p2p_group_seal(seal, to_us);
            attempt(ring, &slot)
        }
        Telemetry::StepProgress(progress) => {
            let slot = encode_step_progress(progress, to_us);
            attempt(ring, &slot)
        }
        Telemetry::CommInit(membership) => encode_comm_init(ring, membership, to_us),
        Telemetry::KernelCopy(c) => attempt(ring, &encode_kernel_copy(c, to_us)),
        // The following variants are intentionally *not* published on the
        // live ring:
        //  - CommOpen / CommClose: currently only affect the summary
        //    heartbeat; the monitor does not consume them.
        //  - Rich Group / NcclOp / P2pParent / ProxyOp: full offline traces
        //    with nested Vec<> children; they may still be written to the
        //    optional NCCL_PROFILER_LATENCY_FILE NDJSON.
        _ => 0,
    }
}

fn encode_ncclop_issue(op: &event::NcclOp, to_us: impl Fn(Instant) -> i64) -> RingSlot {
    let basic = op.basic_info();
    let mut slot = RingSlot::zeroed();
    slot.ts = to_us(basic.start_time());
    slot.rank = basic.rank() as i32;
    slot.comm_hash = op.comm_hash();
    slot.payload_bytes = op.byte_count() as i64;
    if let Some(coll) = op.get_descr().try_cast_to_coll() {
        slot.kind = SlotKind::CollStart as u8;
        pack_str(&mut slot.name, coll.op_type().name());
        slot.seq_num = coll.seq_num() as i64;
    } else if let Some(p2p) = op.get_descr().try_cast_to_p2p() {
        slot.kind = SlotKind::P2pStart as u8;
        pack_str(&mut slot.name, if p2p.is_send() { "send" } else { "recv" });
        slot.is_send = p2p.is_send() as u8;
        slot.peer = p2p.peer();
        slot.group_id = op.parent_group_id().unwrap_or(0);
    }
    slot
}

fn encode_ncclop_complete(op: &event::NcclOp, to_us: impl Fn(Instant) -> i64) -> RingSlot {
    let basic = op.basic_info();
    let start = basic.start_time();
    let end = basic.end_time().unwrap_or(start);
    let mut slot = RingSlot::zeroed();
    slot.ts = to_us(start);
    slot.duration_us = (end - start).as_micros() as i64;
    slot.rank = basic.rank() as i32;
    slot.comm_hash = op.comm_hash();
    slot.payload_bytes = op.byte_count() as i64;
    if let Some(coll) = op.get_descr().try_cast_to_coll() {
        slot.kind = SlotKind::CollComplete as u8;
        pack_str(&mut slot.name, coll.op_type().name());
        slot.seq_num = coll.seq_num() as i64;
    } else if let Some(p2p) = op.get_descr().try_cast_to_p2p() {
        slot.kind = SlotKind::P2pComplete as u8;
        pack_str(&mut slot.name, if p2p.is_send() { "send" } else { "recv" });
        slot.is_send = p2p.is_send() as u8;
        slot.peer = p2p.peer();
        slot.group_id = op.parent_group_id().unwrap_or(0);
    }
    slot
}

fn encode_p2p_group_start(parent: &event::P2pParent, to_us: impl Fn(Instant) -> i64) -> RingSlot {
    let mut slot = RingSlot::zeroed();
    slot.kind = SlotKind::P2pGroupStart as u8;
    slot.ts = to_us(parent.start_time);
    slot.rank = parent.rank as i32;
    slot.comm_hash = parent.comm_hash;
    slot.group_id = parent.group_id;
    slot.n_children = parent.n_children as u32;
    slot.n_peers = parent.n_peers as u32;
    slot.payload_bytes = parent.size as i64;
    slot.self_copy_size = parent.self_copy_size as i64;
    pack_str(&mut slot.name, parent.name);
    slot
}

fn encode_kernel_copy(c: &KernelCopySummary, to_us: impl Fn(Instant) -> i64) -> RingSlot {
    let mut slot = RingSlot::zeroed();
    slot.kind = SlotKind::KernelCopy as u8;
    slot.ts = to_us(c.observed_at);
    slot.rank = c.rank as i32;
    slot.payload_bytes = c.bytes;
    slot.step = c.copy_ns;
    slot.duration_us = c.copy_ns / 1000;
    slot.n_children = c.steps;
    pack_str(&mut slot.name, "KernelCopy");
    slot
}

fn encode_p2p_group_seal(seal: &event::P2pGroupSeal, to_us: impl Fn(Instant) -> i64) -> RingSlot {
    let mut slot = RingSlot::zeroed();
    slot.kind = SlotKind::P2pGroupSeal as u8;
    slot.ts = to_us(seal.observed_at);
    slot.group_id = seal.group_id;
    slot.n_children = seal.n_children as u32;
    pack_str(&mut slot.name, "P2P_GROUP_SEAL");
    slot
}

fn encode_step_progress(
    progress: &event::StepProgress,
    to_us: impl Fn(Instant) -> i64,
) -> RingSlot {
    let mut slot = RingSlot::zeroed();
    slot.kind = SlotKind::StepProgress as u8;
    slot.source = match progress.source {
        event::StepProgressSource::Proxy => STEP_SOURCE_PROXY,
        event::StepProgressSource::Kernel => STEP_SOURCE_KERNEL,
        event::StepProgressSource::KernelCh => STEP_SOURCE_KERNEL_CH,
    };
    slot.stage = match progress.stage {
        event::StepProgressStage::Posted => STEP_STAGE_POSTED,
        event::StepProgressStage::Complete => STEP_STAGE_COMPLETE,
    };
    slot.ts = to_us(progress.observed_at);
    slot.launch_ts = to_us(progress.parent_start);
    slot.rank = progress.rank as i32;
    slot.comm_hash = progress.comm_hash;
    slot.seq_num = progress.seq_num;
    slot.group_id = progress.parent_group_id.unwrap_or(0);
    slot.is_send = progress.is_send as u8;
    slot.peer = progress.peer as i32;
    slot.channel = progress.channel_id as i32;
    slot.step = progress.step;
    slot.size = progress.size as i64;
    let name = match progress.source {
        event::StepProgressSource::Proxy => "ProxyStep",
        event::StepProgressSource::Kernel => "KernelStep",
        event::StepProgressSource::KernelCh => "KernelCh",
    };
    pack_str(&mut slot.name, name);
    pack_str(&mut slot.parent_name, &progress.parent_name);
    slot
}

/// Fans a `CommInit` out into one header slot plus one member slot per
/// entry. Returns the number of slots that actually reached the ring.
fn encode_comm_init(
    ring: &mut RingWriter,
    membership: &crate::profiler::CommMembership,
    to_us: impl Fn(Instant) -> i64,
) -> usize {
    let mut header = RingSlot::zeroed();
    header.kind = SlotKind::CommInitHeader as u8;
    header.ts = to_us(membership.ts);
    header.rank = membership.rank;
    header.world_rank = membership.world_rank;
    header.comm_hash = membership.comm_hash;
    header.n_nodes = membership.n_nodes;
    header.n_ranks = membership.n_ranks;
    pack_str(&mut header.comm_name, &membership.comm_name);
    pack_str(&mut header.host, &membership.host);
    pack_str(&mut header.ip, &membership.ip);

    let mut written = 0usize;
    if ring.push(&header).is_ok() {
        written += 1;
    } else {
        // If the header did not make it, do not fan out the members; a
        // half-init dump only confuses the consumer.
        return 0;
    }
    for m in &membership.members {
        let mut s = RingSlot::zeroed();
        s.kind = SlotKind::CommInitMember as u8;
        s.ts = header.ts;
        s.rank = membership.rank;
        s.comm_hash = membership.comm_hash;
        s.member_rank = m.rank;
        s.member_world_rank = m.world_rank;
        s.member_node = m.node;
        s.member_cuda_dev = m.cuda_dev;
        s.member_host_hash = m.host_hash;
        s.member_bus_id = m.bus_id;
        if ring.push(&s).is_ok() {
            written += 1;
        } else {
            // Ring is full mid-fan-out. Consumer will time out this partial
            // COMM_INIT and drop it; do not spin forever waiting for room.
            break;
        }
    }
    written
}

async fn exporter(
    profiler: &'static Profiler,
    mut rx: mpsc::Receiver<Telemetry>,
    mut online_rx: mpsc::Receiver<Telemetry>,
    mut deadline_rx: mpsc::Receiver<Telemetry>,
    otel_latency_hist_manager: Option<Arc<Mutex<otel_utils::HistogramManager>>>,
) -> std::io::Result<()> {
    let mut latency_file = if let Some(template) = profiler.config.latency_file.as_ref() {
        let path = template.replace("%p", &format!("{}", profiler.pid));
        build_bufwriter(path).await
    } else {
        None
    };

    // Live-telemetry file-backed SPSC ring. This is the only live-path
    // transport; the deprecated NDJSON Unix stream socket has been removed.
    let mut live_ring: Option<RingWriter> =
        profiler
            .config
            .latency_ring_dir
            .as_ref()
            .and_then(|dir_template| {
                let dir_str = dir_template.replace("%p", &format!("{}", profiler.pid));
                open_live_ring(std::path::Path::new(&dir_str), profiler.pid as u32)
            });
    if live_ring.is_none() && profiler.config.latency_ring_dir.is_some() {
        warn!("live-ring requested but not opened; online monitor will see nothing.");
    }

    let mut summary_file = if let Some(template) = profiler.config.summary_file.as_ref() {
        let path = template.replace("%p", &format!("{}", profiler.pid));
        build_bufwriter(path).await
    } else {
        None
    };

    let mut summary = if summary_file.is_some() || profiler.config.heartbeat {
        Some(CollectiveSummary::new())
    } else {
        None
    };

    let mut gpuviz: Option<gpuviz::HistogramManager> = profiler
        .gpuviz_lib
        .clone()
        .map(gpuviz::HistogramManager::new);

    let mut otel_tracer: Option<OtelTracer> =
        if profiler.config.otel_enable && profiler.config.otel_trace_ncclop {
            Some(opentelemetry::global::tracer("CoMMA"))
        } else {
            None
        };

    let mut otel_seqnum_gauge: Option<OtelGauge<i64>> = if profiler.config.otel_enable {
        let meter = opentelemetry::global::meter("nccl");
        Some(meter.i64_gauge("nccl.collective.seq_num").build())
    } else {
        None
    };

    let mut summary_interval = tokio::time::interval(profiler.config.summary_interval);

    // the very first tick completes immediately
    summary_interval.tick().await;

    let mut uploader_interval = tokio::time::interval(profiler.config.heartbeat_upload_interval);
    uploader_interval.tick().await;

    // Periodic latency-file flush for online consumers (straggler monitor).
    let mut latency_flush_interval =
        if profiler.config.latency_flush_interval > Duration::from_secs(0) {
            let mut i = tokio::time::interval(profiler.config.latency_flush_interval);
            i.tick().await;
            Some(i)
        } else {
            None
        };

    let mut otel_metrics_grouping_interval = if profiler.config.otel_enable {
        let mut i =
            tokio::time::interval(profiler.config.otel_metrics_cardinality_grouping_interval);
        i.tick().await;
        Some(i)
    } else {
        None
    };

    let mut regular_closed = false;
    let mut online_closed = false;
    let mut deadline_closed = false;
    loop {
        tokio::select! {
            // Deadline lifecycle has its own biased lane. High-volume RDMA
            // timestamps and progress remain online, but cannot delay parent
            // start/seal/completion records that drive the monitor clock.
            biased;
            online_item = async {
                tokio::select! {
                    biased;
                    item = deadline_rx.recv(), if !deadline_closed => (true, item),
                    item = online_rx.recv(), if !online_closed => (false, item),
                }
            }, if !deadline_closed || !online_closed => {
                let (from_deadline_lane, maybe_telemetry) = online_item;
                match maybe_telemetry {
                    Some(telemetry) => {
                        if let Some(file) = latency_file.as_mut() {
                            let r = telemetry
                                .write_to_file(
                                    file,
                                    |t| profiler.instant_to_timestamp(t).as_micros() as _)
                                .await;
                            if let Err(e) = r {
                                error!("Failed to log latency telemetry to file: {}. Stop logging.", e);
                                latency_file = None;
                            }
                        }

                        // Live telemetry: encode into packed POD slots and
                        // push onto the file-backed SPSC ring. Non-blocking:
                        // a full ring bumps the ring's `dropped` counter.
                        if let Some(ring) = live_ring.as_mut() {
                            let _ = ring_publish(ring, &telemetry, profiler);
                        }

                        if let Some(summary) = summary.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    summary.add_op(op);
                                },
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        summary.add_op(op);
                                    }
                                },
                                Telemetry::NcclOpIssued(op) => {
                                    summary.record_op_issue(op);
                                },
                                Telemetry::CommOpen(comm) => {
                                    summary.record_comm_open(comm.clone());
                                }
                                Telemetry::CommClose(comm_hash) => {
                                    summary.record_comm_close(*comm_hash);
                                },
                                _ => {}
                            }
                        }

                        if let Some(gpuviz) = gpuviz.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    let _ = gpuviz.add_ncclop(op, |t| {
                                        profiler.instant_to_timestamp(*t).as_nanos() as _
                                    });
                                }
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        let _ = gpuviz.add_ncclop(op, |t| {
                                            profiler.instant_to_timestamp(*t).as_nanos() as _
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }

                        if let Some(otel_tracer) = otel_tracer.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    let _ = otel_utils::add_ncclop_trace(otel_tracer, profiler, op);
                                }
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        let _ = otel_utils::add_ncclop_trace(otel_tracer, profiler, op);
                                    }
                                }
                                _ => {}
                            }
                        }

                        if let Some(gauge) = otel_seqnum_gauge.as_mut() {
                            if let Telemetry::NcclOpIssued(op) = &telemetry {
                                let _ = otel_utils::record_ncclop_seqnum(gauge, profiler, op);
                            }
                        }
                    },
                    None => {
                        if from_deadline_lane {
                            deadline_closed = true;
                        } else {
                            online_closed = true;
                        }
                    },
                }
            },
            maybe_telemetry = rx.recv(), if !regular_closed => {
                match maybe_telemetry {
                    Some(telemetry) => {
                        if let Some(file) = latency_file.as_mut() {
                            let r = telemetry
                                .write_to_file(
                                    file,
                                    |t| profiler.instant_to_timestamp(t).as_micros() as _)
                                .await;
                            if let Err(e) = r {
                                error!("Failed to log latency telemetry to file: {}. Stop logging.", e);
                                latency_file = None;
                            }
                        }

                        if let Some(summary) = summary.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    summary.add_op(op);
                                },
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        summary.add_op(op);
                                    }
                                },
                                Telemetry::NcclOpIssued(op) => {
                                    summary.record_op_issue(op);
                                },
                                Telemetry::CommOpen(comm) => {
                                    summary.record_comm_open(comm.clone());
                                }
                                Telemetry::CommClose(comm_hash) => {
                                    summary.record_comm_close(*comm_hash);
                                },
                                _ => {}
                            }
                        }

                        if let Some(gpuviz) = gpuviz.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    let _ = gpuviz.add_ncclop(op, |t| {
                                        profiler.instant_to_timestamp(*t).as_nanos() as _
                                    });
                                }
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        let _ = gpuviz.add_ncclop(op, |t| {
                                            profiler.instant_to_timestamp(*t).as_nanos() as _
                                        });
                                    }
                                }
                                _ => {}
                            }
                        }

                        if let Some(otel_tracer) = otel_tracer.as_mut() {
                            match &telemetry {
                                Telemetry::NcclOp(op) => {
                                    let _ = otel_utils::add_ncclop_trace(otel_tracer, profiler, op);
                                }
                                Telemetry::P2pParent(parent) => {
                                    for op in &parent.children {
                                        let _ = otel_utils::add_ncclop_trace(otel_tracer, profiler, op);
                                    }
                                }
                                _ => {}
                            }
                        }

                        if let Some(gauge) = otel_seqnum_gauge.as_mut() {
                            if let Telemetry::NcclOpIssued(op) = &telemetry {
                                let _ = otel_utils::record_ncclop_seqnum(gauge, profiler, op);
                            }
                        }
                    },
                    None => regular_closed = true,
                }
            },
            _ = async { latency_flush_interval.as_mut().unwrap().tick().await },
                    if latency_flush_interval.is_some() && latency_file.is_some() => {
                if let Some(file) = latency_file.as_mut() {
                    if let Err(e) = file.flush().await {
                        error!("Failed to flush latency telemetry file: {}. Stop logging.", e);
                        latency_file = None;
                    }
                }
            },
            _ = summary_interval.tick(), if summary.is_some() => {
                if let Some(file) = summary_file.as_mut() {
                    let s = summary.as_mut().unwrap();
                    let r = s.write_to_file(file).await;
                    if let Err(e) = r {
                        error!("Failed to log telemetry summary to file: {}. Stop logging.", e);
                    }
                }
            },
            _ = uploader_interval.tick(), if profiler.config.heartbeat => {
                if summary.is_some() && gpuviz.is_some() {
                    let summary = summary.as_mut().unwrap();
                    let gpuviz = gpuviz.as_mut().unwrap();
                    gpuviz.send_heartbeat(summary.generate_heartbeat(&profiler.remote_net_bytes));
                }
            },
            _ = async { otel_metrics_grouping_interval.as_mut().unwrap().tick().await }, if otel_metrics_grouping_interval.is_some() => {
                if let Some(manager) = otel_latency_hist_manager.as_ref() {
                    if let Ok(mut inner) = manager.lock() {
                        inner.update_priority();
                    }
                }
            },
        }
        // Flush pending live-ring writes on every polling boundary so the
        // consumer's cache-visible write head does not lag by a batch.
        if let Some(ring) = live_ring.as_mut() {
            ring.flush();
        }
        if regular_closed && online_closed && deadline_closed {
            break;
        }
    }

    if let Some(ring) = live_ring.as_mut() {
        ring.close();
    }

    if let Some(file) = latency_file.as_mut() {
        file.flush().await?;
    }

    if let Some(summary) = summary.as_mut() {
        if let Some(file) = summary_file.as_mut() {
            summary.write_to_file(file).await?;
            file.flush().await?;
        }
        if profiler.config.heartbeat {
            if let Some(gpuviz) = gpuviz.as_mut() {
                gpuviz.send_heartbeat(summary.generate_heartbeat(&profiler.remote_net_bytes));
            }
        }
    }

    Ok(())
}

struct Exporter {
    tx: mpsc::Sender<Telemetry>,
    online_tx: Option<mpsc::Sender<Telemetry>>,
    deadline_tx: Option<mpsc::Sender<Telemetry>>,
    otel_latency_hist_manager: Option<Arc<Mutex<otel_utils::HistogramManager>>>,
    gpuviz: Option<Mutex<gpuviz::HistogramManager<Arc<gpuviz::Connection>>>>,
    track_step_fifo_wait: bool,
}

impl Exporter {
    #[cfg(test)]
    fn new(tx: mpsc::Sender<Telemetry>) -> Self {
        Self {
            tx,
            online_tx: None,
            deadline_tx: None,
            otel_latency_hist_manager: None,
            gpuviz: None,
            track_step_fifo_wait: false,
        }
    }
}

impl Export for Exporter {
    fn export(&self, ctx: &mut PollingContext, maybe_retry_ms: Option<u64>) {
        // Scan each currently pending item once. If the bulk queue is full,
        // rotate that item to the back so a compact online record behind it
        // can still reach the priority lane in this polling pass.
        let mut remaining = ctx.pending_telemetry.len();
        while remaining > 0 {
            remaining -= 1;
            let Some(telemetry) = ctx.pending_telemetry.pop_front() else {
                break;
            };
            let deadline = is_deadline_telemetry(&telemetry);
            let online = deadline || is_online_evidence(&telemetry);
            let tx = if deadline {
                self.deadline_tx
                    .as_ref()
                    .or(self.online_tx.as_ref())
                    .unwrap_or(&self.tx)
            } else if online {
                self.online_tx.as_ref().unwrap_or(&self.tx)
            } else {
                &self.tx
            };
            if let Err(err) = tx.try_send(telemetry) {
                match err {
                    TrySendError::Full(v) => {
                        if deadline || maybe_retry_ms.is_some() {
                            ctx.pending_telemetry.push_front(v);
                            if let Some(ms) = maybe_retry_ms {
                                std::thread::sleep(Duration::from_micros(ms));
                            }
                            break;
                        } else {
                            ctx.pending_telemetry.push_back(v);
                        }
                    }
                    _ => {
                        static ONETIME_LOG: Once = Once::new();
                        ONETIME_LOG.call_once(|| {
                            error!(
                                "Channel to telemetry exporter is unexpectedly closed. \
                                All future telemetry will be dropped."
                            );
                        });
                    }
                }
            }
        }
    }

    fn get_latency_histogram(
        &self,
        key: &NcclOpKey,
    ) -> Option<impl std::iter::IntoIterator<Item = Arc<dyn AtomicHistogram<EventStep>>>> {
        let mut histograms: Vec<Arc<dyn AtomicHistogram<EventStep>>> = Vec::new();
        if let Some(manager) = self.otel_latency_hist_manager.as_ref() {
            if let Ok(mut lg) = manager.lock() {
                let h = Arc::new(lg.get_histogram(key.clone()));
                histograms.push(h as _);
            }
        }
        if let Some(gpuviz_mutex) = self.gpuviz.as_ref() {
            if let Ok(mut gpuviz) = gpuviz_mutex.lock() {
                if self.track_step_fifo_wait {
                    if let Ok(h) = gpuviz.get_connection(key, Some(gpuviz::ConnectionType::Net)) {
                        histograms.push(h.clone() as _);
                    }
                    if let Ok(h) = gpuviz.get_connection(key, Some(gpuviz::ConnectionType::E2e)) {
                        histograms.push(h.clone() as _);
                    }
                    if let Ok(h) = gpuviz.get_connection(key, Some(gpuviz::ConnectionType::Cts)) {
                        histograms.push(h.clone() as _);
                    }
                } else if let Ok(h) = gpuviz.get_connection(key, None) {
                    histograms.push(h.clone() as _);
                }
            }
        }
        if histograms.is_empty() {
            None
        } else {
            Some(histograms)
        }
    }
}

fn is_deadline_telemetry(telemetry: &Telemetry) -> bool {
    matches!(
        telemetry,
        Telemetry::NcclOpIssued(_)
            | Telemetry::P2pParentStart(_)
            | Telemetry::P2pGroupSeal(_)
            | Telemetry::NcclOpComplete(_)
            | Telemetry::CommOpen(_)
            | Telemetry::CommInit(_)
            | Telemetry::CommClose(_)
    )
}

fn is_online_evidence(telemetry: &Telemetry) -> bool {
    matches!(
        telemetry,
        Telemetry::StepProgress(_) | Telemetry::KernelCopy(_)
    )
}

#[cfg(test)]
mod deadline_lane_tests {
    use super::*;

    #[test]
    fn p2p_seal_uses_deadline_lane() {
        let seal = Telemetry::P2pGroupSeal(event::P2pGroupSeal {
            group_id: 7,
            n_children: 4,
            observed_at: Instant::now(),
        });
        assert!(is_deadline_telemetry(&seal));
        assert!(!is_online_evidence(&seal));
    }
}

async fn main_loop(
    profiler: &'static Profiler,
    stop: oneshot::Receiver<()>,
) -> std::io::Result<()> {
    if profiler.config.otel_enable {
        if otel_utils::init_meter_provider(&profiler.config).is_none() {
            log::warn!("failed to init otel meter provider");
        }
        if otel_utils::init_tracer_provider(&profiler.config).is_none() {
            log::warn!("failed to init otel tracer provider");
        }
    }
    const TELEMETRY_CHANNEL_SZ: usize = 4096;
    let (tx, rx) = mpsc::channel::<Telemetry>(TELEMETRY_CHANNEL_SZ);
    const ONLINE_CHANNEL_SZ: usize = 1024;
    let (online_tx, online_rx) = mpsc::channel::<Telemetry>(ONLINE_CHANNEL_SZ);
    const DEADLINE_CHANNEL_SZ: usize = 256;
    let (deadline_tx, deadline_rx) = mpsc::channel::<Telemetry>(DEADLINE_CHANNEL_SZ);
    let otel_latency_hist_manager = if profiler.config.otel_enable {
        Some(Arc::new(Mutex::new(otel_utils::HistogramManager::new(
            "nccl.net_send.latency",
            "ns",
            profiler.config.otel_metrics_max_cardinality,
        ))))
    } else {
        None
    };
    let export_worker = tokio::task::spawn(exporter(
        profiler,
        rx,
        online_rx,
        deadline_rx,
        otel_latency_hist_manager.clone(),
    ));
    let exporter = Exporter {
        tx,
        online_tx: Some(online_tx),
        deadline_tx: Some(deadline_tx),
        otel_latency_hist_manager,
        gpuviz: profiler
            .gpuviz_lib
            .as_ref()
            .map(|lib| Mutex::new(gpuviz::HistogramManager::new(lib.clone()))),
        track_step_fifo_wait: profiler.config.track_step_fifo_wait,
    };

    let stop_signal = Arc::new(AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();
    let polling_worker = tokio::task::spawn_blocking(move || {
        let mut polling_ctx = PollingContext::new(profiler, stop_signal_clone);
        polling_loop(&mut polling_ctx, exporter)
    });

    let stop_signal_copy = stop_signal.clone();
    let timer_tick = if profiler.config.use_cached_clock {
        Some(std::thread::spawn(move || {
            while !stop_signal_copy.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_micros(5));
                profiler.cached_clock.update_cache(Instant::now());
            }
        }))
    } else {
        None
    };

    stop.await.unwrap();
    stop_signal.store(true, Ordering::Release);
    polling_worker.await?;
    if let Some(t) = timer_tick {
        let _ = t.join();
    }
    export_worker.await??;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::config;
    use crate::daemon;
    use crate::nccl_metadata::Version as _;
    use crate::profiler::{thread_local_state, ThreadLocalState, Version};
    use crate::slab;
    use crate::step_tracker;
    use crate::{profiler_shim, scoped_profiler_test};

    use std::io::BufRead;
    use std::sync::Mutex;

    // create a mutex to avoid multiple test cases allocating ncclops concurrently
    static NCCLOP_TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn e2e_mock() {
        let _lg = NCCLOP_TEST_MUTEX.lock().unwrap();

        const N_PROXYOP: usize = 1 << 16;

        let temp_dir = tempfile::tempdir().unwrap();
        let temp_dir_path: &str = temp_dir.as_ref().to_str().unwrap();
        let pid = 42;
        let latency_template = format!("{}/latency-%p.txt", temp_dir_path);
        let summary_template = format!("{}/summary-%p.txt", temp_dir_path);
        // create a mock Profiler object
        let mut profiler = Profiler::new(Version::V1);
        profiler.pid = pid;
        profiler.config.track_group = true;
        profiler.config.track_ncclop = true;
        profiler.config.track_proxyop = true;
        profiler.config.track_interprocess_proxyop = false;
        profiler.config.latency_file = Some(latency_template.clone());
        profiler.config.summary_file = Some(summary_template.clone());
        // Profiler::new snapshots the startup config into runtime gates before
        // this test overrides its config fields.
        profiler
            .gates
            .apply_update(&crate::runtime_gates::GateUpdate {
                track_proxyop: Some(true),
                ..Default::default()
            });
        scoped_profiler_test(profiler, |_profiler, thread_state| {
            let group_descr = profiler_shim::tests::dummy_group_descr();
            let coll_descr = profiler_shim::tests::dummy_coll_descr();
            let proxyop_descr = profiler_shim::tests::dummy_proxyop_descr();

            let group = Box::new(event::Group::from_descr(&group_descr, Instant::now()));
            let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                &coll_descr,
                Instant::now(),
                /* id = */ 0,
                /* comm_hash_override = */ None,
            ));
            let proxyop_descr_casted = unsafe { proxyop_descr.cast_to_proxyop() };
            let proxyops: Vec<_> = (0..N_PROXYOP)
                .map(|id| {
                    event::ProxyOpInfo::from_descr(
                        proxyop_descr_casted,
                        id as u32,
                        Some(Communicator::new()),
                    )
                })
                .collect();

            thread_state.send_to_daemon(Message::Group(group), true);
            thread_state.send_to_daemon(Message::NcclOp(coll), true);

            let mut step_batch_list = slab::FreeList::new_list(proxyops.len());
            for proxyop in proxyops.into_iter() {
                let mut step_batch = step_batch_list
                    .alloc(
                        |p| unsafe { daemon::StepBatch::init(p) },
                        Some(&thread_state.profiler.free_step_batch),
                        false,
                    )
                    .unwrap();
                step_batch.push(step_tracker::EventStep {
                    step: 0,
                    size: 65536,
                    start_time: 123,
                    fifo_wait_dur_ns: None,
                    dur_ns: 256,
                });
                thread_state
                    .send_to_daemon(Message::StepBatch(proxyop.clone(), step_batch, true), true);
            }
        });

        // validate by opening the file and check if it is empty
        // we don't check content here as that would be too coupled
        // with implementation details
        let templates = [&latency_template, &summary_template];
        for t in templates {
            let path = t.replace("%p", &format!("{}", pid));
            let file = std::fs::File::open(path).unwrap();
            assert!(file.metadata().unwrap().len() > 0);
        }

        let path = latency_template.replace("%p", &format!("{}", pid));
        let file = std::fs::File::open(path).unwrap();
        let mut to_find = [
            ("\"COLL\"", false),
            ("\"GROUP\"", false),
            ("\"PROXY\"", false),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<&'static str, bool>>();

        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            let line = line.unwrap();
            for (key, b) in to_find.iter_mut() {
                if line.contains(key) {
                    *b = true;
                }
            }
        }

        assert!(
            to_find.iter().all(|t| *t.1),
            "missing telemetry categories: {:?}",
            to_find
                .iter()
                .filter_map(|(category, found)| (!found).then_some(*category))
                .collect::<Vec<_>>()
        );
    }

    fn reclaim_test_template<F>(n_ncclop: usize, client: F)
    where
        F: FnOnce(
                &mut ThreadLocalState,
                &std::sync::Barrier,
                mpsc::Receiver<Telemetry>,
                Arc<AtomicBool>,
            ) + Send,
    {
        let _lg = NCCLOP_TEST_MUTEX.lock().unwrap();

        let profiler = Box::new(Profiler::new(Version::V1));

        std::thread::scope(|s| {
            let (tx, rx) = mpsc::channel::<Telemetry>(n_ncclop * 2);
            let stop_var = Arc::new(AtomicBool::new(false));
            let stop_var_clone = stop_var.clone();
            let barrier = Arc::new(std::sync::Barrier::new(2));
            let barr = barrier.clone();
            let profiler_ref = &profiler;
            s.spawn(move || {
                let mut ctx = PollingContext::new(profiler_ref, stop_var_clone);
                polling_loop(&mut ctx, Exporter::new(tx));
                barr.wait();
                assert_eq!(ctx.ncclops.len(), 0);
                barr.wait();
            });

            let profiler_ref = &profiler;
            s.spawn(move || {
                let (mut thread_state, control) = thread_local_state(profiler_ref);
                profiler_ref.register_thread(control);
                client(&mut thread_state, &barrier, rx, stop_var);
                barrier.wait();
            });
        });
    }

    #[test]
    fn reclaim_eventual() {
        const N_NCCLOP: usize = 50;

        reclaim_test_template(N_NCCLOP, |thread_state, barrier, mut rx, stop_var| {
            for op_idx in 0..N_NCCLOP {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    Instant::now(),
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            let mut n_reported = 0;
            while let Some(t) = rx.blocking_recv() {
                if std::matches!(t, daemon::Telemetry::NcclOp(_)) {
                    n_reported += 1;
                }
            }
            assert_eq!(n_reported, N_NCCLOP);
        });
    }

    #[test]
    fn reclaim_timeout() {
        let config: &config::Config = &config::CONFIG;
        let n_ncclop: usize = config.max_tracked_ncclop - 1;
        const NCCLOP_TIMEOUT: Duration = Duration::from_secs(100);

        reclaim_test_template(n_ncclop, |thread_state, barrier, mut rx, stop_var| {
            for op_idx in 0..n_ncclop {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    Instant::now() - NCCLOP_TIMEOUT,
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }

            // block until rx is not empty
            let t0 = Instant::now();
            while rx.is_empty() && t0.elapsed().as_secs() < 10 {
                std::thread::sleep(Duration::from_micros(10));
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            let mut n_reported = 0;
            while let Some(t) = rx.blocking_recv() {
                if std::matches!(t, daemon::Telemetry::NcclOp(_)) {
                    n_reported += 1;
                }
            }
            assert_eq!(n_reported, n_ncclop);
        });
    }

    #[test]
    fn reclaim_force() {
        let config: &config::Config = &config::CONFIG;
        let n_ncclop: usize = config.max_tracked_ncclop + 1000;

        reclaim_test_template(n_ncclop, |thread_state, barrier, mut rx, stop_var| {
            for op_idx in 0..n_ncclop {
                let coll_descr = profiler_shim::tests::dummy_coll_descr();
                let coll = slab::AllocatedNode::new(event::NcclOp::from_descr(
                    &coll_descr,
                    // use a late start time to make sure timeout does not happen
                    Instant::now() + Duration::from_secs(3600),
                    op_idx,
                    /* comm_hash_override= */ None,
                ));
                thread_state.send_to_daemon(Message::NcclOp(coll), true);
            }

            // block until rx is not empty
            let t0 = Instant::now();
            while rx.is_empty() && t0.elapsed().as_secs() < 10 {
                std::thread::sleep(Duration::from_micros(10));
            }
            stop_var.store(true, Ordering::Release);
            barrier.wait();
            let mut n_reported = 0;
            while let Some(t) = rx.blocking_recv() {
                if std::matches!(t, daemon::Telemetry::NcclOp(_)) {
                    n_reported += 1;
                }
            }
            assert_eq!(n_reported, n_ncclop);
        });
    }
}
