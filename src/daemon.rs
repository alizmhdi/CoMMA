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

use crate::event;
use crate::event::ProfilerEvent as _;
use crate::fixed_batch;
use crate::nccl_metadata;
use crate::nccl_metadata::P2p as _;
use crate::profiler;
use crate::profiler::Communicator;
use crate::profiler::Profiler;
use crate::shm_fifo;
use crate::slab;
use crate::spsc;
use crate::step_tracker::EventStep;

use log::error;

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub trait Daemon {
    fn new(profiler: &'static Profiler) -> Self;
}

const RETRY_MS: u64 = 100;

pub trait Export {
    fn export(&self, ctx: &mut PollingContext, maybe_retry_ms: Option<u64>);
    fn get_latency_histogram(
        &self,
        _key: &nccl_metadata::NcclOpKey,
    ) -> Option<impl std::iter::IntoIterator<Item = Arc<dyn AtomicHistogram<EventStep>>>> {
        None::<Vec<_>>
    }
}

pub trait AtomicHistogram<T>: std::fmt::Debug + Sync + Send {
    fn record(&self, data: &T);
}

#[derive(Debug)]
pub struct FifoReceiver<T> {
    last_fetch: Instant,
    receiver: spsc::Receiver<T>,
    pending_msg: VecDeque<T>,
}

impl<T> FifoReceiver<T> {
    pub fn new(receiver: spsc::Receiver<T>) -> Self {
        Self {
            last_fetch: Instant::now(),
            receiver,
            pending_msg: VecDeque::new(),
        }
    }

    pub fn fetch_from_sender(&mut self) {
        self.receiver.try_fetch_from_sender();
    }

    pub fn recv_many(
        &mut self,
        max_idle: std::time::Duration,
        max_recv: usize,
        force_fetch: bool,
    ) -> usize {
        let mut fetched = false;

        let now = Instant::now();
        if force_fetch || now - self.last_fetch > max_idle {
            self.fetch_from_sender();
            fetched = true;
        }
        let mut n_recv = 0;
        while let Some(msg) = self.receiver.recv() {
            fetched = true;
            self.pending_msg.push_back(msg);
            n_recv += 1;
            if n_recv > max_recv {
                break;
            }
        }

        if fetched {
            self.last_fetch = Instant::now();
        }
        n_recv
    }

    pub fn process_many<H>(&mut self, max_process: usize, mut handler: H)
    where
        H: FnMut(T),
    {
        let mut n_processed = 0;
        while let Some(msg) = self.pending_msg.pop_front() {
            handler(msg);
            n_processed += 1;
            if n_processed >= max_process {
                break;
            }
        }
    }
}

#[derive(Debug)]
pub enum ControlMessage {
    NewThread(ThreadControl),
    /// Apply runtime gate update (also used by tests / future IPC).
    SetGates(crate::runtime_gates::GateUpdate),
}

#[derive(Debug)]
pub enum Message {
    Group(Box<event::Group>),
    NcclOp(slab::AllocatedNode<event::NcclOp>),
    ProxyOpLite(
        /* start time */ Instant,
        /* duration ns */ u64,
        event::ProxyOpInfo,
    ),
    ProxyOpExtra(event::ProxyOpExtra),
    ProxyOp(/* id = */ u32),
    StepBatch(
        event::ProxyOpInfo,
        slab::AllocatedNode<StepBatch>,
        /* is_last =*/ bool,
    ),
    KernelCh(
        /* start time */ Instant,
        /* duration ns */ u64,
        /* parent handle */ usize,
    ),
    KernelStep(event::KernelEventStep, /* parent handle */ usize),
    StepProgress {
        source: event::StepProgressSource,
        stage: event::StepProgressStage,
        observed_at: Instant,
        parent: usize,
        is_send: bool,
        peer: u32,
        channel_id: u8,
        step: i64,
        size: usize,
    },
    CommOpen(Communicator),
    CommInit(crate::profiler::CommMembership),
    CommClose(/* comm_hash = */ u64),
}

#[derive(Debug)]
pub enum InterProcessMessage {
    ProxyOpStart(
        /* handle: */ usize,
        /* pid: */ libc::pid_t,
        /* thread idx: */ usize,
        /* id: */ u32,
        Instant,
    ),
    ProxyOpEnd(/* handle: */ usize, Instant),
    ReplyProxyOpComm(/* thread idx */ usize, /* proxy id = */ u32, u64),
}

pub const STEP_BATCH_SZ: usize = 64;
pub type StepBatch = fixed_batch::Batch<EventStep, STEP_BATCH_SZ>;

type PendingProgress = (
    event::StepProgressSource,
    event::StepProgressStage,
    Instant,
    bool,
    u32,
    u8,
    i64,
    usize,
);

/// This separate Telemetry type should make a copy of the Message type.
/// Doing so decouples:
///   1. event management of those shared profiler API handler,
///      which has higher performance requirements
///   2. resource used by the telemetry exporter, which may be blocking for I/O
pub enum Telemetry {
    Group(Box<event::Group>),
    NcclOpIssued(Box<event::NcclOp>), // copybara:strip(hang detection)
    /// Compact GPU completion, independent of delayed child attachment.
    NcclOpComplete(Box<event::NcclOp>),
    NcclOp(Box<event::NcclOp>),
    P2pParent(Box<event::P2pParent>),
    StepProgress(event::StepProgress),
    ProxyOp(Box<event::ProxyOp>),
    CommOpen(Communicator),
    CommInit(crate::profiler::CommMembership),
    CommClose(/* comm_hash = */ u64),
}

/// Accumulator for P2Ps that share one NCCL Group. Native COLLs in the same
/// group are emitted on their own; P2Ps wait here and nest under P2P_GROUP.
#[derive(Default)]
struct P2pGroupAcc {
    expected: usize,
    ended: bool,
    children: Vec<Box<event::NcclOp>>,
    self_copy_size: usize,
}

pub struct PollingContext<'a> {
    profiler: &'a Profiler,
    pub ncclops: BTreeMap<usize, Box<event::NcclOp>>,
    pub pending_telemetry: VecDeque<Telemetry>,
    stop: Arc<AtomicBool>,

    free_ncclop: slab::FreeList<event::NcclOp>,
    free_proxyop: slab::FreeList<event::ProxyOp>,
    free_step_batch: slab::FreeList<StepBatch>,

    peer_rank_fifo: HashMap<libc::pid_t, shm_fifo::mpsc::Sender<InterProcessMessage>>,
    pending_ipc_msg: HashMap<libc::pid_t, VecDeque<InterProcessMessage>>,
    p2p_groups: HashMap<u64, P2pGroupAcc>,
    /// One compact, live progress marker per NCCL parent/source/peer.  Raw
    /// KernelStep callbacks can number in the thousands for one collective;
    /// exporting all of them delays the very deadline decision they serve.
    active_progress: HashMap<(usize, u8, u32), event::StepProgress>,
    /// KernelCh and its parent operation can be delivered by different NCCL
    /// threads. Preserve a completion that wins that race until the launch
    /// record arrives instead of losing the end-of-collective signal.
    early_completions: HashMap<usize, (Instant, Instant)>,
    /// Progress callbacks can race ahead of their parent launch because NCCL
    /// invokes profiler callbacks from several threads. Preserve the compact
    /// transition and replay it as soon as the parent is registered.
    early_progress: HashMap<usize, Vec<PendingProgress>>,
    online_completed: HashSet<usize>,
}

impl<'a> PollingContext<'a> {
    pub fn new(profiler: &'a Profiler, stop_signal: Arc<AtomicBool>) -> Self {
        Self {
            profiler,
            ncclops: BTreeMap::new(),
            pending_telemetry: VecDeque::new(),
            stop: stop_signal,
            free_ncclop: slab::FreeList::default(),
            free_proxyop: slab::FreeList::default(),
            free_step_batch: slab::FreeList::default(),
            peer_rank_fifo: HashMap::new(),
            pending_ipc_msg: HashMap::new(),
            p2p_groups: HashMap::new(),
            active_progress: HashMap::new(),
            early_completions: HashMap::new(),
            early_progress: HashMap::new(),
            online_completed: HashSet::new(),
        }
    }

    fn progress_source_id(source: event::StepProgressSource) -> u8 {
        match source {
            event::StepProgressSource::Proxy => 0,
            event::StepProgressSource::Kernel => 1,
            event::StepProgressSource::KernelCh => 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn note_step_progress(
        &mut self,
        source: event::StepProgressSource,
        stage: event::StepProgressStage,
        observed_at: Instant,
        parent: usize,
        is_send: bool,
        peer: u32,
        channel_id: u8,
        step: i64,
        size: usize,
    ) {
        // A cross-thread posted callback can arrive after KernelCh. Never
        // resurrect progress for a collective already closed.
        if self.online_completed.contains(&parent) {
            return;
        }
        // An individual transport-step completion is not the parent end.
        // Parent completion closes every representative endpoint together.
        if stage == event::StepProgressStage::Complete {
            return;
        }
        let key = (parent, Self::progress_source_id(source), peer);
        if self.active_progress.contains_key(&key) {
            return;
        }
        let Some(ncclop) = self.get_ncclop(parent) else {
            let pending = self.early_progress.entry(parent).or_default();
            if !pending
                .iter()
                .any(|(s, _, _, _, p, _, _, _)| *s == source && *p == peer)
            {
                pending.push((
                    source,
                    stage,
                    observed_at,
                    is_send,
                    peer,
                    channel_id,
                    step,
                    size,
                ));
            }
            return;
        };
        let progress = event::StepProgress::from_parent(
            ncclop,
            source,
            stage,
            observed_at,
            is_send,
            peer,
            channel_id,
            step,
            size,
        );
        self.active_progress.insert(key, progress.clone());
        self.pending_telemetry
            .push_back(Telemetry::StepProgress(progress));
    }

    fn finish_parent_progress(&mut self, parent: usize, observed_at: Instant) {
        let keys: Vec<_> = self
            .active_progress
            .keys()
            .filter(|(p, _, _)| *p == parent)
            .copied()
            .collect();
        for key in keys {
            if let Some(mut progress) = self.active_progress.remove(&key) {
                progress.stage = event::StepProgressStage::Complete;
                progress.observed_at = observed_at;
                self.pending_telemetry
                    .push_back(Telemetry::StepProgress(progress));
            }
        }
    }

    fn complete_ncclop(&mut self, parent: usize, child_start: Instant, end_time: Instant) {
        if self.online_completed.contains(&parent) {
            return;
        }

        let Some(ncclop) = self.get_ncclop(parent) else {
            self.early_completions
                .entry(parent)
                .and_modify(|(start, end)| {
                    *start = (*start).min(child_start);
                    *end = (*end).max(end_time);
                })
                .or_insert((child_start, end_time));
            return;
        };
        ncclop_update(ncclop, child_start, Some(end_time));
        let completed = ncclop.clone();

        self.online_completed.insert(parent);
        self.finish_parent_progress(parent, end_time);
        self.pending_telemetry
            .push_back(Telemetry::NcclOpComplete(Box::new(completed)));
    }

    fn note_ncclop_start(&mut self, op: &event::NcclOp) {
        if !op.is_p2p() {
            return;
        }
        let Some(gid) = op.parent_group_id() else {
            return;
        };
        self.p2p_groups.entry(gid).or_default().expected += 1;
    }

    fn group_has_inflight(&self, gid: u64) -> bool {
        self.ncclops
            .values()
            .any(|op| op.is_p2p() && op.parent_group_id() == Some(gid))
    }

    fn try_emit_p2p_parent(&mut self, gid: u64, force: bool) {
        let (n_children, ended, expected) = match self.p2p_groups.get(&gid) {
            Some(g) => (g.children.len(), g.ended, g.expected),
            None => return,
        };
        let ready = n_children > 0
            && (force || (ended && n_children >= expected && !self.group_has_inflight(gid)));
        if !ready {
            return;
        }
        let g = self.p2p_groups.remove(&gid).unwrap();
        let partial = force && (g.children.len() < g.expected || !g.ended);
        self.emit_p2p_group(gid, g.children, partial, g.self_copy_size);
    }

    fn emit_p2p_group(
        &mut self,
        gid: u64,
        children: Vec<Box<event::NcclOp>>,
        partial: bool,
        self_copy_size: usize,
    ) {
        let mut children: Vec<event::NcclOp> = children.into_iter().map(|b| *b).collect();
        children.sort_by_key(|c| (c.basic_info().start_time(), c.id()));
        for c in &mut children {
            c.clear_parent_group_id();
        }
        let start = children
            .iter()
            .map(|c| c.basic_info().start_time())
            .min()
            .unwrap_or_else(Instant::now);
        let end = children
            .iter()
            .filter_map(|c| c.basic_info().end_time())
            .max()
            .unwrap_or(start);
        let mut peers = HashSet::new();
        let mut n_send = 0usize;
        let mut n_recv = 0usize;
        let mut send_bytes = 0usize;
        for c in &children {
            if let Some(p2p) = c.get_descr().try_cast_to_p2p() {
                peers.insert(p2p.peer());
                if p2p.is_send() {
                    n_send += 1;
                    send_bytes = send_bytes.saturating_add(c.byte_count());
                } else {
                    n_recv += 1;
                }
            }
        }
        let name = if peers.len() >= 2 {
            "all_to_all"
        } else if n_send > 0 && n_recv > 0 {
            "sendrecv"
        } else if n_send > 0 {
            "send"
        } else {
            "recv"
        };
        let rank = children.first().map(|c| c.basic_info().rank()).unwrap_or(0);
        let comm_hash = children.first().map(|c| c.comm_hash()).unwrap_or(0);
        self.pending_telemetry
            .push_back(Telemetry::P2pParent(Box::new(event::P2pParent {
                group_id: gid,
                rank,
                comm_hash,
                start_time: start,
                end_time: if end > start { end } else { start },
                size: send_bytes,
                n_children: children.len(),
                n_peers: peers.len(),
                name,
                children,
                partial,
                self_copy_size,
            })));
    }

    fn flush_p2p_parents(&mut self) {
        let gids: Vec<u64> = self.p2p_groups.keys().copied().collect();
        for gid in gids {
            self.try_emit_p2p_parent(gid, true);
        }
    }

    fn reclaim_ncclop(&mut self, mut op: event::NcclOp) {
        self.finish_parent_progress(
            op.id(),
            op.basic_info().end_time().unwrap_or_else(Instant::now),
        );
        if !op.is_p2p() {
            self.pending_telemetry
                .push_back(Telemetry::NcclOp(Box::new(op)));
            return;
        }
        match op.parent_group_id() {
            Some(gid) => {
                self.p2p_groups
                    .entry(gid)
                    .or_default()
                    .children
                    .push(Box::new(op));
                self.try_emit_p2p_parent(gid, false);
            }
            None => {
                op.clear_parent_group_id();
                self.emit_p2p_group(0, vec![Box::new(op)], false, 0);
            }
        }
    }

    fn get_ncclop(&mut self, id: usize) -> Option<&mut event::NcclOp> {
        self.ncclops.get_mut(&id).map(|b| b.as_mut())
    }

    fn get_ipc_fifo(
        &mut self,
        pid: libc::pid_t,
    ) -> Option<&mut shm_fifo::mpsc::Sender<InterProcessMessage>> {
        if self.profiler.config.track_interprocess_proxyop {
            use std::collections::hash_map::Entry;
            match self.peer_rank_fifo.entry(pid) {
                Entry::Vacant(e) => {
                    let tx = shm_fifo::mpsc::Sender::new(&ipc_shm_path(pid));
                    if tx.is_err() {
                        return None;
                    }
                    Some(e.insert(tx.unwrap()))
                }
                Entry::Occupied(e) => Some(e.into_mut()),
            }
        } else {
            None
        }
    }

    fn try_commit_ipc(&mut self) {
        if self.profiler.config.track_interprocess_proxyop {
            for (_, fifo) in self.peer_rank_fifo.iter_mut() {
                let _ = fifo.try_commit();
            }
        }
    }

    fn append_ipc_message(&mut self, pid: libc::pid_t, msg: InterProcessMessage) {
        self.pending_ipc_msg.entry(pid).or_default().push_back(msg);
    }

    // copybara:strip_begin(gpuviz)
    fn try_add_histogram<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        ncclop_id: usize,
        proxyop_id: u32,
    ) -> Option<()>
    where
        E: Export,
    {
        let ncclop = self.get_ncclop(ncclop_id)?;
        let op_key = ncclop.op_key();
        let comm_hash = op_key.get_comm_hash();
        let proxyop = thread_state.proxyops.get_mut(&proxyop_id)?;
        if proxyop.info().rank != op_key.local_rank() {
            static LOG_TIMER: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
            if let Ok(mut lg) = LOG_TIMER.lock() {
                let should_log = if let Some(t) = *lg {
                    t.elapsed().as_secs() >= 60
                } else {
                    true
                };

                if should_log {
                    log::error!(
                        "Found rank mismatch between proxyop and its parent: {} vs {}",
                        proxyop.info().rank,
                        op_key.local_rank()
                    );
                    *lg = Some(Instant::now());
                }
            }
        }
        self.try_add_histogram_from_comm_hash(thread_state, exporter, proxyop_id, comm_hash)
    }

    fn try_add_histogram_from_comm_hash<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        proxyop_id: u32,
        comm_hash: u64,
    ) -> Option<()>
    where
        E: Export,
    {
        let proxyop = if let Some(p) = thread_state.proxyops.get_mut(&proxyop_id) {
            p
        } else {
            &mut thread_state
                .pending_comm_hash_proxyops
                .get_mut(&proxyop_id)?
                .0
        };
        if proxyop.has_step_histograms() {
            return None;
        }
        let conn_key = proxyop.info().op_key(comm_hash);
        if let Some(h_list) = exporter.get_latency_histogram(&conn_key) {
            proxyop.set_step_histograms(h_list.into_iter().map(|h| h as _).collect());
        }
        Some(())
    }
    // copybara:strip_end

    fn should_wait_for_comm_hash(&self, proxyop: &event::ProxyOp) -> bool {
        self.profiler.config.track_interprocess_proxyop
            && proxyop.aggregate_steps()
            && proxyop.info().pid != self.profiler.pid
            && proxyop.info().parent().is_some()
            && !proxyop.has_step_histograms()
    }

    fn add_proxyop<E>(
        &mut self,
        thread_state: &mut ThreadState,
        exporter: &mut E,
        info: &event::ProxyOpInfo,
        start_time: Instant,
    ) where
        E: Export,
    {
        let id = info.id;
        let mut op = Box::new(event::ProxyOp::from_info(info, start_time));
        let mut aggregate_steps = false;
        // only aggregate steps when:
        // 0. aggregate_steps flag is set
        // 1. we know the parent (and therefore comm hash)
        // 2. this proxyop is originated from current process OR
        //    we are tracking interprocess proxyop
        // Prefer runtime gates (comma-monitor mid-flight enable) over the
        // process-start Config snapshot.
        let gates = &self.profiler.gates;
        let track_steps = gates.track_steps();
        let aggregate_cfg = gates.aggregate_steps();
        let track_interprocess = gates.track_interprocess_proxyop();
        if info.parent().is_some() {
            aggregate_steps =
                aggregate_cfg && (info.pid == self.profiler.pid || track_interprocess);
        }
        op.init_step_tracking(track_steps, aggregate_steps);
        self.handle_proxyop_start(thread_state, info, op.basic_info().start_time());

        thread_state.proxyops.insert(id, op);
        // copybara:strip_begin(gpuviz)
        if let Some(parent) = info.parent() {
            if aggregate_steps && info.pid == self.profiler.pid {
                self.try_add_histogram(thread_state, exporter, parent, id);
            }
        }
        // copybara:strip_end
    }

    fn handle_proxyop_start(
        &mut self,
        thread_state: &mut ThreadState,
        info: &event::ProxyOpInfo,
        time: Instant,
    ) {
        if let Some(parent) = info.parent() {
            let pid = info.pid;
            if pid == self.profiler.pid {
                if let Some(ncclop) = self.get_ncclop(parent) {
                    ncclop_update(ncclop, time, None);
                }
            } else {
                let msg = InterProcessMessage::ProxyOpStart(
                    parent,
                    self.profiler.pid,
                    thread_state.idx,
                    info.id,
                    time,
                );
                self.append_ipc_message(pid, msg);
            }
        }
    }

    #[allow(clippy::vec_box)]
    fn handle_proxyop_end(
        &mut self,
        end_time: Instant,
        info: &event::ProxyOpInfo,
        proxyops: Vec<Box<event::ProxyOp>>,
        send_ipc: bool,
    ) {
        // Runtime gates: when the monitor escalates after an anomaly, newly
        // completed ProxyOps must be attached to the parent COLL JSON even
        // though process-start Config still has track_proxyop/steps=false.
        let gates = &self.profiler.gates;
        let record_proxyop = gates.track_proxyop() || gates.track_steps();
        if let Some(parent_handle) = info.parent() {
            if info.pid == self.profiler.pid {
                if let Some(ncclop) = self.get_ncclop(parent_handle) {
                    ncclop_update(ncclop, end_time, Some(end_time));
                    if record_proxyop {
                        for p in proxyops {
                            ncclop.add_proxyop(*p);
                        }
                    }
                }
                return;
            } else if send_ipc {
                let msg = InterProcessMessage::ProxyOpEnd(parent_handle, end_time);
                self.append_ipc_message(info.pid, msg);
            }
        }
        if record_proxyop {
            for p in proxyops {
                self.pending_telemetry.push_back(Telemetry::ProxyOp(p));
            }
        }
    }

    fn finalize_pending_comm_hash_proxyop(&mut self, proxyop: Box<event::ProxyOp>) {
        let info = proxyop.info().clone();
        let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);
        self.handle_proxyop_end(end_time, &info, vec![proxyop], false);
    }

    /// Construct a Telemetry type from Message.
    /// Resource used by `msg` should be release ASAP so the profiler API handlers won't be blocked on
    /// them
    fn handle_fifo_message<E>(
        &mut self,
        msg: Message,
        thread_state: &mut ThreadState,
        exporter: &mut E,
    ) where
        E: Export,
    {
        match msg {
            Message::Group(group) => {
                let gid = group.id();
                let acc = self.p2p_groups.entry(gid).or_default();
                acc.ended = true;
                acc.self_copy_size = acc.self_copy_size.max(group.self_copy_size());
                self.try_emit_p2p_parent(gid, false);
                if self.profiler.config.track_group {
                    self.pending_telemetry.push_back(Telemetry::Group(group));
                }
            }
            Message::NcclOp(op) => {
                let id = op.id();
                let op = self.free_ncclop.take_and_free(op);
                self.note_ncclop_start(&op);
                let _ = self.ncclops.insert(id, Box::new(op.clone()));
                // copybara:strip_begin(hang detection)
                self.pending_telemetry
                    .push_back(Telemetry::NcclOpIssued(Box::new(op)));
                // copybara:strip_end
                if let Some(progress) = self.early_progress.remove(&id) {
                    for (source, stage, observed_at, is_send, peer, channel, step, size) in progress
                    {
                        self.note_step_progress(
                            source,
                            stage,
                            observed_at,
                            id,
                            is_send,
                            peer,
                            channel,
                            step,
                            size,
                        );
                    }
                }
                if let Some((child_start, end_time)) = self.early_completions.remove(&id) {
                    self.complete_ncclop(id, child_start, end_time);
                }
                if self.free_ncclop.num_free() >= slab::FREELIST_BATCH {
                    self.free_ncclop.try_publish(&self.profiler.free_ncclop);
                }
            }
            Message::ProxyOpLite(start_time, dur_ns, info) => {
                self.handle_proxyop_start(thread_state, &info, start_time);
                let end_time = start_time + Duration::from_nanos(dur_ns);
                self.handle_proxyop_end(end_time, &info, Vec::new(), true);
            }
            Message::ProxyOpExtra(extra) => {
                thread_state.aux_msg.push(Message::ProxyOpExtra(extra));
            }
            Message::ProxyOp(id) => {
                let proxyop = thread_state.proxyops.remove(&id);
                let extra = if let Some(Message::ProxyOpExtra(_)) = thread_state.aux_msg.last() {
                    let msg = thread_state.aux_msg.pop().unwrap();
                    if let Message::ProxyOpExtra(extra) = msg {
                        Some(extra)
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(mut proxyop) = proxyop {
                    if let Some(extra) = extra {
                        proxyop.add_extra_info(extra);
                    }
                    let info = proxyop.info().clone();
                    let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);

                    let mut ops = Vec::new();
                    if self.should_wait_for_comm_hash(&proxyop) {
                        thread_state
                            .pending_comm_hash_proxyops
                            .insert(id, (proxyop, Instant::now()));
                    } else {
                        ops.push(proxyop);
                    }
                    self.handle_proxyop_end(end_time, &info, ops, true);
                } else {
                    // error!("proxyop {} not found!", id);
                }
            }
            Message::StepBatch(info, mut steps, is_last) => {
                if !thread_state.proxyops.contains_key(&info.id) {
                    self.add_proxyop(
                        thread_state,
                        exporter,
                        &info,
                        self.profiler.init_instant + Duration::from_nanos(steps[0].start_time),
                    );
                }
                let proxyop = thread_state.proxyops.get_mut(&info.id).unwrap();
                let init_instant = self.profiler.init_instant;
                for step in steps.drain() {
                    let start_time = init_instant + Duration::from_nanos(step.start_time as _);
                    let end_time = init_instant + Duration::from_nanos(step.end_time_ns() as _);
                    proxyop.try_set_start_time(start_time);
                    proxyop.set_end_time(end_time);
                    proxyop.add_step(step);
                }
                if is_last {
                    let mut proxyop = thread_state.proxyops.remove(&info.id).unwrap();
                    let extra = if let Some(Message::ProxyOpExtra(_)) = thread_state.aux_msg.last()
                    {
                        let msg = thread_state.aux_msg.pop().unwrap();
                        if let Message::ProxyOpExtra(extra) = msg {
                            Some(extra)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    if let Some(extra) = extra {
                        proxyop.add_extra_info(extra);
                    }
                    let end_time = proxyop.basic_info().end_time().unwrap_or_else(Instant::now);

                    let mut ops = Vec::new();
                    if self.should_wait_for_comm_hash(&proxyop) {
                        thread_state
                            .pending_comm_hash_proxyops
                            .insert(info.id, (proxyop, Instant::now()));
                    } else {
                        ops.push(proxyop);
                    }
                    self.handle_proxyop_end(end_time, &info, ops, true);
                }
                self.free_step_batch.free(steps);
                if self.free_step_batch.num_free() >= slab::FREELIST_BATCH {
                    self.free_step_batch
                        .try_publish(&self.profiler.free_step_batch);
                }
            }
            Message::KernelCh(start_time, duration, parent) => {
                let end_time = start_time + Duration::from_nanos(duration);
                self.complete_ncclop(parent, start_time, end_time);
            }
            Message::KernelStep(step, parent) => {
                if let Some(ncclop) = self.get_ncclop(parent) {
                    ncclop.add_kernel_step(step);
                }
            }
            Message::StepProgress {
                source,
                stage,
                observed_at,
                parent,
                is_send,
                peer,
                channel_id,
                step,
                size,
            } => {
                self.note_step_progress(
                    source,
                    stage,
                    observed_at,
                    parent,
                    is_send,
                    peer,
                    channel_id,
                    step,
                    size,
                );
            }
            Message::CommOpen(comm) => {
                self.pending_telemetry.push_back(Telemetry::CommOpen(comm));
            }
            Message::CommInit(membership) => {
                self.pending_telemetry
                    .push_back(Telemetry::CommInit(membership));
            }
            Message::CommClose(comm_hash) => {
                self.pending_telemetry
                    .push_back(Telemetry::CommClose(comm_hash));
            }
        }
    }

    fn handle_ipc_message<E>(
        &mut self,
        /* copybara:strip_begin(gpuviz) */
        threads: &mut [ThreadControl],
        exporter: &mut E,
        /* copybara:strip_end_and_replace
        _threads: &mut [ThreadControl],
        _exporter: &mut E,
         */
        msg: InterProcessMessage,
    ) where
        E: Export,
    {
        match msg {
            InterProcessMessage::ProxyOpStart(handle, pid, thread_idx, id, time) => {
                if let Some(ncclop) = self.get_ncclop(handle) {
                    ncclop_update(ncclop, time, None);
                    let comm = ncclop.op_key().get_comm_hash();
                    let reply = InterProcessMessage::ReplyProxyOpComm(thread_idx, id, comm);
                    self.append_ipc_message(pid, reply);
                }
            }
            InterProcessMessage::ProxyOpEnd(handle, time) => {
                if let Some(ncclop) = self.get_ncclop(handle) {
                    ncclop_update(ncclop, time, Some(time));
                }
            }
            /* copybara:strip_begin(gpuviz) */
            InterProcessMessage::ReplyProxyOpComm(thread_idx, id, comm) => {
                let t = &mut threads[thread_idx];
                if self.profiler.config.aggregate_steps {
                    let _ = self.try_add_histogram_from_comm_hash(
                        &mut t.daemon_state,
                        exporter,
                        id,
                        comm,
                    );
                    // If the proxyop has already ended and was waiting for this comm_hash,
                    // finalize it now.
                    if let Some((proxyop, _)) =
                        t.daemon_state.pending_comm_hash_proxyops.remove(&id)
                    {
                        self.finalize_pending_comm_hash_proxyop(proxyop);
                    }
                }
            } /* copybara:strip_end_and_replace
              InterProcessMessage::ReplyProxyOpComm(_, _, _) => {}
              */
        }
    }

    fn try_reclaim_ncclops_from_map<P>(&mut self, mut should_reclaim: P)
    where
        P: FnMut(&Box<event::NcclOp>) -> ReclaimAction,
    {
        let mut to_keep = Vec::new();
        while let Some((k, v)) = self.ncclops.pop_first() {
            match should_reclaim(&v) {
                ReclaimAction::Keep => to_keep.push((k, v)),
                ReclaimAction::Reclaim => self.reclaim_ncclop(*v),
                ReclaimAction::Stop => {
                    self.ncclops.insert(k, v);
                    break;
                }
            }
        }
        for (k, v) in to_keep {
            self.ncclops.insert(k, v);
        }
    }

    pub fn reclaim_all_ncclops_in_map(&mut self) {
        self.try_reclaim_ncclops_from_map(|_| ReclaimAction::Reclaim);
    }
}

/// control data used by each thread.
///
/// This struct is used by the daemon thread to communicate with each thread.
#[derive(Debug)]
pub struct ThreadControl {
    pub ncclop_fifo: FifoReceiver<Message>,
    pub fifo: FifoReceiver<Message>,
    daemon_state: ThreadState,
}

#[derive(Debug, Default)]
struct ThreadState {
    idx: usize,
    proxyops: HashMap<u32, Box<event::ProxyOp>>,
    pending_comm_hash_proxyops: HashMap<u32, (Box<event::ProxyOp>, Instant)>,
    aux_msg: Vec<Message>,
}

impl ThreadControl {
    pub fn new(ncclop_fifo: FifoReceiver<Message>, fifo: FifoReceiver<Message>) -> Self {
        Self {
            ncclop_fifo,
            fifo,
            daemon_state: ThreadState::default(),
        }
    }
}

fn ncclop_update(op: &mut event::NcclOp, st: Instant, et: Option<Instant>) {
    op.update_child_start_time(st);
    if let Some(t) = et {
        op.basic_info_mut().update_end_time(t);
    }
}

fn should_reclaim_ncclop(op: &event::NcclOp, comp_delay: Duration, timeout: Duration) -> bool {
    let end_time = op.basic_info().end_time();
    if let Some(et) = end_time {
        let now = Instant::now();
        if now - et > comp_delay {
            return true;
        }
    }
    let start_time = op.basic_info().start_time();
    if start_time.elapsed() > timeout {
        return true;
    }
    false
}

#[derive(Debug)]
enum ReclaimAction {
    Keep,
    Reclaim,
    Stop,
}

fn ipc_shm_path(pid: libc::pid_t) -> String {
    format!("nccl-profiler-{}", pid)
}

const FIFO_FETCH_INTERVAL: Duration = Duration::from_secs(1);
const FIFO_PROCESS_BATCH: usize = 512;
const ONLINE_FIFO_PROCESS_BATCH: usize = profiler::EVENT_QUEUE_SZ;
const FIFO_RECV_BATCH: usize = profiler::EVENT_QUEUE_SZ;
const IPC_RECV_BATCH: usize = 512;
const IPC_SEND_BATCH: usize = 64;
const NCCLOP_RECLAIM_BATCH: usize = 512;

pub fn polling_loop<E>(ctx: &mut PollingContext, mut exporter: E)
where
    E: Export,
{
    let mut threads: Vec<ThreadControl> = Vec::new();

    let mut ipc_fifo_rx: Option<shm_fifo::mpsc::Receiver<InterProcessMessage>> =
        if ctx.profiler.config.track_interprocess_proxyop {
            let path = ipc_shm_path(ctx.profiler.pid);
            let rx = shm_fifo::mpsc::Receiver::new(&path, 1024);
            if rx.is_err() {
                error!("failed to create shm fifo {}", path);
            }
            rx.ok()
        } else {
            None
        };

    let ncclop_timeout = ctx.profiler.config.ncclop_timeout;
    let ncclop_comp_delay = ctx.profiler.config.ncclop_completion_delay;

    while !ctx.stop.load(Ordering::Acquire) {
        while let Some(ctrl_msg) = ctx.profiler.ctrl_fifo.pop() {
            match ctrl_msg {
                ControlMessage::NewThread(mut ctrl) => {
                    ctrl.daemon_state.idx = threads.len();
                    threads.push(ctrl);
                }
                ControlMessage::SetGates(update) => {
                    let _ = ctx.profiler.gates.apply_update(&update);
                }
            }
        }

        let mut n_recv = 0;
        for thread in threads.iter_mut() {
            n_recv += thread
                .fifo
                .recv_many(FIFO_FETCH_INTERVAL, FIFO_RECV_BATCH, false);
        }

        let mut ipc_msg = Vec::new();
        if let Some(rx) = ipc_fifo_rx.as_mut() {
            while let Some(msg) = rx.recv() {
                ipc_msg.push(msg);
                if ipc_msg.len() > IPC_RECV_BATCH {
                    break;
                }
            }
        }

        n_recv += ipc_msg.len();

        for thread in threads.iter_mut() {
            thread
                .ncclop_fifo
                .recv_many(FIFO_FETCH_INTERVAL, FIFO_RECV_BATCH, n_recv > 0);
            thread
                .ncclop_fifo
                .process_many(ONLINE_FIFO_PROCESS_BATCH, |msg| {
                    ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
                });
        }

        for thread in threads.iter_mut() {
            thread.fifo.process_many(FIFO_PROCESS_BATCH, |msg| {
                ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
            });
        }

        for msg in ipc_msg {
            ctx.handle_ipc_message(&mut threads, &mut exporter, msg);
        }

        let mut pending_ipc_msg = std::mem::take(&mut ctx.pending_ipc_msg);
        pending_ipc_msg.retain(|pid, msgs| {
            if let Some(tx) = ctx.get_ipc_fifo(*pid) {
                let mut n_sent = 0;
                while let Some(msg) = msgs.pop_front() {
                    if let Err(m) = tx.send(msg) {
                        msgs.push_front(m);
                        break;
                    }
                    n_sent += 1;
                    if n_sent >= IPC_SEND_BATCH {
                        break;
                    }
                }
                return !msgs.is_empty();
            }
            true
        });

        ctx.pending_ipc_msg = pending_ipc_msg;

        ctx.try_commit_ipc();

        let comm_hash_ipc_timeout = ctx.profiler.config.comm_hash_ipc_timeout;
        for thread in threads.iter_mut() {
            thread.daemon_state.pending_comm_hash_proxyops =
                std::mem::take(&mut thread.daemon_state.pending_comm_hash_proxyops)
                    .into_iter()
                    .flat_map(|(id, (proxyop, start_time))| {
                        if start_time.elapsed() > comm_hash_ipc_timeout {
                            ctx.finalize_pending_comm_hash_proxyop(proxyop);
                            None
                        } else {
                            Some((id, (proxyop, start_time)))
                        }
                    })
                    .collect();
        }

        if ctx.free_proxyop.num_free() >= slab::FREELIST_BATCH {
            ctx.free_proxyop.try_publish(&ctx.profiler.free_proxyop);
        }

        let mut n_processed = 0;
        ctx.try_reclaim_ncclops_from_map(|op| {
            if n_processed > NCCLOP_RECLAIM_BATCH {
                return ReclaimAction::Stop;
            }
            n_processed += 1;
            if should_reclaim_ncclop(op, ncclop_comp_delay, ncclop_timeout) {
                ReclaimAction::Reclaim
            } else {
                ReclaimAction::Keep
            }
        });

        if ctx.ncclops.len() > ctx.profiler.config.max_tracked_ncclop {
            // force reclaim
            let mut num_to_reclaim = ctx.ncclops.len() - ctx.profiler.config.max_tracked_ncclop;
            ctx.try_reclaim_ncclops_from_map(|_| {
                if num_to_reclaim > 0 {
                    num_to_reclaim -= 1;
                    ReclaimAction::Reclaim
                } else {
                    ReclaimAction::Stop
                }
            });
        }

        exporter.export(ctx, None)
    }

    for thread in threads.iter_mut() {
        thread
            .ncclop_fifo
            .recv_many(FIFO_FETCH_INTERVAL, usize::MAX, true);
        thread.ncclop_fifo.process_many(usize::MAX, |msg| {
            ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
        });
    }

    for thread in threads.iter_mut() {
        thread.fifo.recv_many(FIFO_FETCH_INTERVAL, usize::MAX, true);
        thread.fifo.process_many(usize::MAX, |msg| {
            ctx.handle_fifo_message(msg, &mut thread.daemon_state, &mut exporter);
        });
    }

    for thread in threads.iter_mut() {
        for (_, (proxyop, _)) in thread.daemon_state.pending_comm_hash_proxyops.drain() {
            ctx.finalize_pending_comm_hash_proxyop(proxyop);
        }
    }

    ctx.reclaim_all_ncclops_in_map();
    ctx.flush_p2p_parents();

    // Shutdown may reclaim more records than the bounded Tokio channels can
    // accept in one pass. Keep draining until every final trace has actually
    // crossed into the exporter; otherwise categories disappear depending on
    // queue timing.
    while !ctx.pending_telemetry.is_empty() {
        exporter.export(ctx, Some(RETRY_MS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiler::Version;
    use crate::profiler_shim;

    struct NoopExport;

    impl Export for NoopExport {
        fn export(&self, _ctx: &mut PollingContext, _maybe_retry_ms: Option<u64>) {}
    }

    fn coll(id: usize, start: Instant) -> slab::AllocatedNode<event::NcclOp> {
        let descr = profiler_shim::tests::dummy_coll_descr();
        slab::AllocatedNode::new(event::NcclOp::from_descr(&descr, start, id, None))
    }

    #[test]
    fn completion_before_parent_is_preserved_and_late_progress_is_ignored() {
        let profiler = Profiler::new(Version::V1);
        let mut ctx = PollingContext::new(&profiler, Arc::new(AtomicBool::new(false)));
        let mut thread = ThreadState::default();
        let mut exporter = NoopExport;
        let parent = 42;
        let child_start = Instant::now();
        let duration = Duration::from_millis(3);

        ctx.handle_fifo_message(
            Message::KernelCh(child_start, duration.as_nanos() as u64, parent),
            &mut thread,
            &mut exporter,
        );
        assert!(ctx.early_completions.contains_key(&parent));
        assert!(!ctx.online_completed.contains(&parent));

        ctx.handle_fifo_message(
            Message::NcclOp(coll(parent, child_start - Duration::from_millis(1))),
            &mut thread,
            &mut exporter,
        );
        assert!(!ctx.early_completions.contains_key(&parent));
        assert!(ctx.online_completed.contains(&parent));
        assert!(matches!(
            ctx.pending_telemetry.pop_front(),
            Some(Telemetry::NcclOpIssued(_))
        ));
        let complete = ctx.pending_telemetry.pop_front();
        match complete {
            Some(Telemetry::NcclOpComplete(op)) => {
                assert_eq!(op.basic_info().end_time(), Some(child_start + duration));
            }
            _ => panic!("expected online completion after launch record"),
        }

        ctx.handle_fifo_message(
            Message::StepProgress {
                source: event::StepProgressSource::Kernel,
                stage: event::StepProgressStage::Posted,
                observed_at: child_start + duration,
                parent,
                is_send: true,
                peer: 1,
                channel_id: 0,
                step: 0,
                size: 4096,
            },
            &mut thread,
            &mut exporter,
        );
        assert!(ctx.active_progress.is_empty());
        assert!(ctx.pending_telemetry.is_empty());
    }

    #[test]
    fn progress_before_parent_is_replayed_before_completion() {
        let profiler = Profiler::new(Version::V1);
        let mut ctx = PollingContext::new(&profiler, Arc::new(AtomicBool::new(false)));
        let mut thread = ThreadState::default();
        let mut exporter = NoopExport;
        let parent = 44;
        let start = Instant::now();

        ctx.handle_fifo_message(
            Message::StepProgress {
                source: event::StepProgressSource::KernelCh,
                stage: event::StepProgressStage::Posted,
                observed_at: start,
                parent,
                is_send: true,
                peer: u32::MAX,
                channel_id: u8::MAX,
                step: -1,
                size: 0,
            },
            &mut thread,
            &mut exporter,
        );
        assert_eq!(ctx.early_progress.get(&parent).map(Vec::len), Some(1));

        ctx.handle_fifo_message(
            Message::NcclOp(coll(parent, start - Duration::from_millis(1))),
            &mut thread,
            &mut exporter,
        );
        assert!(!ctx.early_progress.contains_key(&parent));
        assert_eq!(ctx.active_progress.len(), 1);
        assert!(matches!(
            ctx.pending_telemetry.pop_front(),
            Some(Telemetry::NcclOpIssued(_))
        ));
        assert!(matches!(
            ctx.pending_telemetry.pop_front(),
            Some(Telemetry::StepProgress(event::StepProgress {
                source: event::StepProgressSource::KernelCh,
                stage: event::StepProgressStage::Posted,
                ..
            }))
        ));
    }

    #[test]
    fn completion_closes_inflight_progress_before_emitting_collective_end() {
        let profiler = Profiler::new(Version::V1);
        let mut ctx = PollingContext::new(&profiler, Arc::new(AtomicBool::new(false)));
        let mut thread = ThreadState::default();
        let mut exporter = NoopExport;
        let parent = 43;
        let start = Instant::now();

        ctx.handle_fifo_message(
            Message::NcclOp(coll(parent, start)),
            &mut thread,
            &mut exporter,
        );
        ctx.pending_telemetry.clear();
        ctx.handle_fifo_message(
            Message::StepProgress {
                source: event::StepProgressSource::Proxy,
                stage: event::StepProgressStage::Posted,
                observed_at: start,
                parent,
                is_send: true,
                peer: 2,
                channel_id: 1,
                step: 7,
                size: 8192,
            },
            &mut thread,
            &mut exporter,
        );
        assert_eq!(ctx.active_progress.len(), 1);
        ctx.pending_telemetry.clear();

        ctx.handle_fifo_message(
            Message::KernelCh(start, 1_000, parent),
            &mut thread,
            &mut exporter,
        );
        assert!(ctx.active_progress.is_empty());
        assert!(matches!(
            ctx.pending_telemetry.pop_front(),
            Some(Telemetry::StepProgress(event::StepProgress {
                stage: event::StepProgressStage::Complete,
                ..
            }))
        ));
        assert!(matches!(
            ctx.pending_telemetry.pop_front(),
            Some(Telemetry::NcclOpComplete(_))
        ));
    }
}
