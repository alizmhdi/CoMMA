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

//! File-backed SPSC ring of packed POD records for live latency telemetry.
//!
//! One ring per profiler process (per training rank). CoMMA's export task
//! writes packed slots; the local `comma-monitor` reads and converts them
//! back into `RawEvent` values before feeding the detector. The ring layout
//! is intentionally simple:
//!
//!  * `RingHeader` at file offset `0`, holding the SPSC cursor cells plus
//!    a magic/version so the consumer can validate the map on open.
//!  * `capacity` back-to-back `RingSlot` entries. `capacity` is a power of
//!    two so the mask-based indexing is branchless.
//!
//! Non-goals of this transport (see the online detection contract):
//!  * There is no live Unix stream fallback. The ring is the only transport.
//!  * The producer never blocks on a full ring: it increments `dropped`
//!    and drops the slot. The consumer is expected to keep up.
//!  * The control Unix socket (`NCCL_PROFILER_CONTROL_SOCK`) is unrelated
//!    and must stay for gate RPC.
//!
//! This file is duplicated in `monitor/src/live_ring.rs`. Keep the two
//! byte-for-byte in sync (protected by `RING_LAYOUT_VERSION`).

use std::ffi::CString;
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// ASCII 'LIVERING' — validates a mapped file is actually a live-ring.
pub const RING_MAGIC: u64 = 0x4C495645_52494E47;
/// Bumped on any incompatible slot layout change.
pub const RING_LAYOUT_VERSION: u32 = 1;
/// Default slot count when the producer creates a new ring. Power of two.
pub const DEFAULT_RING_CAPACITY: u32 = 8192;

// Bounded string buffers embedded directly in the slot. Values are stored
// NUL-terminated. Anything longer is truncated on write, with the terminator
// preserved by the encode helpers.
pub const NAME_LEN: usize = 24;
pub const PARENT_NAME_LEN: usize = 16;
pub const COMM_NAME_LEN: usize = 32;
pub const HOST_LEN: usize = 32;
pub const IP_LEN: usize = 16;

/// Slot discriminator. `Empty` is reserved for uninitialized entries.
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SlotKind {
    Empty = 0,
    /// `NcclOpIssued` when the op is a collective.
    CollStart = 1,
    /// `NcclOpComplete` when the op is a collective (compact `online_complete`).
    CollComplete = 2,
    /// `NcclOpIssued` when the op is a P2P child (send/recv). Carries the
    /// parent group id in `group_id`.
    P2pStart = 3,
    /// `NcclOpComplete` when the op is a P2P child.
    P2pComplete = 4,
    /// `P2pParentStart` — first-child issue of a P2P_GROUP parent.
    P2pGroupStart = 5,
    /// `P2pGroupSeal` — final child-count marker.
    P2pGroupSeal = 6,
    /// `StepProgress` (proxy / kernel / kernel_ch), begin or complete.
    StepProgress = 7,
    /// First record of a multi-slot `CommInit` dump.
    CommInitHeader = 8,
    /// One member of the current `CommInit` dump.
    CommInitMember = 9,
    /// Same-host KernelStep copy timing of one completed op: `payload_bytes`
    /// summed send-step bytes, `step` summed GPU copy time (end - ready, ns),
    /// `n_children` step count, `ts` op end. Lets the monitor measure the
    /// intra-host copy bandwidth without the rich nested trace.
    KernelCopy = 10,
}

impl SlotKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(SlotKind::Empty),
            1 => Some(SlotKind::CollStart),
            2 => Some(SlotKind::CollComplete),
            3 => Some(SlotKind::P2pStart),
            4 => Some(SlotKind::P2pComplete),
            5 => Some(SlotKind::P2pGroupStart),
            6 => Some(SlotKind::P2pGroupSeal),
            7 => Some(SlotKind::StepProgress),
            8 => Some(SlotKind::CommInitHeader),
            10 => Some(SlotKind::KernelCopy),
            9 => Some(SlotKind::CommInitMember),
            _ => None,
        }
    }
}

// Encoded `source` for `StepProgress`.
pub const STEP_SOURCE_PROXY: u8 = 0;
pub const STEP_SOURCE_KERNEL: u8 = 1;
pub const STEP_SOURCE_KERNEL_CH: u8 = 2;

// Encoded `stage` for `StepProgress`.
pub const STEP_STAGE_POSTED: u8 = 0;
pub const STEP_STAGE_COMPLETE: u8 = 1;

/// A packed slot. Fixed size, `#[repr(C)]`, POD (no `String`/`Vec`). All
/// fields are always present; consumers must inspect `kind` before reading a
/// field. Unused fields carry sentinel values (`-1` for optional peers,
/// `0` for unset counters).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RingSlot {
    // ---- discriminator + small flags ---- offset 0
    pub kind: u8,
    pub is_send: u8,
    /// `STEP_SOURCE_*` for `StepProgress`; 0 otherwise.
    pub source: u8,
    /// `STEP_STAGE_*` for `StepProgress`; 0 otherwise.
    pub stage: u8,
    pub _pad0: u32,

    // ---- identity ---- offset 8
    pub rank: i32,
    pub world_rank: i32,

    // ---- timing (us) ---- offset 16
    pub ts: i64,
    pub launch_ts: i64,
    pub duration_us: i64,

    // ---- keys ---- offset 40
    pub comm_hash: u64,
    pub group_id: u64,
    pub seq_num: i64,

    // ---- payload sizes ---- offset 64
    pub payload_bytes: i64,
    pub self_copy_size: i64,

    // ---- P2P / StepProgress child identity ---- offset 80
    pub peer: i32,
    pub channel: i32,
    pub step: i64,
    pub size: i64,

    // ---- P2P_GROUP shape ---- offset 104
    pub n_children: u32,
    pub n_peers: u32,

    // ---- CommInit header ---- offset 112
    pub n_nodes: i32,
    pub n_ranks: i32,

    // ---- CommInit member ---- offset 120
    pub member_host_hash: u64,
    pub member_bus_id: i64,
    pub member_rank: i32,
    pub member_world_rank: i32,
    pub member_node: i32,
    pub member_cuda_dev: i32,

    // ---- name buffers ---- offset 160
    pub name: [u8; NAME_LEN],
    pub parent_name: [u8; PARENT_NAME_LEN],
    pub comm_name: [u8; COMM_NAME_LEN],
    pub host: [u8; HOST_LEN],
    pub ip: [u8; IP_LEN],
}

impl RingSlot {
    pub const fn zeroed() -> Self {
        RingSlot {
            kind: SlotKind::Empty as u8,
            is_send: 0,
            source: 0,
            stage: 0,
            _pad0: 0,
            rank: -1,
            world_rank: -1,
            ts: 0,
            launch_ts: 0,
            duration_us: 0,
            comm_hash: 0,
            group_id: 0,
            seq_num: 0,
            payload_bytes: 0,
            self_copy_size: 0,
            peer: -1,
            channel: -1,
            step: -1,
            size: 0,
            n_children: 0,
            n_peers: 0,
            n_nodes: 0,
            n_ranks: 0,
            member_host_hash: 0,
            member_bus_id: 0,
            member_rank: -1,
            member_world_rank: -1,
            member_node: -1,
            member_cuda_dev: -1,
            name: [0; NAME_LEN],
            parent_name: [0; PARENT_NAME_LEN],
            comm_name: [0; COMM_NAME_LEN],
            host: [0; HOST_LEN],
            ip: [0; IP_LEN],
        }
    }
}

/// Copy `src` bytes into `dst`, always leaving a trailing NUL. `dst` is
/// zeroed first so a short write does not leak previous state.
pub fn pack_str(dst: &mut [u8], src: &str) {
    for b in dst.iter_mut() {
        *b = 0;
    }
    let max = dst.len().saturating_sub(1);
    let bytes = src.as_bytes();
    let n = bytes.len().min(max);
    dst[..n].copy_from_slice(&bytes[..n]);
}

/// Extract a NUL-terminated string, dropping any garbage past the terminator.
pub fn unpack_str(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).into_owned()
}

/// Cache-line padded cell holding one SPSC cursor. Padding isolates each
/// cursor so producer and consumer do not false-share.
#[repr(C, align(128))]
pub struct AtomicCellU64 {
    pub value: AtomicU64,
    _pad: [u8; 120],
}

impl AtomicCellU64 {
    pub const fn new(v: u64) -> Self {
        AtomicCellU64 {
            value: AtomicU64::new(v),
            _pad: [0; 120],
        }
    }
}

#[repr(C, align(128))]
pub struct AtomicCellU32 {
    pub value: AtomicU32,
    _pad: [u8; 124],
}

impl AtomicCellU32 {
    pub const fn new(v: u32) -> Self {
        AtomicCellU32 {
            value: AtomicU32::new(v),
            _pad: [0; 124],
        }
    }
}

/// Fixed-size header at offset 0 of the ring file. Slots start immediately
/// after this struct (which is 128-B aligned so slot ptrs stay cache aligned).
#[repr(C, align(128))]
pub struct RingHeader {
    pub magic: u64,
    pub version: u32,
    pub slot_size: u32,
    pub capacity: u32,
    pub producer_pid: u32,
    pub reserved0: u64,
    pub reserved1: u64,
    /// Consumer's next read index. Monotonic; wraps only when interpreted
    /// modulo `capacity`.
    pub rd_idx: AtomicCellU64,
    /// Producer's committed write head. Consumer only reads slots strictly
    /// below this value.
    pub wr_idx: AtomicCellU64,
    /// Published-visible write head; separated so the producer can batch
    /// stores in `wr_idx` and only pay the release fence on `flush`.
    pub wr_idx_pub: AtomicCellU64,
    /// Number of slots the producer dropped because the ring was full.
    pub dropped: AtomicCellU64,
    /// Set to 1 once the producer closes the ring. Consumer treats this
    /// as an EOF equivalent (retires the rank detector).
    pub closed: AtomicCellU32,
}

pub const fn ring_file_size(capacity: u32) -> usize {
    size_of::<RingHeader>() + (capacity as usize) * size_of::<RingSlot>()
}

// -----------------------------------------------------------------
// Producer-side mmap wrapper.
// -----------------------------------------------------------------

/// Owning mmap of a producer-created ring file. Truncates + mmaps the file
/// on `create`, initializes the header, and provides SPSC push semantics.
pub struct RingWriter {
    mem: *mut libc::c_void,
    map_size: usize,
    capacity: u32,
    // Local write index cache so the SPSC hot path never reads shared state
    // on the happy case (only on wraparound / full).
    wr_idx_cache: u64,
    rd_idx_cache: u64,
    // Batching: how many uncommitted slots are buffered before we release.
    batch_size: u32,
    pending_commit: u32,
}

// SAFETY: the mmap region is process-private in the sense that no other
// thread inside this process touches it concurrently. The consumer runs in
// a different process and coordinates through atomics.
unsafe impl Send for RingWriter {}

impl RingWriter {
    /// Create + truncate `path` to hold `capacity` slots, then mmap it and
    /// initialize the header.
    pub fn create(path: &Path, capacity: u32, producer_pid: u32) -> std::io::Result<Self> {
        assert!(
            capacity.is_power_of_two(),
            "capacity must be a power of two"
        );
        let path_c = CString::new(path.as_os_str().to_str().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "non-utf8 ring path")
        })?)?;
        let map_size = ring_file_size(capacity);
        unsafe {
            // Mode 0666: the consumer (`comma-monitor`) needs to mmap the
            // file with PROT_READ | PROT_WRITE so it can publish `rd_idx`
            // back to the producer. Producer and consumer are typically
            // different local users (producer: NCCL training container as
            // root; consumer: unprivileged host user), so the file must be
            // world-writable inside the local-tmp ring directory.
            let fd = libc::open(
                path_c.as_ptr(),
                libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC,
                0o666,
            );
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }
            // Force mode 0666 regardless of process umask so the consumer
            // (running as an unprivileged user on the host) can open the
            // file O_RDWR and mmap it PROT_READ|PROT_WRITE.
            let _ = libc::fchmod(fd, 0o666);
            if libc::ftruncate(fd, map_size as libc::off_t) < 0 {
                let err = std::io::Error::last_os_error();
                libc::close(fd);
                return Err(err);
            }
            let mem = libc::mmap(
                std::ptr::null_mut(),
                map_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            // The fd is not needed after mmap; the map holds a reference to
            // the underlying file.
            libc::close(fd);
            if mem == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            // Zero-fill (ftruncate on a fresh file already zeros, but do it
            // defensively) and initialize the header.
            let hdr_ptr = mem as *mut RingHeader;
            std::ptr::write(
                hdr_ptr,
                RingHeader {
                    magic: RING_MAGIC,
                    version: RING_LAYOUT_VERSION,
                    slot_size: size_of::<RingSlot>() as u32,
                    capacity,
                    producer_pid,
                    reserved0: 0,
                    reserved1: 0,
                    rd_idx: AtomicCellU64::new(0),
                    wr_idx: AtomicCellU64::new(0),
                    wr_idx_pub: AtomicCellU64::new(0),
                    dropped: AtomicCellU64::new(0),
                    closed: AtomicCellU32::new(0),
                },
            );
            Ok(RingWriter {
                mem,
                map_size,
                capacity,
                wr_idx_cache: 0,
                rd_idx_cache: 0,
                batch_size: 32,
                pending_commit: 0,
            })
        }
    }

    #[inline]
    fn header(&self) -> &RingHeader {
        // SAFETY: `mem` points at a valid mmap of at least
        // `size_of::<RingHeader>()` bytes for the lifetime of `self`.
        unsafe { &*(self.mem as *const RingHeader) }
    }

    #[inline]
    fn slot_ptr(&self, idx: u64) -> *mut RingSlot {
        // SAFETY: `mem + size_of::<RingHeader>()` is the start of the slot
        // array; `capacity` slots follow. Masked `idx` is always in range.
        unsafe {
            let base = (self.mem as *mut u8).add(size_of::<RingHeader>()) as *mut RingSlot;
            base.add((idx & ((self.capacity as u64) - 1)) as usize)
        }
    }

    /// Push a slot. Returns `Ok(())` on success, `Err(())` if the ring is
    /// full (the producer's `dropped` counter is incremented). Never blocks.
    pub fn push(&mut self, slot: &RingSlot) -> Result<(), ()> {
        let capacity = self.capacity as u64;
        let wr = self.wr_idx_cache;
        // Fast path: local write head still leaves room compared to cached rd.
        if wr.wrapping_sub(self.rd_idx_cache) >= capacity {
            let fresh = self.header().rd_idx.value.load(Ordering::Acquire);
            self.rd_idx_cache = fresh;
            if wr.wrapping_sub(fresh) >= capacity {
                self.header().dropped.value.fetch_add(1, Ordering::Relaxed);
                return Err(());
            }
        }
        // SAFETY: slot pointer is in range; producer is single-threaded.
        unsafe {
            std::ptr::write_volatile(self.slot_ptr(wr), *slot);
        }
        self.wr_idx_cache = wr.wrapping_add(1);
        self.pending_commit = self.pending_commit.saturating_add(1);
        // Publish immediately if we have hit the batch bound.
        if self.pending_commit >= self.batch_size {
            self.flush();
        } else {
            // Keep wr_idx bumped so the consumer can also read straggler
            // records when it polls before the next flush; wr_idx_pub is
            // only updated with Release below.
            self.header()
                .wr_idx
                .value
                .store(self.wr_idx_cache, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Publish all pending slots with Release ordering so the consumer sees
    /// their contents. Called at every polling boundary.
    pub fn flush(&mut self) {
        let wr = self.wr_idx_cache;
        let hdr = self.header();
        hdr.wr_idx.value.store(wr, Ordering::Relaxed);
        hdr.wr_idx_pub.value.store(wr, Ordering::Release);
        self.pending_commit = 0;
    }

    /// Mark the ring as closed. Consumer treats this as EOF for the rank.
    pub fn close(&mut self) {
        self.flush();
        self.header().closed.value.store(1, Ordering::Release);
    }

    pub fn dropped(&self) -> u64 {
        self.header().dropped.value.load(Ordering::Relaxed)
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

impl Drop for RingWriter {
    fn drop(&mut self) {
        // Best-effort close + munmap. The underlying file stays on disk so
        // the consumer can drain remaining slots after we exit.
        let hdr = self.header();
        hdr.closed.value.store(1, Ordering::Release);
        unsafe {
            libc::munmap(self.mem, self.map_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_size_is_stable() {
        // The consumer relies on `slot_size` when it opens a ring. If this
        // assertion trips, bump `RING_LAYOUT_VERSION` and update the
        // monitor-side mirror of this file to match.
        assert!(size_of::<RingSlot>() <= 512);
        assert_eq!(size_of::<RingSlot>() % 8, 0);
    }

    #[test]
    fn header_layout_is_128_aligned() {
        assert_eq!(std::mem::align_of::<RingHeader>(), 128);
        assert_eq!(std::mem::align_of::<AtomicCellU64>(), 128);
    }

    #[test]
    fn pack_and_unpack_str_roundtrip() {
        let mut buf = [0u8; 8];
        pack_str(&mut buf, "hello");
        assert_eq!(&buf[..6], b"hello\0");
        assert_eq!(unpack_str(&buf), "hello");

        pack_str(&mut buf, "this_is_way_too_long");
        assert_eq!(buf[7], 0, "trailing NUL preserved on truncation");
        assert_eq!(unpack_str(&buf).len(), 7);
    }

    #[test]
    fn push_returns_err_on_full_and_bumps_dropped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("ring.bin");
        let mut w = RingWriter::create(&path, 4, 42).unwrap();
        let slot = RingSlot::zeroed();
        for _ in 0..4 {
            w.push(&slot).unwrap();
        }
        assert!(w.push(&slot).is_err());
        assert_eq!(w.dropped(), 1);
    }
}
