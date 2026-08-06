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

use crate::profiler;
use crate::profiler_shim;

use serde_json::json;

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::LazyLock;

pub fn datatype_num_bytes(datatype: i32) -> usize {
    /*
     * /* Data types */
     * typedef enum { ncclInt8       = 0, ncclChar       = 0,
     *                ncclUint8      = 1,
     *                ncclInt32      = 2, ncclInt        = 2,
     *                ncclUint32     = 3,
     *                ncclInt64      = 4,
     *                ncclUint64     = 5,
     *                ncclFloat16    = 6, ncclHalf       = 6,
     *                ncclFloat32    = 7, ncclFloat      = 7,
     *                ncclFloat64    = 8, ncclDouble     = 8,
     *                ncclBfloat16   = 9,
     * } ncclDataType_t;
     */
    match datatype {
        0..=1 => 1,
        2..=3 | 7 | 9 => 4,
        4..=5 | 8 => 8,
        6 => 2,
        _ => 1,
    }
}

static DATATYPE_NAME_TO_BYTES: LazyLock<HashMap<&'static CStr, usize>> = LazyLock::new(|| {
    [
        (c"ncclInt8", 1),
        (c"ncclInt32", 4),
        (c"ncclUint32", 4),
        (c"ncclInt64", 8),
        (c"ncclUint64", 8),
        (c"ncclFloat16", 2),
        (c"ncclFloat32", 4),
        (c"ncclFloat64", 8),
        (c"ncclBfloat16", 2),
        (c"ncclFloat8e4m3", 1),
        (c"ncclFloat8e5m2", 1),
    ]
    .iter()
    .cloned()
    .collect()
});

fn datatype_name_to_nbytes(name: &CStr) -> usize {
    // as an optimization we use the last few character to infer the datatype size
    let bytes = name.to_bytes();
    let i = bytes.len() - 1;
    match bytes[i] {
        b'8' => 1, // 8
        b'6' => 2, // 16
        b'2' => match bytes[i - 1] {
            b'3' => 4, // 32
            _ => 1,
        },
        b'4' => 8, // 64
        _ => DATATYPE_NAME_TO_BYTES.get(name).cloned().unwrap_or(1),
    }
}

/// # Safety
///
/// ptr should point to a valid null-terminated CStr
/// only ptr passed from NCCL (through profiler API) should be used as argument
unsafe fn datatype_c_str_ptr_to_nbytes(ptr: *const libc::c_char) -> usize {
    if !ptr.is_null() {
        datatype_name_to_nbytes(CStr::from_ptr(ptr))
    } else {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NcclOpType {
    Broadcast,
    Reduce,
    AllGather,
    ReduceScatter,
    AllReduce,
    Send,
    Recv,
    Unknown,
}

static NCCLOP_NAME_LOOKUP: LazyLock<HashMap<&'static CStr, NcclOpType>> = LazyLock::new(|| {
    [
        (c"Broadcast", NcclOpType::Broadcast),
        (c"Reduce", NcclOpType::Reduce),
        (c"AllGather", NcclOpType::AllGather),
        (c"ReduceScatter", NcclOpType::ReduceScatter),
        (c"AllReduce", NcclOpType::AllReduce),
        (c"Send", NcclOpType::Send),
        (c"Recv", NcclOpType::Recv),
    ]
    .iter()
    .cloned()
    .collect()
});

impl NcclOpType {
    pub fn from_nccl_func(nccl_func: u8) -> NcclOpType {
        match nccl_func {
            0 => NcclOpType::Broadcast,
            1 => NcclOpType::Reduce,
            2 => NcclOpType::AllGather,
            3 => NcclOpType::ReduceScatter,
            4 => NcclOpType::AllReduce,
            6 => NcclOpType::Send,
            7 => NcclOpType::Recv,
            _ => NcclOpType::Unknown,
        }
    }

    pub fn from_c_str(name: &CStr) -> NcclOpType {
        let bytes = name.to_bytes();
        let last_idx = bytes.len() - 1;
        match bytes[0] {
            b'B' => NcclOpType::Broadcast,
            b'R' => match bytes[last_idx] {
                b'e' => NcclOpType::Reduce,
                b'r' => NcclOpType::ReduceScatter,
                b'v' => NcclOpType::Recv,
                _ => NcclOpType::Unknown,
            },
            b'A' => match bytes[last_idx] {
                b'r' => NcclOpType::AllGather,
                b'e' => NcclOpType::AllReduce,
                _ => NcclOpType::Unknown,
            },
            b'S' => NcclOpType::Send,
            _ => NCCLOP_NAME_LOOKUP
                .get(name)
                .cloned()
                .unwrap_or(NcclOpType::Unknown),
        }
    }

    /// # Safety
    ///
    /// ptr should point to a valid null-terminated CStr
    pub unsafe fn from_c_str_ptr(ptr: *const libc::c_char) -> NcclOpType {
        if !ptr.is_null() {
            Self::from_c_str(CStr::from_ptr(ptr))
        } else {
            NcclOpType::Unknown
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            NcclOpType::Broadcast => "broadcast",
            NcclOpType::Reduce => "reduce",
            NcclOpType::AllGather => "all_gather",
            NcclOpType::ReduceScatter => "reduce_scatter",
            NcclOpType::AllReduce => "all_reduce",
            NcclOpType::Send => "send",
            NcclOpType::Recv => "recv",
            _ => "unknown",
        }
    }

    pub fn name_cstr(&self) -> &'static CStr {
        match self {
            NcclOpType::Broadcast => c"broadcast",
            NcclOpType::Reduce => c"reduce",
            NcclOpType::AllGather => c"all_gather",
            NcclOpType::ReduceScatter => c"reduce_scatter",
            NcclOpType::AllReduce => c"all_reduce",
            NcclOpType::Send => c"send",
            NcclOpType::Recv => c"recv",
            _ => c"unknown",
        }
    }
}

trait IntoNcclOpType {
    unsafe fn into_ncclop_type(self) -> NcclOpType;
}

impl IntoNcclOpType for u8 {
    /// # Safety
    ///
    /// the number must be a valid "nccl func" field from NCCL core
    unsafe fn into_ncclop_type(self) -> NcclOpType {
        NcclOpType::from_nccl_func(self)
    }
}

impl IntoNcclOpType for *const libc::c_char {
    /// # Safety
    ///
    /// the number must be a valid "nccl func" pointer from NCCL core
    unsafe fn into_ncclop_type(self) -> NcclOpType {
        NcclOpType::from_c_str_ptr(self)
    }
}

pub mod algo {
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::sync::LazyLock;

    pub const TREE: u8 = 0;
    pub const RING: u8 = 1;
    pub const COLLNET_DIRECT: u8 = 2;
    pub const COLLNET_CHAIN: u8 = 3;
    pub const NVLS: u8 = 4;
    pub const NVLS_TREE: u8 = 5;
    pub const PAT: u8 = 6;
    pub const UNKNOWN: u8 = 7;

    // this extra mapping decouples what NCCL use as algo names as what profiler
    // uses for reporting them
    pub(super) static ALGO_NAME_LOOKUP: LazyLock<HashMap<&'static CStr, u8>> =
        LazyLock::new(|| {
            [
                (c"TREE", TREE),
                (c"RING", RING),
                (c"COLLNET_DIRECT", COLLNET_DIRECT),
                (c"COLLNET_CHAIN", COLLNET_CHAIN),
                (c"NVLS", NVLS),
                (c"NVLS_TREE", NVLS_TREE),
                (c"PAT", PAT),
                (c"Unknown", UNKNOWN),
            ]
            .iter()
            .cloned()
            .collect()
        });

    /// # Safety
    ///
    /// ptr should point to a valid null-terminated CStr
    pub unsafe fn from_c_str_ptr(ptr: *const libc::c_char) -> u8 {
        if !ptr.is_null() {
            from_name(CStr::from_ptr(ptr))
        } else {
            UNKNOWN
        }
    }

    pub fn from_name(name: &CStr) -> u8 {
        let bytes = name.to_bytes();
        let last_idx = bytes.len() - 1;
        match bytes[0] {
            b'T' => TREE,
            b'R' => RING,
            b'C' => match bytes[last_idx] {
                b'T' => COLLNET_DIRECT,
                _ => COLLNET_CHAIN,
            },
            b'N' => match bytes[last_idx] {
                b'E' => NVLS_TREE,
                _ => NVLS,
            },
            b'P' => PAT,
            _ => ALGO_NAME_LOOKUP.get(name).cloned().unwrap_or(UNKNOWN),
        }
    }

    pub fn name(algo: u8) -> &'static str {
        match algo {
            TREE => "tree",
            RING => "ring",
            COLLNET_DIRECT => "collnet_direct",
            COLLNET_CHAIN => "collnet_chain",
            NVLS => "nvls",
            NVLS_TREE => "nvls_tree",
            PAT => "pat",
            _ => "unknown",
        }
    }

    pub fn name_cstr(algo: u8) -> &'static std::ffi::CStr {
        match algo {
            TREE => c"tree",
            RING => c"ring",
            COLLNET_DIRECT => c"collnet_direct",
            COLLNET_CHAIN => c"collnet_chain",
            NVLS => c"nvls",
            NVLS_TREE => c"nvls_tree",
            PAT => c"pat",
            _ => c"unknown",
        }
    }

    pub trait IntoAlgo {
        /// # Safety
        ///
        /// must be valid algo from NCCL core
        unsafe fn into_algo(self) -> u8;
    }

    impl IntoAlgo for u8 {
        unsafe fn into_algo(self) -> u8 {
            self
        }
    }

    impl IntoAlgo for *const libc::c_char {
        unsafe fn into_algo(self) -> u8 {
            from_c_str_ptr(self)
        }
    }
}

pub mod proto {
    use std::collections::HashMap;
    use std::ffi::CStr;
    use std::sync::LazyLock;

    pub const LL: u8 = 0;
    pub const LL128: u8 = 1;
    pub const SIMPLE: u8 = 2;
    pub const UNKNOWN: u8 = 3;

    // this extra mapping decouples what NCCL use as proto names as what profiler
    // uses for reporting them
    static PROTO_NAME_LOOKUP: LazyLock<HashMap<&'static CStr, u8>> = LazyLock::new(|| {
        [
            (c"LL", LL),
            (c"LL128", LL128),
            (c"SIMPLE", SIMPLE),
            (c"Unknown", UNKNOWN),
        ]
        .iter()
        .cloned()
        .collect()
    });

    /// # Safety
    ///
    /// ptr should point to a valid null-terminated CStr
    pub unsafe fn from_c_str_ptr(ptr: *const libc::c_char) -> u8 {
        if !ptr.is_null() {
            from_name(CStr::from_ptr(ptr))
        } else {
            UNKNOWN
        }
    }

    pub fn from_name(name: &CStr) -> u8 {
        PROTO_NAME_LOOKUP.get(name).cloned().unwrap_or(UNKNOWN)
    }

    pub fn name(algo: u8) -> &'static str {
        match algo {
            LL => "LL",
            LL128 => "LL128",
            SIMPLE => "SIMPLE",
            _ => "unknown",
        }
    }

    pub trait IntoProto {
        /// # Safety
        ///
        /// must be a valid proto passed from NCCL core
        unsafe fn into_proto(self) -> u8;
    }

    impl IntoProto for u8 {
        unsafe fn into_proto(self) -> u8 {
            self
        }
    }

    impl IntoProto for *const libc::c_char {
        unsafe fn into_proto(self) -> u8 {
            from_c_str_ptr(self)
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum NcclOpKey {
    Collective(
        /* comm = */ u64,
        /* rank = */ usize,
        NcclOpType,
        /* algo = */ u8,
        /* proto = */ u8,
    ),
    P2pSend(
        /* comm = */ u64,
        /* local = */ usize,
        /* peer = */ usize,
    ),
    P2pRecv(
        /* comm = */ u64,
        /* local = */ usize,
        /* peer = */ usize,
    ),
    NetSend(
        /* comm = */ u64,
        /* local = */ usize,
        /* peer = */ usize,
    ),
    NetRecv(
        /* comm = */ u64,
        /* local = */ usize,
        /* peer = */ usize,
    ),
}

impl NcclOpKey {
    fn from_descr<D>(descr: &D, alt_comm_hash: u64) -> Option<Self>
    where
        D: Version + Event,
    {
        match descr.type_() as u32 {
            profiler_shim::ncclProfileColl => {
                // SAFETY: just checked that event type is collective
                let coll = unsafe { descr.cast_to_coll() };
                Some(coll.op_key(alt_comm_hash))
            }
            profiler_shim::ncclProfileP2p => {
                // SAFETY: just checked that event type is p2p
                let p2p = unsafe { descr.cast_to_p2p() };
                Some(p2p.op_key(alt_comm_hash))
            }
            _ => None,
        }
    }

    pub fn get_comm_hash(&self) -> u64 {
        match self {
            Self::Collective(comm, _, _, _, _) => *comm,
            Self::P2pSend(comm, _, _) => *comm,
            Self::P2pRecv(comm, _, _) => *comm,
            Self::NetSend(comm, _, _) => *comm,
            Self::NetRecv(comm, _, _) => *comm,
        }
    }

    pub fn get_coll_op_type(&self) -> NcclOpType {
        match self {
            Self::Collective(_, _, op_type, _, _) => *op_type,
            _ => NcclOpType::Unknown,
        }
    }

    pub fn local_rank(&self) -> usize {
        match self {
            Self::Collective(_, r, _, _, _) => *r,
            Self::P2pSend(_, r, _) => *r,
            Self::P2pRecv(_, r, _) => *r,
            Self::NetSend(_, r, _) => *r,
            Self::NetRecv(_, r, _) => *r,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Collective(comm, rank, op_type, algo, proto) => json!({
                "cat": "collective",
                "op": op_type.name(),
                "comm": format!("0x{:016x}", comm),
                "rank": rank,
                "algo": algo::name(*algo),
                "proto": proto::name(*proto),
            }),
            Self::P2pSend(comm, local, peer) => json!({
                "cat": "point-to-point",
                "op": "send",
                "comm": format!("0x{:016x}", comm),
                "local_rank": local,
                "peer_rank": peer,
            }),
            Self::P2pRecv(comm, local, peer) => json!({
                "cat": "point-to-point",
                "op": "recv",
                "comm": format!("0x{:016x}", comm),
                "local_rank": local,
                "peer_rank": peer,
            }),
            Self::NetSend(comm, local, peer) => json!({
                "cat": "net",
                "op": "send",
                "comm": format!("0x{:016x}", comm),
                "local_rank": local,
                "peer_rank": peer,
            }),
            Self::NetRecv(comm, local, peer) => json!({
                "cat": "net",
                "op": "recv",
                "comm": format!("0x{:016x}", comm),
                "local_rank": local,
                "peer_rank": peer,
            }),
        }
    }
}

fn event_byte_count<D>(descr: &D) -> usize
where
    D: Version + Event,
{
    match descr.type_() as u32 {
        profiler_shim::ncclProfileColl => {
            // SAFETY: just checked that event type is collective
            let coll = unsafe { descr.cast_to_coll() };
            coll.byte_count()
        }
        profiler_shim::ncclProfileP2p => {
            // SAFETY: just checked that event type is p2p
            let p2p = unsafe { descr.cast_to_p2p() };
            p2p.byte_count()
        }
        _ => 0,
    }
}

#[derive(Debug, Clone)]
pub enum EventMetadata {
    V1(profiler_shim::EventDescrV1),
    V2(profiler_shim::EventDescrV2),
    V3(profiler_shim::EventDescrV3),
    V4(profiler_shim::EventDescrV4),
    V6(profiler_shim::EventDescrV6),
}

/// # Safety
///
/// EventMetadata treat pointers in the EventDescr struct
/// as plain data without ever dereferencing them.
unsafe impl Send for EventMetadata {}
unsafe impl Sync for EventMetadata {}

impl EventMetadata {
    pub fn try_get_op_key(&self, alt_comm_hash: u64) -> Option<NcclOpKey> {
        match self {
            Self::V1(descr) => NcclOpKey::from_descr(descr, alt_comm_hash),
            Self::V2(descr) => NcclOpKey::from_descr(descr, alt_comm_hash),
            Self::V3(descr) => NcclOpKey::from_descr(descr, alt_comm_hash),
            Self::V4(descr) => NcclOpKey::from_descr(descr, alt_comm_hash),
            Self::V6(descr) => NcclOpKey::from_descr(descr, alt_comm_hash),
        }
    }

    pub fn get_op_key(&self, alt_comm_hash: u64) -> NcclOpKey {
        self.try_get_op_key(alt_comm_hash).unwrap()
    }

    pub fn byte_count(&self) -> usize {
        match self {
            Self::V1(descr) => event_byte_count(descr),
            Self::V2(descr) => event_byte_count(descr),
            Self::V3(descr) => event_byte_count(descr),
            Self::V4(descr) => event_byte_count(descr),
            Self::V6(descr) => event_byte_count(descr),
        }
    }

    pub fn try_cast_to_coll(&self) -> Option<&dyn Coll> {
        match self {
            Self::V1(descr) => descr.try_cast_to_coll().map(|x| x as _),
            Self::V2(descr) => descr.try_cast_to_coll().map(|x| x as _),
            Self::V3(descr) => descr.try_cast_to_coll().map(|x| x as _),
            Self::V4(descr) => descr.try_cast_to_coll().map(|x| x as _),
            Self::V6(descr) => descr.try_cast_to_coll().map(|x| x as _),
        }
    }

    pub fn try_cast_to_p2p(&self) -> Option<&dyn P2p> {
        match self {
            Self::V1(descr) => descr.try_cast_to_p2p().map(|x| x as _),
            Self::V2(descr) => descr.try_cast_to_p2p().map(|x| x as _),
            Self::V3(descr) => descr.try_cast_to_p2p().map(|x| x as _),
            Self::V4(descr) => descr.try_cast_to_p2p().map(|x| x as _),
            Self::V6(descr) => descr.try_cast_to_p2p().map(|x| x as _),
        }
    }
}

pub trait Event {
    fn rank(&self) -> i32;
    fn type_(&self) -> u64;
    fn parent_obj(&self) -> *mut libc::c_void;
    fn clone_to_metadata(&self) -> EventMetadata;
}

pub trait Version: Event {
    type Coll: Coll;
    type P2p: P2p;
    type ProxyOp: ProxyOp;
    type ProxyStep: ProxyStep;
    type KernelStep: KernelStep;

    fn version() -> profiler::Version;

    /// # Safety
    ///
    /// type of this descriptor must be collective
    #[inline(always)]
    unsafe fn cast_to_coll(&self) -> &Self::Coll {
        let ptr = self as *const Self;
        &*ptr.cast()
    }

    /// # Safety
    ///
    /// type of this descriptor must be p2p
    #[inline(always)]
    unsafe fn cast_to_p2p(&self) -> &Self::P2p {
        let ptr = self as *const Self;
        &*ptr.cast()
    }

    /// # Safety
    ///
    /// type of this descriptor must be proxyop
    #[inline(always)]
    unsafe fn cast_to_proxyop(&self) -> &Self::ProxyOp {
        let ptr = self as *const Self;
        &*ptr.cast()
    }

    /// # Safety
    ///
    /// type of this descriptor must be proxystep
    #[inline(always)]
    unsafe fn cast_to_proxystep(&self) -> &Self::ProxyStep {
        let ptr = self as *const Self;
        &*ptr.cast()
    }

    /// # Safety
    ///
    /// type of this descriptor must be kernelstep (API v6+)
    #[inline(always)]
    unsafe fn cast_to_kernelstep(&self) -> &Self::KernelStep {
        let ptr = self as *const Self;
        &*ptr.cast()
    }

    #[inline(always)]
    fn try_cast_to_coll(&self) -> Option<&Self::Coll> {
        if self.type_() as u32 == profiler_shim::ncclProfileColl {
            // SAFETY: just checked that event type is collective
            Some(unsafe { self.cast_to_coll() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn try_cast_to_p2p(&self) -> Option<&Self::P2p> {
        if self.type_() as u32 == profiler_shim::ncclProfileP2p {
            // SAFETY: just checked that event type is collective
            Some(unsafe { self.cast_to_p2p() })
        } else {
            None
        }
    }
}

pub trait NcclOp: Event {
    fn comm_hash(&self) -> Option<u64>;
    fn byte_count(&self) -> usize;
}

pub trait Coll: NcclOp {
    #![allow(dead_code)]
    fn algo(&self) -> u8;
    fn proto(&self) -> u8;
    fn seq_num(&self) -> u64;
    fn n_max_channel(&self) -> u8;
    fn op_type(&self) -> NcclOpType;
    fn op_key(&self, alt_comm_hash: u64) -> NcclOpKey {
        NcclOpKey::Collective(
            self.comm_hash().unwrap_or(alt_comm_hash),
            self.rank() as _,
            self.op_type(),
            self.algo(),
            self.proto(),
        )
    }
}

pub trait P2p: NcclOp {
    #![allow(dead_code)]
    fn peer(&self) -> i32;
    fn is_send(&self) -> bool;
    fn op_key(&self, alt_comm_hash: u64) -> NcclOpKey {
        let ctor = if self.is_send() {
            NcclOpKey::P2pSend
        } else {
            NcclOpKey::P2pRecv
        };
        ctor(
            self.comm_hash().unwrap_or(alt_comm_hash),
            self.rank() as _,
            self.peer() as _,
        )
    }
}

pub trait ProxyOp: Event {
    #![allow(dead_code)]
    fn peer(&self) -> i32;
    fn is_send(&self) -> bool;
    fn pid(&self) -> libc::pid_t;
    fn n_steps(&self) -> i32;
    fn chunk_size(&self) -> i32;
    fn channel_id(&self) -> u8;
    fn net_op(&self, comm_hash: u64) -> NcclOpKey {
        let ctor = if self.is_send() {
            NcclOpKey::NetSend
        } else {
            NcclOpKey::NetRecv
        };
        ctor(comm_hash, self.rank() as _, self.peer() as _)
    }
}

pub trait ProxyStep: Event {
    fn step(&self) -> i32;
}

/// Per-slice Simple-prims kernel step (profiler API v6+).
pub trait KernelStep: Event {
    fn channel_id(&self) -> u8;
    fn is_send(&self) -> bool;
    fn peer(&self) -> u8;
    fn step(&self) -> u32;
    fn size(&self) -> u32;
    /// Wait/step begin (GPU globaltimer); 0 if none/recv.
    fn start_ts(&self) -> u64;
    /// Transfer/comm begin (GPU globaltimer).
    fn ready_ts(&self) -> u64;
    /// CoMMA ProxyStep-style wait: `ready_ts - start_ts` when `start_ts != 0`.
    fn fifo_wait_dur_ns(&self) -> u32 {
        derive_fifo_wait_dur_ns(self.start_ts(), self.ready_ts())
    }
}

/// Derive ProxyStep-compatible `fifo_wait_dur_ns` from NCCL `startTs`/`readyTs`.
#[inline(always)]
pub fn derive_fifo_wait_dur_ns(start_ts: u64, ready_ts: u64) -> u32 {
    if start_ts == 0 || ready_ts <= start_ts {
        return 0;
    }
    let d = ready_ts - start_ts;
    if d > u32::MAX as u64 {
        u32::MAX
    } else {
        d as u32
    }
}

pub trait ProxyOpState {
    fn version() -> profiler::Version;
    fn steps(&self) -> i32;
    fn trans_size(&self) -> usize;
}

pub trait ProxyStepState {
    fn trans_size(&self) -> usize;
}

macro_rules! impl_event {
    ($t:tt, $v:expr) => {
        impl Event for $t {
            #[inline(always)]
            fn rank(&self) -> i32 {
                self.0.rank()
            }

            #[inline(always)]
            fn type_(&self) -> u64 {
                self.0.type_() as u64
            }

            #[inline(always)]
            fn parent_obj(&self) -> *mut libc::c_void {
                self.0.parent_obj()
            }

            #[inline(always)]
            fn clone_to_metadata(&self) -> EventMetadata {
                $v(self.0.clone())
            }
        }
    };
}

macro_rules! descr_impl_event {
    ($t:ty, $v:expr) => {
        impl Event for $t {
            #[inline(always)]
            fn rank(&self) -> i32 {
                self.0.rank
            }

            #[inline(always)]
            fn type_(&self) -> u64 {
                self.0.type_ as u64
            }

            #[inline(always)]
            fn parent_obj(&self) -> *mut libc::c_void {
                self.0.parentObj
            }

            #[inline(always)]
            fn clone_to_metadata(&self) -> EventMetadata {
                $v(self.clone())
            }
        }
    };
}

macro_rules! impl_ncclop {
    ($t:ty, $f:tt, $sz_f:expr) => {
        impl NcclOp for $t {
            // SAFETY: this type could only be constructed via cast_*(),
            // which must be called with type_ == ncclProfileColl or
            // type_ == ncclProfilerP2p.
            // Therefore accessing the corresponding union field is safe.

            #[inline(always)]
            fn comm_hash(&self) -> Option<u64> {
                Some(unsafe { self.0 .0.__bindgen_anon_1.$f.commHash })
            }

            #[inline(always)]
            fn byte_count(&self) -> usize {
                #![allow(unused_unsafe)]
                let inner = unsafe { &self.0 .0.__bindgen_anon_1.$f };
                let dt_bytes = unsafe { $sz_f(inner.datatype as _) };
                (inner.count * dt_bytes) as _
            }
        }
    };
}

macro_rules! impl_coll {
    ($t:ty, $ch_f:tt) => {
        impl super::Coll for $t {
            // SAFETY: this type could only be constructed via cast_*(),
            // which must be called with type_ == ncclProfileColl.
            // Therefore accessing the corresponding union field is safe.

            #[inline(always)]
            fn algo(&self) -> u8 {
                unsafe { self.0 .0.__bindgen_anon_1.coll.algo.into_algo() }
            }

            #[inline(always)]
            fn proto(&self) -> u8 {
                unsafe { self.0 .0.__bindgen_anon_1.coll.proto.into_proto() }
            }

            #[inline(always)]
            fn seq_num(&self) -> u64 {
                unsafe { self.0 .0.__bindgen_anon_1.coll.seqNumber }
            }

            #[inline(always)]
            fn n_max_channel(&self) -> u8 {
                unsafe { self.0 .0.__bindgen_anon_1.coll.$ch_f }
            }

            #[inline(always)]
            fn op_type(&self) -> NcclOpType {
                unsafe {
                    let coll = &self.0 .0.__bindgen_anon_1.coll;
                    coll.func.into_ncclop_type()
                }
            }
        }
    };
}

macro_rules! impl_p2p {
    ($t:ty) => {
        impl super::P2p for $t {
            // SAFETY: this type could only be constructed via cast_*(),
            // which must be called with type_ == ncclProfileP2p.
            // Therefore accessing the corresponding union field is safe.

            #[inline(always)]
            fn peer(&self) -> i32 {
                let p2p = unsafe { &self.0 .0.__bindgen_anon_1.p2p };
                p2p.peer
            }

            #[inline(always)]
            fn is_send(&self) -> bool {
                unsafe {
                    let p2p = &self.0 .0.__bindgen_anon_1.p2p;
                    let op_type = p2p.func.into_ncclop_type();
                    op_type == NcclOpType::Send
                }
            }
        }
    };
}

macro_rules! impl_proxyop {
    ($t:ty) => {
        impl super::ProxyOp for $t {
            // SAFETY: this type could only be constructed via cast_*(),
            // which must be called with type_ == ncclProfileProxyOp.
            // Therefore accessing the corresponding union field is safe.

            #[inline(always)]
            fn peer(&self) -> i32 {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.peer
            }

            #[inline(always)]
            fn is_send(&self) -> bool {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.isSend != 0
            }

            #[inline(always)]
            fn pid(&self) -> libc::pid_t {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.pid
            }

            #[inline(always)]
            fn n_steps(&self) -> i32 {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.nSteps
            }

            #[inline(always)]
            fn chunk_size(&self) -> i32 {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.chunkSize
            }

            #[inline(always)]
            fn channel_id(&self) -> u8 {
                let proxyop = unsafe { &self.0 .0.__bindgen_anon_1.proxyOp };
                proxyop.channelId
            }
        }
    };
}

macro_rules! impl_proxyop_state {
    ($t:ty, $ver:expr) => {
        impl ProxyOpState for $t {
            // SAFETY: this type could only be constructed via unsafe method,
            // which must be called when nccl is issuing callback for proxyop.
            // Therefore accessing the corresponding union field is safe.

            #[inline(always)]
            fn version() -> profiler::Version {
                $ver
            }

            #[inline(always)]
            fn steps(&self) -> i32 {
                unsafe { self.0.proxyOp.steps }
            }

            #[inline(always)]
            fn trans_size(&self) -> usize {
                unsafe { self.0.proxyOp.transSize }
            }
        }
    };
}

macro_rules! def_proxystep {
    ($t:tt, $d:tt) => {
        #[allow(unused_parens)]
        #[repr(transparent)]
        pub struct $t($d);
        impl super::ProxyStep for $t {
            fn step(&self) -> i32 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.proxyStep };
                s.step
            }
        }
    };
}

/// Stub KernelStep for profiler API versions that lack the union field.
/// Must never be cast to outside of dead code paths (mask gated on V6).
macro_rules! def_kernelstep_unsupported {
    ($t:tt, $d:tt) => {
        #[allow(unused_parens)]
        #[repr(transparent)]
        pub struct $t($d);
        impl super::KernelStep for $t {
            fn channel_id(&self) -> u8 {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn is_send(&self) -> bool {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn peer(&self) -> u8 {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn step(&self) -> u32 {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn size(&self) -> u32 {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn start_ts(&self) -> u64 {
                panic!("KernelStep unsupported on this profiler API version")
            }
            fn ready_ts(&self) -> u64 {
                panic!("KernelStep unsupported on this profiler API version")
            }
        }
    };
}

macro_rules! def_kernelstep {
    ($t:tt, $d:tt) => {
        #[allow(unused_parens)]
        #[repr(transparent)]
        pub struct $t($d);
        impl super::KernelStep for $t {
            #[inline(always)]
            fn channel_id(&self) -> u8 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.channelId
            }
            #[inline(always)]
            fn is_send(&self) -> bool {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.isSend != 0
            }
            #[inline(always)]
            fn peer(&self) -> u8 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.peer
            }
            #[inline(always)]
            fn step(&self) -> u32 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.step
            }
            #[inline(always)]
            fn size(&self) -> u32 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.size
            }
            #[inline(always)]
            fn start_ts(&self) -> u64 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.startTs
            }
            #[inline(always)]
            fn ready_ts(&self) -> u64 {
                let s = unsafe { &self.0 .0.__bindgen_anon_1.kernelStep };
                s.readyTs
            }
        }
    };
}

mod v1 {
    use super::algo::IntoAlgo;
    use super::proto::IntoProto;
    use super::*;

    #[repr(transparent)]
    pub struct Coll(profiler_shim::EventDescrV1);

    impl_event!(Coll, EventMetadata::V1);
    impl_ncclop!(Coll, coll, datatype_num_bytes);
    impl_coll!(Coll, nMaxChannels);

    #[repr(transparent)]
    pub struct P2p(profiler_shim::EventDescrV1);

    impl_event!(P2p, EventMetadata::V1);
    impl_ncclop!(P2p, p2p, datatype_num_bytes);
    impl_p2p!(P2p);

    #[repr(transparent)]
    pub struct ProxyOp(profiler_shim::EventDescrV1);

    impl_event!(ProxyOp, EventMetadata::V1);
    impl_proxyop!(ProxyOp);

    def_proxystep!(ProxyStep, (profiler_shim::EventDescrV1));
    impl_event!(ProxyStep, EventMetadata::V1);

    def_kernelstep_unsupported!(KernelStep, (profiler_shim::EventDescrV1));
    impl_event!(KernelStep, EventMetadata::V1);
}

mod v2 {
    use super::algo::IntoAlgo;
    use super::proto::IntoProto;
    use super::*;

    #[repr(transparent)]
    pub struct Coll(profiler_shim::EventDescrV2);

    impl_event!(Coll, EventMetadata::V2);
    impl_ncclop!(Coll, coll, datatype_c_str_ptr_to_nbytes);
    impl_coll!(Coll, nMaxChannels);

    #[repr(transparent)]
    pub struct P2p(profiler_shim::EventDescrV2);

    impl_event!(P2p, EventMetadata::V2);
    impl_ncclop!(P2p, p2p, datatype_c_str_ptr_to_nbytes);
    impl_p2p!(P2p);

    #[repr(transparent)]
    pub struct ProxyOp(profiler_shim::EventDescrV2);

    impl_event!(ProxyOp, EventMetadata::V2);
    impl_proxyop!(ProxyOp);

    def_proxystep!(ProxyStep, (profiler_shim::EventDescrV2));
    impl_event!(ProxyStep, EventMetadata::V2);

    def_kernelstep_unsupported!(KernelStep, (profiler_shim::EventDescrV2));
    impl_event!(KernelStep, EventMetadata::V2);
}

mod v3 {
    use super::algo::IntoAlgo;
    use super::proto::IntoProto;
    use super::*;

    #[repr(transparent)]
    pub struct Coll(profiler_shim::EventDescrV3);

    impl_event!(Coll, EventMetadata::V3);
    impl_ncclop!(Coll, coll, datatype_c_str_ptr_to_nbytes);
    impl_coll!(Coll, nMaxChannels);

    #[repr(transparent)]
    pub struct P2p(profiler_shim::EventDescrV3);

    impl_event!(P2p, EventMetadata::V3);
    impl_ncclop!(P2p, p2p, datatype_c_str_ptr_to_nbytes);
    impl_p2p!(P2p);

    #[repr(transparent)]
    pub struct ProxyOp(profiler_shim::EventDescrV3);

    impl_event!(ProxyOp, EventMetadata::V3);
    impl_proxyop!(ProxyOp);

    def_proxystep!(ProxyStep, (profiler_shim::EventDescrV3));
    impl_event!(ProxyStep, EventMetadata::V3);

    def_kernelstep_unsupported!(KernelStep, (profiler_shim::EventDescrV3));
    impl_event!(KernelStep, EventMetadata::V3);
}

mod v4 {
    use super::algo::IntoAlgo;
    use super::proto::IntoProto;
    use super::*;

    #[repr(transparent)]
    pub struct Coll(profiler_shim::EventDescrV4);

    impl_event!(Coll, EventMetadata::V4);

    impl NcclOp for Coll {
        // SAFETY: this type could only be constructed via cast_*(),
        // which must be called with type_ == ncclProfileColl.
        // Therefore accessing the corresponding union field is safe.

        #[inline(always)]
        fn comm_hash(&self) -> Option<u64> {
            None
        }

        #[inline(always)]
        fn byte_count(&self) -> usize {
            let coll = unsafe { &self.0 .0.__bindgen_anon_1.coll };
            let dt_bytes = unsafe { datatype_c_str_ptr_to_nbytes(coll.datatype) };
            (coll.count * dt_bytes) as _
        }
    }

    impl_coll!(Coll, nChannels);

    #[repr(transparent)]
    pub struct P2p(profiler_shim::EventDescrV4);

    impl_event!(P2p, EventMetadata::V4);

    impl NcclOp for P2p {
        // SAFETY: this type could only be constructed via cast_*(),
        // which must be called with type_ == ncclProfileP2p.
        // Therefore accessing the corresponding union field is safe.

        #[inline(always)]
        fn comm_hash(&self) -> Option<u64> {
            None
        }

        #[inline(always)]
        fn byte_count(&self) -> usize {
            let p2p = unsafe { &self.0 .0.__bindgen_anon_1.p2p };
            let dt_bytes = unsafe { datatype_c_str_ptr_to_nbytes(p2p.datatype) };
            (p2p.count * dt_bytes) as _
        }
    }

    impl_p2p!(P2p);

    #[repr(transparent)]
    pub struct ProxyOp(profiler_shim::EventDescrV4);

    impl_event!(ProxyOp, EventMetadata::V4);
    impl_proxyop!(ProxyOp);

    def_proxystep!(ProxyStep, (profiler_shim::EventDescrV4));
    impl_event!(ProxyStep, EventMetadata::V4);

    def_kernelstep_unsupported!(KernelStep, (profiler_shim::EventDescrV4));
    impl_event!(KernelStep, EventMetadata::V4);
}

mod v6 {
    use super::algo::IntoAlgo;
    use super::proto::IntoProto;
    use super::*;

    #[repr(transparent)]
    pub struct Coll(profiler_shim::EventDescrV6);

    impl_event!(Coll, EventMetadata::V6);

    impl NcclOp for Coll {
        // SAFETY: this type could only be constructed via cast_*(),
        // which must be called with type_ == ncclProfileColl.
        // Therefore accessing the corresponding union field is safe.

        #[inline(always)]
        fn comm_hash(&self) -> Option<u64> {
            None
        }

        #[inline(always)]
        fn byte_count(&self) -> usize {
            let coll = unsafe { &self.0 .0.__bindgen_anon_1.coll };
            let dt_bytes = unsafe { datatype_c_str_ptr_to_nbytes(coll.datatype) };
            (coll.count * dt_bytes) as _
        }
    }

    impl_coll!(Coll, nChannels);

    #[repr(transparent)]
    pub struct P2p(profiler_shim::EventDescrV6);

    impl_event!(P2p, EventMetadata::V6);

    impl NcclOp for P2p {
        // SAFETY: this type could only be constructed via cast_*(),
        // which must be called with type_ == ncclProfileP2p.
        // Therefore accessing the corresponding union field is safe.

        #[inline(always)]
        fn comm_hash(&self) -> Option<u64> {
            None
        }

        #[inline(always)]
        fn byte_count(&self) -> usize {
            let p2p = unsafe { &self.0 .0.__bindgen_anon_1.p2p };
            let dt_bytes = unsafe { datatype_c_str_ptr_to_nbytes(p2p.datatype) };
            (p2p.count * dt_bytes) as _
        }
    }

    impl_p2p!(P2p);

    #[repr(transparent)]
    pub struct ProxyOp(profiler_shim::EventDescrV6);

    impl_event!(ProxyOp, EventMetadata::V6);
    impl_proxyop!(ProxyOp);

    def_proxystep!(ProxyStep, (profiler_shim::EventDescrV6));
    impl_event!(ProxyStep, EventMetadata::V6);

    def_kernelstep!(KernelStep, (profiler_shim::EventDescrV6));
    impl_event!(KernelStep, EventMetadata::V6);
}

descr_impl_event!(profiler_shim::EventDescrV1, EventMetadata::V1);

impl Version for profiler_shim::EventDescrV1 {
    type Coll = v1::Coll;
    type P2p = v1::P2p;
    type ProxyOp = v1::ProxyOp;
    type ProxyStep = v1::ProxyStep;
    type KernelStep = v1::KernelStep;

    fn version() -> profiler::Version {
        profiler::Version::V1
    }
}

descr_impl_event!(profiler_shim::EventDescrV2, EventMetadata::V2);

impl Version for profiler_shim::EventDescrV2 {
    type Coll = v2::Coll;
    type P2p = v2::P2p;
    type ProxyOp = v2::ProxyOp;
    type ProxyStep = v2::ProxyStep;
    type KernelStep = v2::KernelStep;

    fn version() -> profiler::Version {
        profiler::Version::V2
    }
}

descr_impl_event!(profiler_shim::EventDescrV3, EventMetadata::V3);

impl Version for profiler_shim::EventDescrV3 {
    type Coll = v3::Coll;
    type P2p = v3::P2p;
    type ProxyOp = v3::ProxyOp;
    type ProxyStep = v3::ProxyStep;
    type KernelStep = v3::KernelStep;

    fn version() -> profiler::Version {
        profiler::Version::V3
    }
}

descr_impl_event!(profiler_shim::EventDescrV4, EventMetadata::V4);

impl Version for profiler_shim::EventDescrV4 {
    type Coll = v4::Coll;
    type P2p = v4::P2p;
    type ProxyOp = v4::ProxyOp;
    type ProxyStep = v4::ProxyStep;
    type KernelStep = v4::KernelStep;

    fn version() -> profiler::Version {
        profiler::Version::V4
    }
}

descr_impl_event!(profiler_shim::EventDescrV6, EventMetadata::V6);

impl Version for profiler_shim::EventDescrV6 {
    type Coll = v6::Coll;
    type P2p = v6::P2p;
    type ProxyOp = v6::ProxyOp;
    type ProxyStep = v6::ProxyStep;
    type KernelStep = v6::KernelStep;

    fn version() -> profiler::Version {
        profiler::Version::V6
    }
}

#[repr(transparent)]
pub struct ProxyOpStateV1(profiler_shim::ncclProfilerEventStateArgs_v1_t);

impl ProxyOpStateV1 {
    /// # Safety
    ///
    /// input must be known to be a proxyop variant.
    pub unsafe fn cast_from_union(u: &profiler_shim::ncclProfilerEventStateArgs_v1_t) -> &Self {
        let ptr = u as *const profiler_shim::ncclProfilerEventStateArgs_v1_t;
        &*ptr.cast()
    }

    #[cfg(test)]
    pub fn new(steps: i32, trans_size: usize) -> Self {
        // SAFETY: The type itself indicates that this is the proxyOp variant of union
        unsafe {
            let mut inner: profiler_shim::ncclProfilerEventStateArgs_v1_t = std::mem::zeroed();
            inner.proxyOp.steps = steps;
            inner.proxyOp.transSize = trans_size;
            Self(inner)
        }
    }
}

impl_proxyop_state!(ProxyOpStateV1, profiler::Version::V1);

#[repr(transparent)]
pub struct ProxyOpStateV2(profiler_shim::ncclProfilerEventStateArgs_v2_t);

impl ProxyOpStateV2 {
    /// # Safety
    ///
    /// input must be known to be a proxyop variant.
    pub unsafe fn cast_from_union(u: &profiler_shim::ncclProfilerEventStateArgs_v2_t) -> &Self {
        let ptr = u as *const profiler_shim::ncclProfilerEventStateArgs_v2_t;
        &*ptr.cast()
    }
}

impl_proxyop_state!(ProxyOpStateV2, profiler::Version::V2);

#[repr(transparent)]
pub struct ProxyOpStateV3(profiler_shim::ncclProfilerEventStateArgs_v3_t);

impl ProxyOpStateV3 {
    /// # Safety
    ///
    /// input must be known to be a proxyop variant.
    pub unsafe fn cast_from_union(u: &profiler_shim::ncclProfilerEventStateArgs_v3_t) -> &Self {
        let ptr = u as *const profiler_shim::ncclProfilerEventStateArgs_v3_t;
        &*ptr.cast()
    }
}

impl_proxyop_state!(ProxyOpStateV3, profiler::Version::V3);

/// For V4 API, steps are no longer tracked via proxy op state.
/// The "ProxyStep" event is used
#[repr(transparent)]
pub struct ProxyStepStateV4(profiler_shim::ncclProfilerEventStateArgs_v4_t);

impl ProxyStepStateV4 {
    /// # Safety
    ///
    /// input must be known to be a proxyop variant.
    pub unsafe fn cast_from_union(u: &profiler_shim::ncclProfilerEventStateArgs_v4_t) -> &Self {
        let ptr = u as *const profiler_shim::ncclProfilerEventStateArgs_v4_t;
        &*ptr.cast()
    }
}

impl ProxyStepState for ProxyStepStateV4 {
    fn trans_size(&self) -> usize {
        unsafe { self.0.proxyStep.transSize }
    }
}

/// V6 state args are typedef'd to v5 (includes kernelStep.pTimer).
#[repr(transparent)]
pub struct ProxyStepStateV6(profiler_shim::ncclProfilerEventStateArgs_v6_t);

impl ProxyStepStateV6 {
    /// # Safety
    ///
    /// input must be known to be a proxystep variant.
    pub unsafe fn cast_from_union(u: &profiler_shim::ncclProfilerEventStateArgs_v6_t) -> &Self {
        let ptr = u as *const profiler_shim::ncclProfilerEventStateArgs_v6_t;
        &*ptr.cast()
    }

    #[inline(always)]
    pub fn kernel_step_ptimer(&self) -> u64 {
        unsafe { self.0.kernelStep.pTimer }
    }
}

impl ProxyStepState for ProxyStepStateV6 {
    fn trans_size(&self) -> usize {
        unsafe { self.0.proxyStep.transSize }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_fifo_wait_from_timestamps() {
        assert_eq!(derive_fifo_wait_dur_ns(0, 1_000_000), 0);
        assert_eq!(derive_fifo_wait_dur_ns(1_000_000, 1_000_000), 0);
        assert_eq!(derive_fifo_wait_dur_ns(1_000_000, 999_999), 0);
        assert_eq!(derive_fifo_wait_dur_ns(1_000_000, 1_002_000), 2_000);
        assert_eq!(
            derive_fifo_wait_dur_ns(1, 1 + (u32::MAX as u64) + 10),
            u32::MAX
        );
    }

    #[test]
    fn to_json_lossless() {
        let op = NcclOpKey::Collective(0x123, 42, NcclOpType::AllGather, algo::RING, proto::LL128);
        let json = op.to_json();
        assert_eq!(json["cat"], "collective");
        assert_eq!(json["op"], "all_gather");
        assert_eq!(json["comm"], "0x0000000000000123");
        assert_eq!(json["rank"], 42);
        assert_eq!(json["algo"], "ring");
        assert_eq!(json["proto"], "LL128");

        let op = NcclOpKey::P2pSend(0x123, 0, 42);
        let json = op.to_json();
        assert_eq!(json["cat"], "point-to-point");
        assert_eq!(json["op"], "send");
        assert_eq!(json["comm"], "0x0000000000000123");
        assert_eq!(json["local_rank"], 0);
        assert_eq!(json["peer_rank"], 42);

        let op = NcclOpKey::P2pRecv(0x123, 8, 256);
        let json = op.to_json();
        assert_eq!(json["cat"], "point-to-point");
        assert_eq!(json["op"], "recv");
        assert_eq!(json["comm"], "0x0000000000000123");
        assert_eq!(json["local_rank"], 8);
        assert_eq!(json["peer_rank"], 256);
    }

    #[test]
    fn datatype_size_map() {
        for (s, n_bytes) in DATATYPE_NAME_TO_BYTES.iter() {
            assert_eq!(
                datatype_name_to_nbytes(s),
                *n_bytes,
                "mismatch {:?}: {} vs {}",
                s,
                datatype_name_to_nbytes(s),
                *n_bytes
            );
        }
    }

    #[test]
    fn ncclop_type_map() {
        for (s, t) in NCCLOP_NAME_LOOKUP.iter() {
            assert_eq!(NcclOpType::from_c_str(s), *t);
        }
    }

    #[test]
    fn algo_name_map() {
        for (s, algo) in algo::ALGO_NAME_LOOKUP.iter() {
            assert_eq!(algo::from_name(s), *algo);
        }
    }
}
