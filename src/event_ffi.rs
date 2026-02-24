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

use crate::event::{Event, Group, ProxyStep};
use crate::profiler::{KernelCh, ProxyOpLocalData};
use crate::slab;

use static_assertions::const_assert;

pub type Handle = *mut libc::c_void;

pub const N_TYPE_BITS: u32 = 4;
pub const HANDLE_TYPE_MASK: usize = (1 << N_TYPE_BITS) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Group,
    NcclOpLite, // NCCL op without step tracking
    NcclOp,
    ProxyOpLite,
    ProxyOp,
    Dummy,
    SmallNcclOp,
    ProxyStep,
    KernelCh,
}

impl Type {
    fn to_bits(self) -> usize {
        match self {
            Type::Group => 0b00,
            Type::NcclOpLite => 0b01,
            Type::NcclOp => 0b10,
            Type::ProxyOpLite => 0b11,
            Type::ProxyOp => 0b100,
            Type::Dummy => 0b101,
            Type::SmallNcclOp => 0b110,
            Type::ProxyStep => 0b111,
            Type::KernelCh => 0b1000,
        }
    }

    fn from_bits(bits: usize) -> Self {
        match bits {
            0b00 => Type::Group,
            0b01 => Type::NcclOpLite,
            0b10 => Type::NcclOp,
            0b11 => Type::ProxyOpLite,
            0b100 => Type::ProxyOp,
            0b101 => Type::Dummy,
            0b110 => Type::SmallNcclOp,
            0b111 => Type::ProxyStep,
            0b1000 => Type::KernelCh,
            _ => panic!("unknown bit pattern"),
        }
    }
}

fn handle(handle: usize, t: Type) -> Handle {
    (handle | t.to_bits()) as _
}

pub fn get_handle_type(handle: Handle) -> Type {
    let val = handle as usize;
    let type_bits = val & HANDLE_TYPE_MASK;
    Type::from_bits(type_bits)
}

fn get_handle_inner(handle: Handle) -> usize {
    (handle as usize) & !HANDLE_TYPE_MASK
}

fn to_ncclop_handle(handle: Handle) -> usize {
    get_handle_inner(handle) >> N_TYPE_BITS
}

pub trait AsFFI: Sized {
    fn into_ffi(self) -> Handle;

    /// # Safety
    ///
    /// `handle` parameter must be a valid handle returned by calling `into_ffi()`
    unsafe fn from_ffi(handle: Handle) -> Option<Self>;
}

const_assert!(std::mem::align_of::<Group>() >= (1 << N_TYPE_BITS));
const_assert!(std::mem::align_of::<ProxyStep>() >= (1 << N_TYPE_BITS));
const_assert!(std::mem::align_of::<KernelCh>() >= (1 << N_TYPE_BITS));
const_assert!(std::mem::align_of::<ProxyOpLocalData>() >= (1 << N_TYPE_BITS));

impl AsFFI for Event {
    fn into_ffi(self) -> Handle {
        match self {
            Event::Group(group) => {
                let ptr = Box::into_raw(group);
                handle(ptr as _, Type::Group)
            }
            /*
            Event::NcclOpLite(op) => {
                let slab_handle = slab::Entry::into_handle(op) << N_TYPE_BITS;
                handle(slab_handle as _, Type::NcclOpLite)
            }
            Event::NcclOp(op) => {
                let slab_handle = slab::Entry::into_handle(op) << N_TYPE_BITS;
                handle(slab_handle as _, Type::NcclOp)
            }
            */
            Event::NcclOpLite(id) => handle(id << N_TYPE_BITS, Type::NcclOpLite),
            Event::NcclOp(id) => handle(id << N_TYPE_BITS, Type::NcclOp),
            Event::ProxyOpLite(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::ProxyOpLite)
            }
            Event::KernelCh(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::KernelCh)
            }
            /*
            Event::ProxyOp(id) => {
                let v = (id as usize) << N_TYPE_BITS;
                handle(v, Type::ProxyOp)
            }
            */
            Event::ProxyOp(op) => {
                let ptr = slab::AllocatedNode::into_raw(op);
                handle(ptr as _, Type::ProxyOp)
            }
            Event::Dummy(val) => {
                let v = val << N_TYPE_BITS;
                handle(v, Type::Dummy)
            }
            Event::SmallNcclOp(val) => {
                let v = val << N_TYPE_BITS;
                handle(v, Type::SmallNcclOp)
            }
            Event::ProxyStep(step) => {
                let ptr = slab::AllocatedNode::into_raw(step);
                handle(ptr as _, Type::ProxyStep)
            }
        }
    }

    unsafe fn from_ffi(handle: Handle) -> Option<Self> {
        let handle_type = get_handle_type(handle);
        match handle_type {
            Type::Group => {
                let ptr = get_handle_inner(handle) as _;
                Some(Event::Group(Box::from_raw(ptr)))
            }
            /*
            Type::NcclOpLite => {
                let handle = to_ncclop_handle(handle);
                slab::Entry::from_handle(handle).map(Event::NcclOp)
            }
            Type::NcclOp => {
                let handle = to_ncclop_handle(handle);
                slab::Entry::from_handle(handle).map(Event::NcclOp)
            }
            */
            Type::NcclOpLite => {
                let handle = to_ncclop_handle(handle);
                Some(Event::NcclOpLite(handle))
            }
            Type::NcclOp => {
                let handle = to_ncclop_handle(handle);
                Some(Event::NcclOp(handle))
            }
            Type::ProxyOpLite => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyOpLite(proxyop))
            }
            Type::KernelCh => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::KernelCh(proxyop))
            }
            /*
            Type::ProxyOp => {
                let handle = get_handle_inner(handle);
                let id = handle >> N_TYPE_BITS;
                Some(Event::ProxyOp(id as u32))
            }
            */
            Type::ProxyOp => {
                let handle = get_handle_inner(handle) as _;
                let proxyop = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyOp(proxyop))
            }
            Type::Dummy => {
                let handle = get_handle_inner(handle);
                let v = handle >> N_TYPE_BITS;
                Some(Event::Dummy(v))
            }
            Type::SmallNcclOp => {
                let handle = get_handle_inner(handle);
                let v = handle >> N_TYPE_BITS;
                Some(Event::SmallNcclOp(v))
            }
            Type::ProxyStep => {
                let handle = get_handle_inner(handle) as _;
                let proxystep = slab::AllocatedNode::from_raw(handle);
                Some(Event::ProxyStep(proxystep))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ProxyParent {
    NcclOp(/* slab handle */ usize),
    Dummy(/* comm_hash */ usize),
    Null,
}

impl ProxyParent {
    pub fn from_ffi(handle: Handle) -> Self {
        if !handle.is_null() {
            if get_handle_type(handle) != Type::Dummy {
                Self::NcclOp(to_ncclop_handle(handle))
            } else {
                Self::Dummy(get_handle_inner(handle))
            }
        } else {
            Self::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::event::ProfilerEvent as _;
    use crate::profiler;
    use crate::profiler_shim::tests::{dummy_group_descr, dummy_proxyop_descr};

    use std::time::Instant;

    #[test]
    fn type_bits_inv() {
        let types = [Type::Group, Type::NcclOp, Type::ProxyOp, Type::Dummy];
        for t in types {
            assert_eq!(Type::from_bits(t.to_bits()), t);
        }
    }

    #[test]
    fn ffi_handle_inv() {
        // note: the list of handles should have last N_TYPE_BITS zeroed
        let handles = [0, 48, 1024, 0xabcd1230, 0xfffffff0];
        let types = [Type::Group, Type::NcclOp, Type::ProxyOp, Type::Dummy];
        for h in handles {
            for t in types {
                let ffi = handle(h, t);
                assert_eq!(get_handle_inner(ffi), h);
                assert_eq!(get_handle_type(ffi), t);
            }
        }
    }

    #[test]
    fn group_event_mock() {
        let t0 = Instant::now();
        let group_descr = dummy_group_descr();
        let event = Event::new_group(&group_descr, t0);
        let handle = Event::into_ffi(event);

        // SAFETY: handle is valid return value of into_ffi()
        if let Some(Event::Group(mut group)) = unsafe { Event::from_ffi(handle) } {
            assert_eq!(
                group.basic_info().rank(),
                group_descr.as_ref().rank as usize
            );
            assert_eq!(group.basic_info().start_time(), t0);

            let t1 = Instant::now();
            group.basic_info_mut().update_end_time(t1);
            let record = group.trace_record(|t| (t - t0).as_micros() as _);
            assert_eq!(record["cat"], "GROUP");
            assert_eq!(record["dur"], (t1 - t0).as_micros() as u64);
        } else {
            panic!("failed to get group from ffi");
        }
    }

    #[test]
    fn proxyop_event_mock() {
        use crate::nccl_metadata::Version as _;

        let proxyop_descr = dummy_proxyop_descr();
        let event = Event::ProxyOp(slab::AllocatedNode::new(profiler::ProxyOpLocalData::new(
            42,
            // SAFETY: proxyop_descr is a proxyop descriptor
            unsafe { proxyop_descr.cast_to_proxyop() },
            false,
            false,
        )));
        let handle = Event::into_ffi(event);
        // SAFETY: handle is valid return value of into_ffi()
        if let Some(Event::ProxyOp(proxydata)) = unsafe { Event::from_ffi(handle) } {
            assert_eq!(proxydata.info.id, 42);
            assert!(proxydata.info.is_send);
        } else {
            panic!("failed get proxyop from ffi");
        }
    }
}
