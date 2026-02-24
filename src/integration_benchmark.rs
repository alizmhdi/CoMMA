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

use clap::Parser;
use nccl_profiler::profiler_shim;

use std::mem::MaybeUninit;
use std::vec::Vec;

type EventStateArg = profiler_shim::ncclProfilerEventStateArgs_v2_t;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Number of collectives
    num_coll: usize,
    proxyop_per_coll: usize,
    steps_per_proxyop: usize,

    #[arg(default_value_t = 2)]
    num_parallel_proxyop: usize,

    #[arg(default_value_t = 2)]
    num_parallel_step: usize,

    #[arg(short, long, default_value_t = 65536)]
    step_size: usize,
}

fn dummy_coll_descr() -> profiler_shim::ncclProfilerEventDescr_v2_t {
    let mut descr = MaybeUninit::uninit();
    let ptr: *mut profiler_shim::ncclProfilerEventDescr_v2_t = descr.as_mut_ptr();

    // SAFETY: ptr points to a valid memory
    // The descriptor is a plain C struct so this initialization is safe.
    unsafe {
        (*ptr).type_ = profiler_shim::ncclProfileColl as _;
        (*ptr).parentObj = std::ptr::null_mut();
        (*ptr).rank = 1;
        let coll = &mut (*ptr).__bindgen_anon_1.coll;
        coll.commHash = 0x1234;
        coll.seqNumber = 123;
        coll.func = c"AllGather".as_ptr();
        coll.algo = c"RING".as_ptr();
        coll.proto = c"SIMPLE".as_ptr();
        coll.count = 1024 * 1024 * 64;
        coll.datatype = c"ncclInt8".as_ptr();
        descr.assume_init()
    }
}

fn dummy_proxyop_descr(
    parent: *mut libc::c_void,
    pid: libc::pid_t,
) -> profiler_shim::ncclProfilerEventDescr_v2_t {
    let mut descr = MaybeUninit::uninit();
    let ptr: *mut profiler_shim::ncclProfilerEventDescr_v2_t = descr.as_mut_ptr();
    // SAFETY: ptr points to a valid memory
    // The descriptor is a plain C struct so this initialization is safe.
    unsafe {
        (*ptr).type_ = profiler_shim::ncclProfileProxyOp as _;
        (*ptr).parentObj = parent;
        (*ptr).rank = 1;
        let proxyop = &mut (*ptr).__bindgen_anon_1.proxyOp;
        proxyop.pid = pid;
        proxyop.channelId = 123;
        proxyop.peer = 8;
        proxyop.nSteps = 64;
        proxyop.chunkSize = 1 << 20;
        proxyop.isSend = 1;
        descr.assume_init()
    }
}

fn main() {
    let args = Args::parse();
    let profiler_symbols = nccl_profiler::ncclProfiler_v2;

    // SAFETY: calling the C API exposed by CoMMA
    // init is safe to call as long as both input pointers are valid
    let comm = unsafe {
        let mut comm = std::ptr::null_mut();
        let mut mask = 0;
        profiler_symbols.init.unwrap()(&raw mut comm, &raw mut mask);
        comm
    };

    let mut colls = Vec::with_capacity(args.num_coll);
    for _ in 0..args.num_coll {
        let descr = dummy_coll_descr();
        // SAFETY: calling the C API exposed by CoMMA
        // descr is a valid descriptor
        let c = unsafe {
            let mut handle = std::ptr::null_mut();
            assert_eq!(
                profiler_symbols.startEvent.unwrap()(
                    comm,
                    &raw mut handle,
                    &descr as *const _ as _
                ),
                profiler_shim::ncclResult_t_ncclSuccess
            );
            handle
        };
        colls.push(c);
    }

    for c in colls.iter() {
        // SAFETY: calling the C API exposed by CoMMA
        // handle is valid
        unsafe {
            assert_eq!(
                profiler_symbols.stopEvent.unwrap()(*c),
                profiler_shim::ncclResult_t_ncclSuccess
            );
        }
    }

    // SAFETY: getpid() is a safe C API
    let my_pid = unsafe { libc::getpid() };

    let mut proxyops = Vec::with_capacity(args.num_parallel_proxyop);

    for c in colls.iter() {
        for p_i in 0..args.proxyop_per_coll {
            // SAFETY: calling the C API exposed by CoMMA
            // descr is a valid descriptor
            let handle = unsafe {
                let descr = dummy_proxyop_descr(*c, my_pid);
                let mut handle = std::ptr::null_mut();
                assert_eq!(
                    profiler_symbols.startEvent.unwrap()(
                        comm,
                        &raw mut handle,
                        &descr as *const _ as _
                    ),
                    profiler_shim::ncclResult_t_ncclSuccess
                );
                handle
            };
            proxyops.push(handle);
            if proxyops.len() == args.num_parallel_proxyop
                || p_i + args.num_parallel_proxyop > args.proxyop_per_coll
            {
                let n_round = args.steps_per_proxyop / args.num_parallel_step;
                // SAFETY: zero initiated EventStateArg is valid
                let mut step_args: EventStateArg = unsafe { std::mem::zeroed() };
                for h in proxyops.iter() {
                    for _ in 0..args.num_parallel_step {
                        // SAFETY: calling the C API exposed by CoMMA
                        unsafe {
                            assert_eq!(
                                profiler_symbols.recordEventState.unwrap()(
                                    *h,
                                    profiler_shim::proxy_event_state::v2::SEND_REM_FIFO_WAIT,
                                    &raw mut step_args
                                ),
                                profiler_shim::ncclResult_t_ncclSuccess
                            );
                        }
                    }
                }
                for _ in 0..n_round {
                    // SAFETY: accessing union variant
                    unsafe {
                        step_args.proxyOp.transSize += args.step_size;
                    }
                    for h in proxyops.iter() {
                        for _ in 0..args.num_parallel_step {
                            // SAFETY: calling the C API exposed by CoMMA
                            // following the same convension as NCCL
                            unsafe {
                                assert_eq!(
                                    profiler_symbols.recordEventState.unwrap()(
                                        *h,
                                        profiler_shim::proxy_event_state::v1::SEND_TRANSMITTED,
                                        &raw mut step_args
                                    ),
                                    profiler_shim::ncclResult_t_ncclSuccess
                                );
                            }
                        }
                    }
                    for h in proxyops.iter() {
                        for _ in 0..args.num_parallel_step {
                            // SAFETY: calling the C API exposed by CoMMA
                            // following the same convension as NCCL
                            unsafe {
                                assert_eq!(
                                    profiler_symbols.recordEventState.unwrap()(
                                        *h,
                                        profiler_shim::proxy_event_state::v1::SEND_DONE,
                                        &raw mut step_args
                                    ),
                                    profiler_shim::ncclResult_t_ncclSuccess
                                );
                            }
                        }
                    }
                    // SAFETY: access proxyop variant of the union.
                    // for proxy step, this is what NCCL would update
                    unsafe {
                        step_args.proxyOp.steps += 1;
                    }
                }

                for h in proxyops.iter() {
                    // SAFETY: calling the C API exposed by CoMMA
                    // following the same convension as NCCL
                    unsafe {
                        assert_eq!(
                            profiler_symbols.stopEvent.unwrap()(*h),
                            profiler_shim::ncclResult_t_ncclSuccess
                        );
                    }
                }
                proxyops.clear();
            }
        }
    }
}
