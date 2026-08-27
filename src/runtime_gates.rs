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

//! Runtime start-gate for CoMMA logging metrics.
//!
//! Toggle bits without touching in-flight events:
//! - Update atomics that `start_event` consults (refuse new starts only).
//! - Rewrite NCCL's process-global `ncclProfilerEventMask` so *new* tasks
//!   pick up enablement at enqueue.
//!
//! Live control is via Unix-socket RPC (`control_rpc`); see monitor `control` module.

use crate::config::Config;
use crate::profiler::Version;
use crate::profiler_shim;

use log::{info, warn};

use serde::{Deserialize, Serialize};

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, Ordering};

/// Desired metric enablement + retained pointer to NCCL's activation mask.
#[derive(Debug)]
pub struct RuntimeGates {
    track_group: AtomicBool,
    track_ncclop: AtomicBool,
    track_proxyop: AtomicBool,
    track_interprocess_proxyop: AtomicBool,
    track_steps: AtomicBool,
    track_recv_steps: AtomicBool,
    track_step_fifo_wait: AtomicBool,
    aggregate_steps: AtomicBool,
    track_kernel_ch: AtomicBool,
    track_kernel_step: AtomicBool,
    track_recv_kernel_step: AtomicBool,
    api_v6: AtomicBool,
    /// Points at NCCL `ncclProfilerEventMask` (same address every `init`).
    event_mask: AtomicPtr<i32>,
}

impl RuntimeGates {
    pub fn new(config: &Config, version: Version) -> Self {
        Self {
            track_group: AtomicBool::new(config.track_group),
            track_ncclop: AtomicBool::new(config.track_ncclop),
            track_proxyop: AtomicBool::new(config.track_proxyop),
            track_interprocess_proxyop: AtomicBool::new(config.track_interprocess_proxyop),
            track_steps: AtomicBool::new(config.track_steps),
            track_recv_steps: AtomicBool::new(config.track_recv_steps),
            track_step_fifo_wait: AtomicBool::new(config.track_step_fifo_wait),
            aggregate_steps: AtomicBool::new(config.aggregate_steps),
            track_kernel_ch: AtomicBool::new(config.track_kernel_ch),
            track_kernel_step: AtomicBool::new(config.track_kernel_step),
            track_recv_kernel_step: AtomicBool::new(config.track_recv_kernel_step),
            api_v6: AtomicBool::new(matches!(version, Version::V6)),
            event_mask: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub fn track_group(&self) -> bool {
        self.track_group.load(Ordering::Acquire)
    }
    pub fn track_ncclop(&self) -> bool {
        self.track_ncclop.load(Ordering::Acquire)
    }
    pub fn track_proxyop(&self) -> bool {
        self.track_proxyop.load(Ordering::Acquire)
    }
    pub fn track_interprocess_proxyop(&self) -> bool {
        self.track_interprocess_proxyop.load(Ordering::Acquire)
    }
    pub fn track_steps(&self) -> bool {
        self.track_steps.load(Ordering::Acquire)
    }
    pub fn track_recv_steps(&self) -> bool {
        self.track_recv_steps.load(Ordering::Acquire)
    }
    pub fn track_step_fifo_wait(&self) -> bool {
        self.track_step_fifo_wait.load(Ordering::Acquire)
    }
    pub fn aggregate_steps(&self) -> bool {
        self.aggregate_steps.load(Ordering::Acquire)
    }
    pub fn track_kernel_ch(&self) -> bool {
        self.track_kernel_ch.load(Ordering::Acquire)
    }
    pub fn track_kernel_step(&self) -> bool {
        self.track_kernel_step.load(Ordering::Acquire)
    }
    pub fn track_recv_kernel_step(&self) -> bool {
        self.track_recv_kernel_step.load(Ordering::Acquire)
    }

    pub fn proxy_step_enabled(&self) -> bool {
        self.track_steps() || self.aggregate_steps()
    }

    pub fn snapshot(&self) -> GateSnapshot {
        GateSnapshot {
            track_group: self.track_group(),
            track_ncclop: self.track_ncclop(),
            track_proxyop: self.track_proxyop(),
            track_interprocess_proxyop: self.track_interprocess_proxyop(),
            track_steps: self.track_steps(),
            track_recv_steps: self.track_recv_steps(),
            track_step_fifo_wait: self.track_step_fifo_wait(),
            aggregate_steps: self.aggregate_steps(),
            track_kernel_ch: self.track_kernel_ch(),
            track_kernel_step: self.track_kernel_step(),
            track_recv_kernel_step: self.track_recv_kernel_step(),
            mask: self.compute_mask(),
        }
    }

    /// Remember NCCL's global mask pointer (idempotent).
    pub fn register_mask_ptr(&self, ptr: *mut i32) {
        if ptr.is_null() {
            return;
        }
        let prev = self.event_mask.swap(ptr, Ordering::AcqRel);
        if !prev.is_null() && prev != ptr {
            warn!("CoMMA activation mask pointer changed unexpectedly");
        }
    }

    /// Compute mask from current gates and publish to NCCL + return value for `init`.
    pub fn publish_mask(&self) -> i32 {
        let mask = self.compute_mask();
        self.store_mask(mask);
        mask
    }

    pub fn compute_mask(&self) -> i32 {
        let mut mask: i32 = 0;
        mask |= profiler_shim::ncclProfileGroup as i32;
        mask |= profiler_shim::ncclProfileColl as i32;
        mask |= profiler_shim::ncclProfileP2p as i32;
        // ProxyOp bit stays on when proxyop tracking is enabled (MoE/P2P paths).
        if self.track_proxyop() {
            mask |= profiler_shim::ncclProfileProxyOp as i32;
        } else {
            // Keep ProxyOp bit for completion even if deep tracking is off —
            // historical CoMMA always set this bit. Prefer always-on ProxyOp mask.
            mask |= profiler_shim::ncclProfileProxyOp as i32;
        }
        if self.track_kernel_ch() {
            mask |= profiler_shim::ncclProfileKernelCh as i32;
        }
        if self.track_kernel_step() && self.api_v6.load(Ordering::Acquire) {
            mask |= profiler_shim::ncclProfileKernelStep as i32;
        }
        if self.proxy_step_enabled() {
            mask |= profiler_shim::ncclProfileProxyStep as i32;
        }
        mask
    }

    fn store_mask(&self, mask: i32) {
        let ptr = self.event_mask.load(Ordering::Acquire);
        if ptr.is_null() {
            return;
        }
        // SAFETY: ptr is &ncclProfilerEventMask for the process lifetime.
        unsafe {
            (*(ptr as *const AtomicI32)).store(mask, Ordering::Release);
        }
    }

    /// Apply a parsed update; republish mask. Returns true if anything changed.
    pub fn apply_update(&self, update: &GateUpdate) -> bool {
        let mut changed = false;
        changed |= swap_bool(&self.track_group, update.track_group);
        changed |= swap_bool(&self.track_ncclop, update.track_ncclop);
        changed |= swap_bool(&self.track_proxyop, update.track_proxyop);
        changed |= swap_bool(
            &self.track_interprocess_proxyop,
            update.track_interprocess_proxyop,
        );
        changed |= swap_bool(&self.track_steps, update.track_steps);
        changed |= swap_bool(&self.track_recv_steps, update.track_recv_steps);
        changed |= swap_bool(&self.track_step_fifo_wait, update.track_step_fifo_wait);
        changed |= swap_bool(&self.aggregate_steps, update.aggregate_steps);
        changed |= swap_bool(&self.track_kernel_ch, update.track_kernel_ch);
        changed |= swap_bool(&self.track_kernel_step, update.track_kernel_step);
        changed |= swap_bool(&self.track_recv_kernel_step, update.track_recv_kernel_step);
        if changed {
            let mask = self.publish_mask();
            info!(
                "CoMMA runtime gates updated: {:?} mask=0x{:x}",
                self.snapshot(),
                mask
            );
        }
        changed
    }
}

fn swap_bool(slot: &AtomicBool, next: Option<bool>) -> bool {
    let Some(v) = next else {
        return false;
    };
    slot.swap(v, Ordering::AcqRel) != v
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_group: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_ncclop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_proxyop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_interprocess_proxyop: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_steps: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_recv_steps: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_step_fifo_wait: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aggregate_steps: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_kernel_ch: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_kernel_step: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub track_recv_kernel_step: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateSnapshot {
    pub track_group: bool,
    pub track_ncclop: bool,
    pub track_proxyop: bool,
    pub track_interprocess_proxyop: bool,
    pub track_steps: bool,
    pub track_recv_steps: bool,
    pub track_step_fifo_wait: bool,
    pub aggregate_steps: bool,
    pub track_kernel_ch: bool,
    pub track_kernel_step: bool,
    pub track_recv_kernel_step: bool,
    pub mask: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_update_json_roundtrip() {
        let u = GateUpdate {
            track_kernel_step: Some(false),
            track_recv_kernel_step: Some(true),
            track_steps: Some(true),
            ..Default::default()
        };
        let s = serde_json::to_string(&u).unwrap();
        let back: GateUpdate = serde_json::from_str(&s).unwrap();
        assert_eq!(back, u);
        assert!(s.contains("track_recv_kernel_step"));
    }
}
