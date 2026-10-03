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

//! Unix-socket RPC for mid-flight CoMMA metric gates (used by comma-monitor).
//!
//! Socket path: `{dir}/control-{pid}.sock` (same directory as `latency-{pid}.txt`).
//! Protocol: one JSON object per line.
//!
//! Requests:
//! ```json
//! {"cmd":"ping"}
//! {"cmd":"get"}
//! {"cmd":"set","track_kernel_step":false,"track_steps":true}
//! ```
//!
//! Responses:
//! ```json
//! {"ok":true,"pid":1234,"gates":{...}}
//! {"ok":false,"error":"..."}
//! ```

use crate::profiler::Profiler;
use crate::runtime_gates::{GateSnapshot, GateUpdate};

use log::{info, warn};
use serde::{Deserialize, Serialize};

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct Request {
    cmd: String,
    #[serde(flatten)]
    update: GateUpdate,
}

#[derive(Debug, Serialize)]
struct Response<'a> {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pid: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    changed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gates: Option<&'a GateSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Resolve control socket path from config / latency file / env.
pub fn resolve_sock_path(
    config_sock: Option<&str>,
    latency_file: Option<&str>,
    pid: libc::pid_t,
) -> Option<PathBuf> {
    let template = config_sock
        .map(str::to_string)
        .or_else(|| std::env::var("NCCL_PROFILER_CONTROL_SOCK").ok())
        .or_else(|| {
            latency_file.and_then(|lf| {
                let p = Path::new(lf);
                let dir = p.parent().unwrap_or_else(|| Path::new("."));
                Some(dir.join("control-%p.sock").to_string_lossy().into_owned())
            })
        })?;
    Some(PathBuf::from(template.replace("%p", &pid.to_string())))
}

pub fn spawn_control_server(profiler: &'static Profiler) {
    let Some(path) = resolve_sock_path(
        profiler.config.control_sock.as_deref(),
        profiler.config.latency_file.as_deref(),
        profiler.pid,
    ) else {
        info!("CoMMA control RPC disabled (no control sock / latency_file path)");
        return;
    };
    if let Err(e) = std::thread::Builder::new()
        .name("comma-control-rpc".into())
        .spawn(move || run_server(path, profiler))
    {
        warn!("failed to spawn CoMMA control RPC thread: {e}");
    }
}

fn run_server(path: PathBuf, profiler: &'static Profiler) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
        // World-writable dir so the host-side monitor (non-root) can connect
        // when CoMMA runs as root inside the training container.
        let _ = fs::set_permissions(parent, fs::Permissions::from_mode(0o777));
    }
    let _ = fs::remove_file(&path);
    // Linux sockaddr_un.sun_path is ~108 bytes; refuse long paths early.
    let path_bytes = path.as_os_str().as_encoded_bytes();
    if path_bytes.len() >= 108 {
        warn!(
            "CoMMA control RPC path too long ({} >= 108): {}",
            path_bytes.len(),
            path.display()
        );
        return;
    }
    let listener = match UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            warn!("CoMMA control RPC bind {}: {e}", path.display());
            return;
        }
    };
    // Connect requires write on the socket inode; container often runs as root.
    let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o666));
    // Make accept interruptible-ish for process teardown.
    let _ = listener.set_nonblocking(false);
    info!(
        "CoMMA control RPC listening on {} (pid={})",
        path.display(),
        profiler.pid
    );
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                if let Err(e) = handle_client(stream, profiler) {
                    warn!("CoMMA control RPC client error: {e}");
                }
            }
            Err(e) => {
                warn!("CoMMA control RPC accept error: {e}");
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn handle_client(stream: UnixStream, profiler: &'static Profiler) -> std::io::Result<()> {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let resp = dispatch_line(line.trim(), profiler);
        writeln!(writer, "{resp}")?;
        writer.flush()?;
    }
    Ok(())
}

fn dispatch_line(line: &str, profiler: &'static Profiler) -> String {
    if line.is_empty() {
        return err_json("empty request");
    }
    let req: Request = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => return err_json(&format!("invalid json: {e}")),
    };
    match req.cmd.as_str() {
        "ping" => {
            let snap = profiler.gates.snapshot();
            ok_json(profiler.pid, None, Some(&snap))
        }
        "get" => {
            let snap = profiler.gates.snapshot();
            ok_json(profiler.pid, None, Some(&snap))
        }
        "set" => {
            let changed = profiler.gates.apply_update(&req.update);
            let snap = profiler.gates.snapshot();
            ok_json(profiler.pid, Some(changed), Some(&snap))
        }
        other => err_json(&format!("unknown cmd {other:?} (want ping|get|set)")),
    }
}

fn ok_json(pid: libc::pid_t, changed: Option<bool>, gates: Option<&GateSnapshot>) -> String {
    serde_json::to_string(&Response {
        ok: true,
        pid: Some(pid as i32),
        changed,
        gates,
        error: None,
    })
    .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialize\"}".into())
}

fn err_json(msg: &str) -> String {
    serde_json::to_string(&Response {
        ok: false,
        pid: None,
        changed: None,
        gates: None,
        error: Some(msg.to_string()),
    })
    .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"serialize\"}".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_from_latency_template() {
        let p = resolve_sock_path(None, Some("/tmp/comma/latency-%p.txt"), 42).unwrap();
        assert_eq!(p, PathBuf::from("/tmp/comma/control-42.sock"));
    }
}
