//! CPU placement of CoMMA's own threads.
//!
//! `COMMA_HELPER_CPUS` (e.g. `26,27,54,55` or `26-27,54-55`) confines the
//! daemon's polling thread, the export runtime's threads and the control RPC
//! thread to those CPUs. They wake thousands of times a second; left to the
//! scheduler, each wakeup can land on (or next to) a core running the
//! training's launch threads, which then issue their kernels and collectives
//! more slowly. The training's own threads are not touched. Unset: no change.

use std::sync::OnceLock;

/// Parsed `COMMA_HELPER_CPUS`; None when unset, empty or invalid.
fn helper_cpus() -> Option<&'static [usize]> {
    static CPUS: OnceLock<Option<Vec<usize>>> = OnceLock::new();
    CPUS.get_or_init(|| parse_cpu_list(&std::env::var("COMMA_HELPER_CPUS").ok()?))
        .as_deref()
}

/// `a,b,c-d` -> sorted CPU ids; None on any malformed entry.
fn parse_cpu_list(s: &str) -> Option<Vec<usize>> {
    let mut out = Vec::new();
    for part in s.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b) = (
                    a.trim().parse::<usize>().ok()?,
                    b.trim().parse::<usize>().ok()?,
                );
                if a > b {
                    return None;
                }
                out.extend(a..=b);
            }
            None => out.push(part.parse::<usize>().ok()?),
        }
    }
    out.sort_unstable();
    out.dedup();
    (!out.is_empty()).then_some(out)
}

/// Pin the calling thread to `COMMA_HELPER_CPUS`, if set. Failures (a CPU
/// outside the process's cpuset, for one) leave the thread where it was.
pub fn pin_current_thread() {
    let Some(cpus) = helper_cpus() else {
        return;
    };
    // SAFETY: cpu_set_t is plain data; CPU_ZERO/CPU_SET only write into it,
    // and sched_setaffinity(0, ...) applies to the calling thread.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            if cpu < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(cpu, &mut set);
            }
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            log::warn!(
                "COMMA_HELPER_CPUS={cpus:?}: sched_setaffinity failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_cpu_list;

    #[test]
    fn parses_lists_and_ranges() {
        assert_eq!(parse_cpu_list("26,27,54,55"), Some(vec![26, 27, 54, 55]));
        assert_eq!(parse_cpu_list("54-55, 26-27"), Some(vec![26, 27, 54, 55]));
        assert_eq!(parse_cpu_list("3,3"), Some(vec![3]));
        assert_eq!(parse_cpu_list(""), None);
        assert_eq!(parse_cpu_list("2-1"), None);
        assert_eq!(parse_cpu_list("a"), None);
    }
}
