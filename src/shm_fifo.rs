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

use crossbeam::utils::CachePadded;

use std::ffi::CString;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug)]
pub struct Shm {
    _shm_fd: libc::c_int,
    mem: *mut libc::c_void,
    _size: usize,
    mapped_size: usize,
    name: Option<CString>,
}

const NUM_COMMIT_RETRY: usize = 64;

impl Shm {
    pub fn create(name: &str, size: usize) -> std::io::Result<Self> {
        // SAFETY: calling libc FFI that are MT safe.
        unsafe {
            let name_cstr = CString::new(name).unwrap();
            let oflag = libc::O_RDWR | libc::O_CREAT | libc::O_TRUNC;
            let shm_fd = libc::shm_open(name_cstr.as_ptr(), oflag, 0o666);
            if shm_fd < 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mapped_size = size;
            if libc::ftruncate(shm_fd, mapped_size as _) < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mem = libc::mmap(
                std::ptr::null_mut(),
                mapped_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            );
            if mem == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self {
                _shm_fd: shm_fd,
                mem,
                _size: size,
                mapped_size,
                name: Some(name_cstr),
            })
        }
    }

    pub fn open(name: &str) -> std::io::Result<Self> {
        // SAFETY: calling libc FFI that are MT safe.
        unsafe {
            let name_cstr = CString::new(name).unwrap();
            let oflag = libc::O_RDWR;
            let shm_fd = libc::shm_open(name_cstr.as_ptr(), oflag, 0o666);
            if shm_fd < 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut stat = MaybeUninit::<libc::stat>::uninit();
            if libc::fstat(shm_fd, stat.as_mut_ptr()) < 0 {
                return Err(std::io::Error::last_os_error());
            }

            let stat = stat.assume_init();

            let mapped_size = stat.st_size as usize;

            let mem = libc::mmap(
                std::ptr::null_mut(),
                mapped_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                shm_fd,
                0,
            );
            if mem == libc::MAP_FAILED {
                return Err(std::io::Error::last_os_error());
            }
            Ok(Self {
                _shm_fd: shm_fd,
                mem,
                _size: mapped_size,
                mapped_size,
                name: None,
            })
        }
    }

    pub fn unlink(&mut self) {
        if let Some(name) = self.name.take() {
            // SAFETY: calling libc FFI that are MT safe.
            unsafe {
                libc::shm_unlink(name.as_ptr());
            }
        }
    }

    pub fn mem(&self) -> *mut libc::c_void {
        self.mem
    }
}

impl std::ops::Drop for Shm {
    fn drop(&mut self) {
        // SAFETY: calling libc FFI that are MT safe.
        unsafe {
            libc::munmap(self.mem, self.mapped_size);
            self.unlink();
        }
    }
}

pub mod mpsc {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    struct FifoHeader<T> {
        rd_idx: CachePadded<AtomicU64>,
        wr_head: CachePadded<AtomicU64>,
        wr_tail: CachePadded<AtomicU64>,
        n_ele: usize,
        _data: MaybeUninit<T>, // this field is to make the alignment correct
    }

    impl<T> FifoHeader<T> {
        fn try_commit_wr(&self, wr_idx: u64) -> bool {
            let mut n_retry = 0;
            while self
                .wr_head
                .compare_exchange(wr_idx, wr_idx + 1, Ordering::Release, Ordering::Relaxed)
                .is_err()
            {
                n_retry += 1;
                if n_retry >= NUM_COMMIT_RETRY {
                    return false;
                }
            }
            true
        }
    }

    #[derive(Debug)]
    pub struct Sender<T> {
        shm: Shm,
        next_wr_idx: Option<u64>,
        pending_commit: Option<u64>,
        _phantom: PhantomData<T>,
    }

    impl<T> Sender<T> {
        pub fn new(name: &str) -> std::io::Result<Self> {
            Ok(Self {
                shm: Shm::open(name)?,
                next_wr_idx: None,
                pending_commit: None,
                _phantom: PhantomData,
            })
        }

        fn get(&mut self) -> (&FifoHeader<T>, &mut [MaybeUninit<T>]) {
            // SAFETY: shm created for fifo always have a valid header
            unsafe {
                let hdr: &FifoHeader<T> = &*self.shm.mem().cast();
                let n_ele = hdr.n_ele;
                let ptr = self.shm.mem().add(std::mem::size_of::<FifoHeader<T>>());
                (hdr, std::slice::from_raw_parts_mut(ptr.cast(), n_ele))
            }
        }

        pub fn send(&mut self, data: T) -> Result<(), T> {
            if self.try_commit().is_err() {
                return Err(data);
            }
            let wr_idx = self.next_wr_idx.take();
            let (hdr, fifo) = self.get();
            let mut rd_idx = hdr.rd_idx.load(Ordering::Relaxed);
            let wr_idx = wr_idx.unwrap_or_else(|| hdr.wr_tail.fetch_add(1, Ordering::Relaxed));

            if rd_idx + hdr.n_ele as u64 <= wr_idx {
                rd_idx = hdr.rd_idx.load(Ordering::Relaxed);
            }

            // retry after loading the rd_idx
            if rd_idx + hdr.n_ele as u64 <= wr_idx {
                self.next_wr_idx = Some(wr_idx);
                return Err(data);
            }

            // SAFETY: wr_idx is pointing to uninit slot
            unsafe {
                std::ptr::write(fifo[wr_idx as usize % hdr.n_ele].as_mut_ptr(), data);
            }

            if hdr
                .wr_head
                .compare_exchange(wr_idx, wr_idx + 1, Ordering::Release, Ordering::Relaxed)
                .is_err()
            {
                self.pending_commit = Some(wr_idx);
            }
            Ok(())
        }

        pub fn try_commit(&mut self) -> Result<(), ()> {
            if let Some(wr_idx) = self.pending_commit {
                let (hdr, _) = self.get();
                if !hdr.try_commit_wr(wr_idx) {
                    return Err(());
                }
                self.pending_commit = None;
            }
            Ok(())
        }
    }

    #[derive(Debug)]
    pub struct Receiver<T> {
        shm: Shm,
        _phantom: PhantomData<T>,
    }

    impl<T> Receiver<T> {
        pub fn new(name: &str, size: usize) -> std::io::Result<Self> {
            let header_sz = std::mem::size_of::<FifoHeader<T>>();
            let arr_bytes = std::mem::size_of::<T>() * size;
            let r = Self {
                shm: Shm::create(name, header_sz + arr_bytes)?,
                _phantom: PhantomData,
            };

            // SAFETY: memory is mapped by mmap().
            unsafe {
                let hdr: *mut FifoHeader<T> = r.shm.mem().cast();
                std::ptr::write(
                    hdr,
                    FifoHeader {
                        rd_idx: CachePadded::new(AtomicU64::new(0)),
                        wr_head: CachePadded::new(AtomicU64::new(0)),
                        wr_tail: CachePadded::new(AtomicU64::new(0)),
                        n_ele: size,
                        _data: MaybeUninit::uninit(),
                    },
                );
            }

            Ok(r)
        }

        fn get(&mut self) -> (&FifoHeader<T>, &mut [MaybeUninit<T>]) {
            // SAFETY: memory already mapped.
            unsafe {
                let hdr: &FifoHeader<T> = &*self.shm.mem().cast();
                let n_ele = hdr.n_ele;
                let ptr = self.shm.mem().add(std::mem::size_of::<FifoHeader<T>>());
                (hdr, std::slice::from_raw_parts_mut(ptr.cast(), n_ele))
            }
        }

        pub fn recv(&mut self) -> Option<T> {
            let (hdr, fifo) = self.get();
            let rd_idx = hdr.rd_idx.load(Ordering::Relaxed);
            let wr_head = hdr.wr_head.load(Ordering::Acquire);
            if rd_idx >= wr_head {
                None
            } else {
                // SAFETY: when rd_idx < wr_head, it points to valid data.
                let v = unsafe { fifo[rd_idx as usize % hdr.n_ele].assume_init_read() };
                hdr.rd_idx.store(rd_idx + 1, Ordering::Release);
                Some(v)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_match() {
        let shm = Shm::create("test", 4096).unwrap();
        let shm_opened = Shm::open("test").unwrap();
        assert_eq!(shm.mapped_size, shm_opened.mapped_size);
    }

    #[test]
    fn content_shared() {
        const N: usize = 256;
        let shm = Shm::create("test-2", N * std::mem::size_of::<u32>()).unwrap();
        let ptr = shm.mem() as *mut u32;
        for i in 0..N {
            // SAFETY: ptr is pointing to valid memory of size N * sizeof(u32)
            unsafe {
                ptr.add(i).write(i as u32);
            }
        }

        let shm_opened = Shm::open("test-2").unwrap();

        // close the original
        std::mem::drop(shm);

        let ptr = shm_opened.mem() as *mut u32;
        for i in 0..N {
            // SAFETY: ptr is pointing to valid memory of size N * sizeof(u32)
            unsafe {
                assert_eq!(ptr.add(i).read(), i as u32);
            }
        }
    }

    #[test]
    fn shm_fifo_mpsc() {
        const SHM_NAME: &str = "test-mpsc";
        const N: usize = 16;
        const WRITE_BATCH: usize = N * 8;
        const N_WRITER: usize = 8;

        let barrier = std::sync::Barrier::new(N_WRITER + 1);
        let tx_thread = |idx: usize| {
            barrier.wait();
            let mut tx = mpsc::Sender::<usize>::new(SHM_NAME).unwrap();
            let mut cnt = 0;
            while cnt < WRITE_BATCH {
                if tx.send(WRITE_BATCH * idx + cnt).is_ok() {
                    cnt += 1;
                }
            }
            while tx.try_commit().is_err() {}
        };
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut rx = mpsc::Receiver::<usize>::new(SHM_NAME, N).unwrap();
                barrier.wait();
                let mut cnt = 0;
                let mut set =
                    (0..(WRITE_BATCH * N_WRITER)).collect::<std::collections::HashSet<_>>();
                while cnt < WRITE_BATCH * N_WRITER {
                    if let Some(v) = rx.recv() {
                        assert!(set.contains(&v));
                        set.remove(&v);
                        cnt += 1;
                    }
                }

                assert_eq!(set.len(), 0);

                // try another N times, we expect to get nothing
                for _ in 0..N {
                    assert!(rx.recv().is_none());
                }
            });

            for i in 0..N_WRITER {
                s.spawn(move || tx_thread(i));
            }
        });
    }
}
