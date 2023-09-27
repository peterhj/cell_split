use crate::{Header, Row, HashVal, HashFun, HashHexVal, AppendState};

use blake2::{Blake2b};
use rustc_serialize::*;
use rustc_serialize::json::{JsonEncoder};

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write, Seek, BufReader, Cursor, SeekFrom};
use std::mem::{swap};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::slice::{from_raw_parts};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::{SyncSender, Receiver, sync_channel, channel};
use std::thread::{JoinHandle, Thread, park, spawn};

#[derive(Clone)]
enum SwapCtl2Thread {
  Shutdown,
  Sync,
  PutHeader(Arc<Mutex<AppendState>>, u16, u64, String, Header),
  PutRowUnsafe(Arc<Mutex<AppendState>>, u16, u32, Row, usize, usize),
}

#[derive(Clone)]
enum SwapThread2Thread {
  Shutdown,
  Sync,
  PutHeader(Arc<Mutex<AppendState>>, u16, u64, String, Header),
  PutRowUnsafe(Arc<Mutex<AppendState>>, u16, u32, Row, usize, usize),
}

#[derive(Clone, Copy)]
enum SwapThread2Ctl {
  // TODO
  Ready(usize),
  //ReadyIO(usize),
}

pub struct SwapThreadPool {
  ctl2th:   Vec<(SyncSender<SwapCtl2Thread>, JoinHandle<()>, JoinHandle<()>, )>,
  th2ctl:   Receiver<SwapThread2Ctl>,
}

impl Drop for SwapThreadPool {
  fn drop(&mut self) {
    for &(ref ctl2th, ref h_pre, _) in self.ctl2th.iter() {
      ctl2th.send(SwapCtl2Thread::Shutdown).unwrap();
      //h_pre.thread().unpark();
    }
    for (_, h_pre, h_post) in self.ctl2th.drain(..) {
      h_pre.join().unwrap();
      h_post.join().unwrap();
    }
  }
}

impl SwapThreadPool {
  pub(crate) fn new(thct: usize) -> SwapThreadPool {
    let mut ctl2th = Vec::with_capacity(thct as _);
    let (th2ctl_tx, th2ctl) = sync_channel(2 * thct as usize);
    for rank in 0 .. thct {
      let (ctl2th_tx, ctl2th_rx) = sync_channel(1);
      let (th2th_tx, th2th_rx) = channel();
      let th2th_tx2 = th2th_tx.clone();
      let th2ctl_tx = th2ctl_tx.clone();
      let h_post = spawn(move || {
        let mut hash_buf = BTreeMap::new();
        let mut hash_fin = BTreeSet::new();
        //let mut headers: Vec<(Arc<Mutex<AppendState>>, u16, u64, String, Header)> = Vec::new();
        //let mut next_headers = Vec::new();
        'recv: loop {
          //park();
          /*if headers.len() > 0 {
            'headers: loop {
              match headers.pop() {
                None => break,
                Some((state, hctr, hoff, htmp, mut header)) => {
              let mut htmp2 = String::new();
              {
                let mut enc = JsonEncoder::new(&mut htmp2);
                header.split.encode(&mut enc).unwrap();
              }
              htmp2.push('\n');
              for (i, row) in header.rows.iter_mut().enumerate() {
                if !hash_fin.contains(&(hctr, i as u32)) {
                match hash_buf.remove(&(hctr, i as u32)) {
                  None => {
                    println!("DEBUG:  SwapThreadPool: rank={} hctr={} hrow={} missing hash (retry)", rank, hctr, i);
                    next_headers.push((state, hctr, hoff, htmp, header));
                    continue 'headers;
                  }
                  Some(h) => {
                    row.hash = h;
                    hash_fin.insert((hctr, i as u32));
                  }
                }
                }
                {
                  let mut enc = JsonEncoder::new(&mut htmp2);
                  row.encode(&mut enc).unwrap();
                }
                htmp2.push('\n');
              }
              if let Some(next) = header.next.as_ref() {
                {
                  let mut enc = JsonEncoder::new(&mut htmp2);
                  next.encode(&mut enc).unwrap();
                }
                htmp2.push('\n');
              }
              assert_eq!(htmp.len(), htmp2.len());
              println!("DEBUG:  SwapThreadPool: rank={} hctr={} hoff={} put header (retry)...", rank, hctr, hoff);
              {
                let mut state = state.lock().unwrap();
                state.file.seek(SeekFrom::Start(hoff)).unwrap();
                state.file.write_all(htmp2.as_bytes()).unwrap();
              }
                }
              }
            }
            swap(&mut headers, &mut next_headers);
          }*/
          match th2th_rx.recv() {
            Ok(SwapThread2Thread::Shutdown) => {
              break;
            }
            Ok(SwapThread2Thread::Sync) => {
              th2ctl_tx.send(SwapThread2Ctl::Ready(rank)).unwrap();
            }
            Ok(SwapThread2Thread::PutHeader(state, hctr, hoff, htmp, mut header)) => {
              let mut htmp2 = String::new();
              if let Some(split) = header.split.as_ref() {
                {
                  let mut enc = JsonEncoder::new(&mut htmp2);
                  split.encode(&mut enc).unwrap();
                }
                htmp2.push('\n');
              }
              for (i, row) in header.rows.iter_mut().enumerate() {
                //assert!(!hash_fin.contains(&(hctr, i as u32)));
                if !hash_fin.contains(&(hctr, i as u32)) {
                match hash_buf.remove(&(hctr, i as u32)) {
                  None => {
                    println!("DEBUG:  SwapThreadPool: rank={} hctr={} hrow={} missing hash", rank, hctr, i);
                    //headers.push((state, hctr, hoff, htmp, header));
                    th2th_tx2.send(SwapThread2Thread::PutHeader(state, hctr, hoff, htmp, header)).unwrap();
                    continue 'recv;
                  }
                  Some(h) => {
                    row.hash = h;
                    hash_fin.insert((hctr, i as u32));
                  }
                }
                }
                {
                  let mut enc = JsonEncoder::new(&mut htmp2);
                  row.encode(&mut enc).unwrap();
                }
                htmp2.push('\n');
              }
              if let Some(next) = header.next.as_ref() {
                {
                  let mut enc = JsonEncoder::new(&mut htmp2);
                  next.encode(&mut enc).unwrap();
                }
                htmp2.push('\n');
              }
              assert_eq!(htmp.len(), htmp2.len());
              //println!("DEBUG:  SwapThreadPool: rank={} hctr={} hoff={} put header...", rank, hctr, hoff);
              {
                let mut state = state.lock().unwrap();
                state.file.seek(SeekFrom::Start(hoff)).unwrap();
                state.file.write_all(htmp2.as_bytes()).unwrap();
              }
            }
            Ok(SwapThread2Thread::PutRowUnsafe(state, hctr, hrow, row, data_ptr, data_sz)) => {
              //println!("DEBUG:  SwapThreadPool: rank={} hctr={} hrow={} put row...", rank, hctr, hrow);
              let Row{off, hash, ..} = row;
              assert!(hash_buf.insert((hctr, hrow), hash).is_none());
              {
                let mut state = state.lock().unwrap();
                let data = unsafe { from_raw_parts(data_ptr as *const u8, data_sz) };
                state.file.seek(SeekFrom::Start(off)).unwrap();
                state.file.write_all(data).unwrap();
              }
            }
            _ => break
          }
        }
      });
      let post_th = h_post.thread().clone();
      let h_pre = spawn(move || {
        loop {
          //park();
          match ctl2th_rx.recv() {
            Ok(SwapCtl2Thread::Shutdown) => {
              th2th_tx.send(SwapThread2Thread::Shutdown).unwrap();
              break;
            }
            Ok(SwapCtl2Thread::Sync) => {
              th2th_tx.send(SwapThread2Thread::Sync).unwrap();
            }
            Ok(SwapCtl2Thread::PutHeader(state, hctr, hoff, htmp, header)) => {
              th2th_tx.send(SwapThread2Thread::PutHeader(state, hctr, hoff, htmp, header)).unwrap();
            }
            Ok(SwapCtl2Thread::PutRowUnsafe(state, hctr, hrow, mut row, data_ptr, data_sz)) => {
              for hval in row.hash.iter_mut() {
                match hval.fun {
                  HashFun::Blake2b => {
                    let mut hasher = Blake2b::default();
                    {
                      let data = unsafe { from_raw_parts(data_ptr as *const u8, data_sz) };
                      hasher.update(data);
                    }
                    assert_eq!(hval.val.buf.len(), 64);
                    hval.val.buf.clear();
                    hval.val.buf.extend_from_slice(hasher.finalize().as_bytes());
                    assert_eq!(hval.val.buf.len(), 64);
                  }
                  _ => unimplemented!()
                }
              }
              th2th_tx.send(SwapThread2Thread::PutRowUnsafe(state, hctr, hrow, row, data_ptr, data_sz)).unwrap();
            }
            _ => break
          }
        }
      });
      ctl2th.push((ctl2th_tx, h_pre, h_post));
    }
    SwapThreadPool{
      ctl2th,
      th2ctl,
    }
  }

  #[inline]
  fn num_threads(&self) -> usize {
    self.ctl2th.len()
  }

  pub(crate) fn shutdown(&self, rank: usize) {
    let &(ref ctl2th, ref h, _) = &self.ctl2th[rank];
    ctl2th.send(SwapCtl2Thread::Shutdown).unwrap();
  }

  pub(crate) fn sync(&self, rank: usize) {
    let &(ref ctl2th, ref h, _) = &self.ctl2th[rank];
    ctl2th.send(SwapCtl2Thread::Sync).unwrap();
  }

  pub(crate) fn wait(&self) -> usize {
    match self.th2ctl.recv() {
      Ok(SwapThread2Ctl::Ready(rank)) => {
        return rank;
      }
      _ => panic!("bug")
    }
  }

  pub(crate) fn put_header(&self, rank: usize, state: Arc<Mutex<AppendState>>, hctr: u16, hoff: u64, htmp: String, header: Header) {
    let &(ref ctl2th, ref h, _) = &self.ctl2th[rank];
    ctl2th.send(SwapCtl2Thread::PutHeader(state, hctr, hoff, htmp, header)).unwrap();
  }

  pub(crate) fn put_row_unsafe(&self, rank: usize, state: Arc<Mutex<AppendState>>, hctr: u16, hrow: u32, row: Row, data_ptr: *const u8, data_sz: usize) {
    {
      let &(ref ctl2th, ref h, _) = &self.ctl2th[rank];
      ctl2th.send(SwapCtl2Thread::PutRowUnsafe(state, hctr, hrow, row, data_ptr as usize, data_sz)).unwrap();
      //h.thread().unpark();
    }
  }
}

pub struct DfParse {
  pub source: PathBuf,
  pub target: PathBuf,
}

impl DfParse {
  pub fn open<P: AsRef<Path>>(path: P) -> Result<DfParse, ()> {
    let out = Command::new("df")
        .arg("--output=source,target")
        .arg(path.as_ref())
        .stdout(Stdio::piped())
        .output()
        .map_err(|_| ())?;
    if !out.status.success() {
      return Err(());
    }
    DfParse::parse(out.stdout)
  }

  pub fn parse<O: AsRef<[u8]>>(out: O) -> Result<DfParse, ()> {
    let out = BufReader::new(Cursor::new(out.as_ref()));
    for (line_nr, line) in out.lines().enumerate() {
      let line = line.unwrap();
      //println!("DEBUG:  DfParse::parse: line={:?}", line.as_bytes());
      match line_nr {
        0 => {}
        1 => {
          let mut sep_start = None;
          let mut sep_fin = None;
          for (i, c) in line.char_indices() {
            if c == ' ' {
              if sep_start.is_none() {
                sep_start = Some(i);
              }
              sep_fin = Some(i);
            }
          }
          if sep_start.is_none() {
            return Err(());
          }
          let source = PathBuf::from(line.get( .. sep_start.unwrap()).unwrap());
          let target = PathBuf::from(line.get(sep_fin.unwrap() + 1 .. ).unwrap());
          return Ok(DfParse{source, target});
        }
        _ => unreachable!()
      }
    }
    Err(())
  }
}
