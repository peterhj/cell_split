extern crate blake2;
//extern crate blake3;
extern crate byteorder;
//extern crate libc;
extern crate rustc_serialize;
extern crate smol_str;

use crate::swap::{SwapThreadPool, DfParse};

use blake2::{Blake2b};
use byteorder::{ReadBytesExt, LittleEndian as LE};
use rustc_serialize::*;
use rustc_serialize::hex::{FromHex, ToHex};
use rustc_serialize::json::{JsonLines, JsonEncoder};
use rustc_serialize::utf8::{len_utf8};
use smol_str::{SmolStr};

use std::collections::{BTreeMap, BTreeSet};
//use std::convert::{TryFrom, TryInto};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::fs::{File, OpenOptions, remove_file};
use std::io::{Read, Write, Seek, SeekFrom, Cursor, Error as IoError};
use std::mem::{replace};
use std::path::{PathBuf, Path};
//use std::ptr::{copy_nonoverlapping};
use std::str::{FromStr};
use std::sync::{Arc, Mutex};

//pub mod mmap;
pub mod safetensor;
pub mod swap;

#[derive(Clone, Default, Debug)]
pub struct Header {
  pub split: Option<SplitHeader>,
  pub rows: Vec<Row>,
  pub next: Option<NextHeader>,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct SplitHeader {
  //pub split: SplitInner,
  pub rank: u32,
  pub size: u32,
  pub split_rep: SplitRepr,
}

/*#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct SplitInner {
}*/

#[derive(Clone, Copy, Debug)]
pub enum SplitRepr {
  // TODO
  One,
  //RoundRobin,
  Balanced,
  //Hash,
}

impl FromStr for SplitRepr {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<SplitRepr, SmolStr> {
    Ok(match s {
      "1" => SplitRepr::One,
      //"round_robin" => SplitRepr::RoundRobin,
      "balanced" => SplitRepr::Balanced,
      _ => return Err(s.into())
    })
  }
}

impl SplitRepr {
  pub fn to_str(&self) -> &'static str {
    match self {
      &SplitRepr::One => "1",
      //&SplitRepr::RoundRobin => "round_robin",
      &SplitRepr::Balanced => "balanced",
    }
  }
}

impl Decodable for SplitRepr {
  fn decode<D: Decoder>(d: &mut D) -> Result<SplitRepr, D::Error> {
    SplitRepr::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid SplitRepr"))
  }
}

impl Encodable for SplitRepr {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self.to_str())
  }
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct NextHeader {
  // TODO
  pub next_off: u64,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct Row {
  // TODO
  pub key:  SmolStr,
  //pub ver:  Version,
  pub ty:   CellType,
  pub rep:  CellRepr,
  pub off:  u64,
  pub eoff: u64,
  pub hash: Vec<HashVal>,
}

//pub type Version = u64;

/*#[derive(Clone, Copy, Debug, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct Version {
  pub rst: u64,
  pub up: i32,
}*/

#[derive(Clone, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct CellType {
  pub shape: Box<[i64]>,
  pub dtype: Dtype,
}

impl Debug for CellType {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "CellType({:?}{})", self.shape, self.dtype.to_str())
  }
}

#[repr(u8)]
enum TypeParser {
  LBrack,
  Len,
}

impl FromStr for CellType {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<CellType, SmolStr> {
    let mut shape = Vec::new();
    let mut parse = TypeParser::LBrack;
    let mut o = 0;
    loop {
      match parse {
        TypeParser::LBrack => {
          let c = s[o .. ].chars().next();
          match c {
            Some('[') => {
              parse = TypeParser::Len;
              o += len_utf8(c.unwrap() as _);
            }
            _ => {
              let shape = shape.into();
              let dtype = Dtype::from_str(&s[o .. ])?;
              return Ok(CellType{shape, dtype});
            }
          }
        }
        TypeParser::Len => {
          let o0 = o;
          // FIXME: allow hexadecimal lengths?
          loop {
            let c = s[o .. ].chars().next();
            match c {
              Some('0') |
              Some('1') |
              Some('2') |
              Some('3') |
              Some('4') |
              Some('5') |
              Some('6') |
              Some('7') |
              Some('8') |
              Some('9') => {
                o += len_utf8(c.unwrap() as _);
              }
              Some(']') => {
                let len: i64 = match (&s[o0 .. o]).parse() {
                  Ok(n) => n,
                  Err(_) => return Err(s.into())
                };
                shape.push(len);
                parse = TypeParser::LBrack;
                o += len_utf8(c.unwrap() as _);
                break;
              }
              _ => return Err(s.into())
            }
          }
        }
      }
    }
  }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum Dtype {
  F64 = 1,
  F32,
  I64,
  I32,
  I16,
  I8,
  U64,
  U32,
  U16,
  U8,
  F16,
  Bf16,
}

impl FromStr for Dtype {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<Dtype, SmolStr> {
    Ok(match s {
      "f64" => Dtype::F64,
      "f32" => Dtype::F32,
      "i64" => Dtype::I64,
      "i32" => Dtype::I32,
      "i16" => Dtype::I16,
      "i8" => Dtype::I8,
      "u64" => Dtype::U64,
      "u32" => Dtype::U32,
      "u16" => Dtype::U16,
      "u8" => Dtype::U8,
      "f16" => Dtype::F16,
      "bf16" => Dtype::Bf16,
      _ => return Err(s.into())
    })
  }
}

impl Dtype {
  pub fn to_str(&self) -> &'static str {
    match self {
      &Dtype::F64 => "f64",
      &Dtype::F32 => "f32",
      &Dtype::I64 => "i64",
      &Dtype::I32 => "i32",
      &Dtype::I16 => "i16",
      &Dtype::I8  => "i8",
      &Dtype::U64 => "u64",
      &Dtype::U32 => "u32",
      &Dtype::U16 => "u16",
      &Dtype::U8  => "u8",
      &Dtype::F16 => "f16",
      &Dtype::Bf16 => "bf16",
    }
  }

  pub fn size_bytes(&self) -> Option<u64> {
    Some(match self {
      &Dtype::F64 => 8,
      &Dtype::F32 => 4,
      &Dtype::I64 => 8,
      &Dtype::I32 => 4,
      &Dtype::I16 => 2,
      &Dtype::I8  => 1,
      &Dtype::U64 => 8,
      &Dtype::U32 => 4,
      &Dtype::U16 => 2,
      &Dtype::U8  => 1,
      &Dtype::F16 => 2,
      &Dtype::Bf16 => 2,
    })
  }
}

impl Decodable for Dtype {
  fn decode<D: Decoder>(d: &mut D) -> Result<Dtype, D::Error> {
    Dtype::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid dtype"))
  }
}

impl Encodable for Dtype {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self.to_str())
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CellRepr {
  Nd,
}

impl FromStr for CellRepr {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<CellRepr, SmolStr> {
    Ok(match s {
      "nd" => CellRepr::Nd,
      _ => return Err(s.into())
    })
  }
}

impl CellRepr {
  pub fn to_str(&self) -> &'static str {
    match self {
      &CellRepr::Nd => "nd",
    }
  }
}

impl Decodable for CellRepr {
  fn decode<D: Decoder>(d: &mut D) -> Result<CellRepr, D::Error> {
    CellRepr::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid dtype"))
  }
}

impl Encodable for CellRepr {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self.to_str())
  }
}

#[derive(Clone, Copy, Debug)]
pub enum HashFun {
  Blake2b,
  Blake3,
}

impl FromStr for HashFun {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<HashFun, SmolStr> {
    Ok(match s {
      "blake2b" => HashFun::Blake2b,
      "blake3" => HashFun::Blake3,
      _ => return Err(s.into())
    })
  }
}

impl HashFun {
  pub fn to_str(&self) -> &'static str {
    match self {
      &HashFun::Blake2b => "blake2b",
      &HashFun::Blake3 => "blake3",
    }
  }
}

impl Decodable for HashFun {
  fn decode<D: Decoder>(d: &mut D) -> Result<HashFun, D::Error> {
    HashFun::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid HashFun"))
  }
}

impl Encodable for HashFun {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self.to_str())
  }
}

#[derive(Clone)]
pub struct HashHexVal {
  pub buf: Vec<u8>,
}

impl Debug for HashHexVal {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "HashHexVal({})", self.buf.to_hex())
  }
}

impl Decodable for HashHexVal {
  fn decode<D: Decoder>(d: &mut D) -> Result<HashHexVal, D::Error> {
    let s = String::decode(d)?;
    // FIXME: rustc_serialize::Decoder::Error should impl
    // Default, FromStr or both.
    let buf = s.from_hex().unwrap();
    Ok(HashHexVal{buf})
  }
}

impl Encodable for HashHexVal {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    let s = self.buf.to_hex();
    e.emit_str(&s)
  }
}

#[derive(Clone, RustcDecodable, RustcEncodable)]
pub struct HashVal {
  pub fun: HashFun,
  pub val: HashHexVal,
}

impl Debug for HashVal {
  fn fmt(&self, f: &mut Formatter) -> FmtResult {
    write!(f, "HashVal({:?}: {})", self.fun, self.val.buf.to_hex())
  }
}

#[derive(Debug)]
pub enum ParseError {
  Io(IoError),
  _Bot,
}

impl From<IoError> for ParseError {
  fn from(e: IoError) -> ParseError {
    ParseError::Io(e)
  }
}

impl Header {
  pub fn parse_bytes(buf: &[u8]) -> Result<Header, ParseError> {
    let mut reader = Cursor::new(buf);
    /*let magic = reader.read_u64::<LE>()?.to_le_bytes();
    if &magic != b"cellsplt" {
      return Err(ParseError::_Bot);
    }*/
    let jit = JsonLines::from_reader(reader);
    let mut split = None;
    let mut rows = Vec::new();
    let mut next = None;
    for (i, j) in jit.enumerate() {
      let j = match j {
        Err(_) => {
          //return Err(ParseError::_Bot);
          break;
        }
        Ok(j) => j
      };
      if i == 0 {
        match j.clone().decode_into::<SplitHeader>() {
          Err(_) => {}
          Ok(h) => {
            split = Some(h);
            continue;
          }
        }
      }
      /*if next.is_some() {
        // FIXME: trailing.
        break;
      }*/
      match j.clone().decode_into::<Row>() {
        Err(_) => {
          match j.decode_into::<NextHeader>() {
            Err(_) => {
              return Err(ParseError::_Bot);
            }
            Ok(h) => {
              next = Some(h);
              break;
            }
          }
        }
        Ok(h) => {
          rows.push(h);
        }
      }
    }
    Ok(Header{split, rows, next})
  }
}

struct AppendSharedState {
  doff: Vec<u64>,
}

pub(crate) struct AppendState {
  file: File,
  hoff: u64,
  hcur: u64,
  htmp: String,
  head: Header,
  hrow: u32,
  hctr: u16,
  sync: bool,
  //pool: bool,
  //splt: bool,
  doff: u64,
  //eoff: u64,
}

impl Drop for AppendState {
  fn drop(&mut self) {
    // FIXME: send to pool.
    /*
    if self.hcur > self.hoff + self.htmp.len() as u64 {
      panic!("bug");
    } else if self.hcur == self.hoff + self.htmp.len() as u64 {
      return;
    }
    assert!(self.htmp.len() < 0x1000);
    assert_eq!(self.hcur, self.hoff);
    self.file.seek(SeekFrom::Start(self.hoff)).unwrap();
    self.file.write_all(self.htmp.as_bytes()).unwrap();
    self.hcur += self.htmp.len() as u64;
    assert!(self.hcur < self.hoff + 0x1000);
    */
  }
}

struct IndexState {
  head: Vec<Header>,
  key:  BTreeMap<SmolStr, IndexVal>,
  //key:  BTreeMap<SmolStr, u32>,
}

struct IndexVal {
  //rank: u32,
  ty:   CellType,
  rep:  CellRepr,
  off:  u64,
  eoff: u64,
}

enum Mode {
  Top,
  Append(AppendSharedState, Vec<Arc<Mutex<AppendState>>>),
  Index(Vec<IndexState>),
  Bot,
}

pub struct CellSplit {
  path: Vec<PathBuf>,
  mode: Mode,
  //rank: u32,
  pool: SwapThreadPool,
}

impl Drop for CellSplit {
  fn drop(&mut self) {
    match &self.mode {
      &Mode::Append(_, ref states) => {
        for rank in 0 .. states.len() {
          let mut state = states[rank].lock().unwrap();
          if state.hcur > state.hoff + state.htmp.len() as u64 {
            panic!("bug");
          } else if state.hcur == state.hoff + state.htmp.len() as u64 {
            continue;
          }
          assert!(state.htmp.len() < 0x1000);
          assert!(state.head.next.is_none());
          assert_eq!(state.hcur, state.hoff);
          let htmp = replace(&mut state.htmp, String::new());
          let head = replace(&mut state.head, Header::default());
          self.pool.put_header(rank, states[rank].clone(), state.hctr, state.hoff, htmp, head);
          state.hctr += 1;
          state.hrow = 0;
          //self.pool.sync(rank);
          state.hcur += state.htmp.len() as u64;
          assert!(state.hcur < state.hoff + 0x1000);
          assert_eq!(state.hcur, state.hoff + state.htmp.len() as u64);
        }
        /*for rank in 0 .. states.len() {
          let mut state = states[rank].lock().unwrap();
          //assert_eq!(self.pool.wait(), rank);
          self.pool.shutdown();
        }*/
      }
      _ => {}
    }
  }
}

impl CellSplit {
  pub fn new1<P: AsRef<Path>>(path: P) -> CellSplit {
    let n = 1;
    let path = path.as_ref();
    CellSplit{
      path: vec![PathBuf::from(path)],
      mode: Mode::Top,
      pool: SwapThreadPool::new(n),
    }
  }

  pub fn new2<P0: AsRef<Path>, P1: AsRef<Path>>(p0: P0, p1: P1) -> CellSplit {
    let n = 2;
    CellSplit{
      path: vec![PathBuf::from(p0.as_ref()), PathBuf::from(p1.as_ref())],
      mode: Mode::Top,
      pool: SwapThreadPool::new(n),
    }
  }

  pub fn new_<P: AsRef<Path>>(paths: &[P]) -> CellSplit {
    //println!("DEBUG:  CellSplit::new_: blake2::about={:?}", blake2::about());
    let n = paths.len();
    let mut path = Vec::with_capacity(paths.len());
    let mut srcs = BTreeSet::new();
    for (r, p) in paths.iter().enumerate() {
      let p = PathBuf::from(p.as_ref());
      match p.parent() {
        None => {
          println!("WARNING:CellSplit::new_: failed to query disk source for rank {} (path parent?)", r);
        }
        Some(p2) => {
          match DfParse::open(&p2) {
            Err(_) => {
              println!("WARNING:CellSplit::new_: failed to query disk source for rank {} (df?)", r);
            }
            Ok(parse) => {
              srcs.insert(parse.source);
            }
          }
        }
      }
      path.push(p);
    }
    if srcs.len() < n {
      println!("WARNING:CellSplit::new_: disk source cardinality {} is less than the size {}", srcs.len(), n);
    }
    CellSplit{
      path,
      mode: Mode::Top,
      pool: SwapThreadPool::new(n),
    }
  }

  pub fn new<P: AsRef<Path>>(mut roots: Vec<PathBuf>, prefix: P) -> CellSplit {
    let n = roots.len();
    let prefix = prefix.as_ref();
    let mut path = roots;
    for p in path.iter_mut() {
      *p = p.join(prefix);
    }
    CellSplit{
      path,
      mode: Mode::Top,
      pool: SwapThreadPool::new(n),
    }
  }

  pub fn _append(&mut self) {
    match &self.mode {
      &Mode::Top => {
        let size = self.path.len() as u32;
        let split_rep = if size == 0 {
          panic!("bug");
        } else if size == 1 {
          SplitRepr::One
        } else {
          SplitRepr::Balanced
        };
        let mut shared = AppendSharedState{doff: Vec::new()};
        let mut states = Vec::new();
        for (r, p) in self.path.iter().enumerate() {
          let rank = r as u32;
          let _ = remove_file(p);
          let file = OpenOptions::new()
            .read(false).write(true).create(true).truncate(true)
            .open(p).unwrap();
          let mut state = AppendState{
            file,
            hoff: 0,
            hcur: 0,
            htmp: String::new(),
            head: Header::default(),
            hrow: 0,
            hctr: 0,
            sync: false,
            //splt: false,
            doff: 0x1000,
            //eoff: 0x1000,
          };
          {
            let split = SplitHeader{
              rank,
              size,
              split_rep,
            };
            {
              let mut enc = JsonEncoder::new(&mut state.htmp);
              split.encode(&mut enc).unwrap();
            }
            state.htmp.push('\n');
            assert!(state.htmp.len() + 40 < 0x1000);
            state.head.split = Some(split);
            //state.splt = true;
          }
          states.push(Arc::new(Mutex::new(state)));
          shared.doff.push(0x1000);
        }
        self.mode = Mode::Append(shared, states);
      }
      &Mode::Append(..) => {}
      &Mode::Index(_) => panic!("bug"),
      _ => panic!("bug")
    }
  }

  pub fn _index(&mut self) {
    match &self.mode {
      &Mode::Top |
      &Mode::Append(..) => {
        let mut states = Vec::new();
        for p in self.path.iter() {
          let mut file = File::open(p).unwrap();
          let mut headers = Vec::new();
          let mut key = BTreeMap::new();
          let mut hbuf = Vec::with_capacity(0x1000);
          let mut hoff = 0;
          let mut end = false;
          while !end {
            hbuf.clear();
            hbuf.resize(0x1000, 0);
            file.seek(SeekFrom::Start(hoff)).unwrap();
            let mut n = 0;
            while n < 0x1000 {
              match file.read(&mut hbuf) {
                Ok(0) => break,
                Ok(m) => n += m,
                // FIXME: IO error should yield Bot mode.
                Err(_) => return
              }
            }
            match Header::parse_bytes(&hbuf) {
              Ok(h) => {
                if h.next.is_none() {
                  end = true;
                } else {
                  hoff = h.next.as_ref().unwrap().next_off;
                }
                for row in h.rows.iter() {
                  match key.get(&row.key) {
                    None => {
                      let val = IndexVal{
                        ty: row.ty.clone(),
                        rep: row.rep,
                        off: row.off,
                        eoff: row.eoff,
                      };
                      key.insert(row.key.clone(), val);
                    }
                    Some(_) => {}
                  }
                }
                headers.push(h);
              }
              // FIXME: IO error should yield Bot mode.
              Err(_) => return
            }
          }
          // FIXME
          let state = IndexState{
            head: headers.clone(),
            key,
          };
          states.push(state);
        }
        self.mode = Mode::Index(states);
      }
      &Mode::Index(_) => {}
      _ => panic!("bug")
    }
  }

  pub fn paths(&self) -> &[PathBuf] {
    &self.path
  }

  pub fn headers(&mut self) -> Vec<(u32, Header)> {
    self.clone_headers()
  }

  pub fn clone_headers(&mut self) -> Vec<(u32, Header)> {
    self._index();
    match &self.mode {
      &Mode::Index(ref states) => {
        let mut headers = Vec::new();
        for (rank, state) in states.iter().enumerate() {
          for h in state.head.iter() {
            headers.push((rank as u32, h.clone()));
          }
        }
        headers
      }
      _ => panic!("bug")
    }
  }

  pub fn clone_keys(&mut self) -> BTreeSet<SmolStr> {
    self._index();
    match &self.mode {
      &Mode::Index(ref states) => {
        let mut keys = BTreeSet::new();
        for state in states.iter() {
          for h in state.head.iter() {
            for row in h.rows.iter() {
              assert!(keys.insert(row.key.clone()));
            }
          }
        }
        keys
      }
      _ => panic!("bug")
    }
  }

  pub fn get<K: AsRef<str>>(&mut self, key: K) -> (CellType, CellRepr, u32, u64, u64) {
    self._index();
    match &self.mode {
      &Mode::Index(ref states) => {
        let key = key.as_ref();
        for (rank, state) in states.iter().enumerate() {
          match state.key.get(key) {
            None => panic!("bug"),
            Some(val) => {
              return (val.ty.clone(), val.rep, rank as u32, val.off, val.eoff);
            }
          }
        }
        panic!("bug");
      }
      _ => panic!("bug")
    }
  }

  /*pub fn sync<P: AsRef<Path>>(&self, path: P) {
    unimplemented!();
  }*/

  /*pub fn sync(&self) {
    unimplemented!();
  }*/

  /*pub fn put<K: AsRef<str>, T: Into<CellType>, R: Into<CellRepr>>(&mut self, key: K, ty: T, rep: R, data: &[u8]) -> (u64, u64) {
    self._append();
    match &mut self.mode {
      &mut Mode::Append(ref mut shared, ref mut states) => {
        let (rank, last_doff) = if states.len() == 0 {
          panic!("bug");
        } else if states.len() == 1 {
          (0, states[0].lock().unwrap().doff)
        } else {
          // FIXME: the nonblocking version of put needs
          // an atomic copy of doff.
          let mut min_doff = None;
          for (rank, state) in states.iter().enumerate() {
            let state = state.lock().unwrap();
            match min_doff {
              None => {
                min_doff = Some((rank, state.doff));
              }
              Some((_, o_doff)) => if state.doff < o_doff {
                min_doff = Some((rank, state.doff));
              }
            }
          }
          min_doff.unwrap()
        };
        let mut state = states[rank].lock().unwrap();
        assert_eq!(last_doff, state.doff);
        assert_eq!(state.hcur, state.hoff);
        let olen = state.htmp.len();
        assert!(olen + 40 < 0x1000);
        let key = key.as_ref();
        let mut row = Row{
          key: key.into(),
          ty: ty.into(),
          rep: rep.into(),
          hash: Vec::new(),
          off: state.doff,
          eoff: state.doff + data.len() as u64,
        };
        let mut hasher = Blake2b::default();
        hasher.update(data);
        row.hash.push(HashVal{
          fun: HashFun::Blake2b,
          val: HashHexVal{buf: hasher.finalize().as_bytes().into()},
        });
        let mut retry = false;
        loop {
          {
            let mut enc = JsonEncoder::new(&mut state.htmp);
            row.encode(&mut enc).unwrap();
          }
          state.htmp.push('\n');
          state.head.rows.push(row.clone());
          if state.htmp.len() + 40 >= 0x1000 {
            assert!(!retry);
            state.htmp.truncate(olen);
            let next_off = ((state.doff + 0x1000 - 1) / 0x1000) * 0x1000;
            let next_doff = next_off + 0x1000;
            let next = NextHeader{next_off};
            {
              let mut enc = JsonEncoder::new(&mut state.htmp);
              next.encode(&mut enc).unwrap();
            }
            state.htmp.push('\n');
            assert!(state.htmp.len() < 0x1000);
            state.head.next = Some(next);
            {
              let prev_hoff = state.hoff;
              state.file.seek(SeekFrom::Start(prev_hoff)).unwrap();
              let &mut AppendState{ref mut file, ref htmp, ..} = &mut *state;
              file.write_all(htmp.as_bytes()).unwrap();
            }
            state.htmp.clear();
            let _ = replace(&mut state.head, Header::default());
            state.hctr += 1;
            state.hrow = 0;
            state.hoff = next_off;
            state.hcur = next_off;
            state.doff = next_doff;
            row.off = state.doff;
            row.eoff = state.doff + data.len() as u64;
            retry = true;
            continue;
          }
          break;
        }
        state.hrow += 1;
        let off = state.doff;
        let eoff = state.doff + data.len() as u64;
        assert_eq!(off, row.off);
        assert_eq!(eoff, row.eoff);
        let next_doff = ((eoff + 0x200 - 1) / 0x200) * 0x200;
        state.doff = next_doff;
        shared.doff[rank] = next_doff;
        {
          state.file.seek(SeekFrom::Start(off)).unwrap();
          state.file.write_all(data).unwrap();
        }
        /*
        state.file.set_len(state.doff + data.len() as u64).unwrap();
        let mem = crate::mmap::MmapFile::from_file_part(
            &state.file,
            state.doff,
            data.len() as _,
        ).unwrap();
        unsafe { copy_nonoverlapping(
            data.as_ptr(),
            mem.as_ptr() as *mut u8,
            data.len(),
        ); }
        drop(mem);
        */
        (off, eoff)
      }
      _ => unreachable!()
    }
  }*/

  pub fn nonblocking_put_unsafe<K: AsRef<str>, T: Into<CellType>, R: Into<CellRepr>>(&mut self, key: K, ty: T, rep: R, data: &[u8]) -> (u64, u64) {
    self._append();
    match &mut self.mode {
      &mut Mode::Append(ref mut shared, ref mut states) => {
        let (rank, last_doff) = if states.len() == 0 {
          panic!("bug");
        } else if states.len() == 1 {
          (0, 0)
        } else {
          let mut min_doff = None;
          for rank in 0 .. states.len() {
            let rank_doff = shared.doff[rank];
            match min_doff {
              None => {
                min_doff = Some((rank, rank_doff));
              }
              Some((_, o_doff)) => if rank_doff < o_doff {
                min_doff = Some((rank, rank_doff));
              }
            }
          }
          min_doff.unwrap()
        };
        let mut state = states[rank].lock().unwrap();
        assert_eq!(last_doff, state.doff);
        assert_eq!(state.hcur, state.hoff);
        let olen = state.htmp.len();
        assert!(olen + 40 < 0x1000);
        let key = key.as_ref();
        let mut row = Row{
          key: key.into(),
          ty: ty.into(),
          rep: rep.into(),
          off: state.doff,
          eoff: state.doff + data.len() as u64,
          hash: Vec::new(),
        };
        let mut hbuf = Vec::with_capacity(64);
        hbuf.resize(64, 0);
        let mut hval = HashVal{
          fun: HashFun::Blake2b,
          val: HashHexVal{buf: hbuf},
        };
        row.hash.push(hval);
        let mut retry = false;
        loop {
          {
            let mut enc = JsonEncoder::new(&mut state.htmp);
            row.encode(&mut enc).unwrap();
          }
          state.htmp.push('\n');
          state.head.rows.push(row.clone());
          if state.htmp.len() + 40 >= 0x1000 {
            assert!(!retry);
            state.htmp.truncate(olen);
            state.head.rows.pop();
            let next_off = ((state.doff + 0x1000 - 1) / 0x1000) * 0x1000;
            let next_doff = next_off + 0x1000;
            let next = NextHeader{next_off};
            {
              let mut enc = JsonEncoder::new(&mut state.htmp);
              next.encode(&mut enc).unwrap();
            }
            state.htmp.push('\n');
            assert!(state.htmp.len() < 0x1000);
            state.head.next = Some(next);
            let htmp = replace(&mut state.htmp, String::new());
            let head = replace(&mut state.head, Header::default());
            /*println!("DEBUG:  CellSplit::nonblocking_put_unsafe: header: rank={} hctr={} hoff={} head.rows.len={}",
                rank, state.hctr, state.hoff, head.rows.len());
            for (i, row) in head.rows.iter().enumerate() {
              println!("DEBUG:  CellSplit::nonblocking_put_unsafe: header:   row[{}]: key={:?} ty={:?} o={} eo={}",
                  i, row.key, row.ty, row.off, row.eoff);
            }*/
            self.pool.put_header(rank, states[rank].clone(), state.hctr, state.hoff, htmp, head);
            state.hctr += 1;
            state.hrow = 0;
            state.hoff = next_off;
            state.hcur = next_off;
            state.doff = next_doff;
            row.off = state.doff;
            row.eoff = state.doff + data.len() as u64;
            retry = true;
            continue;
          }
          break;
        }
        let hctr = state.hctr;
        let hrow = state.hrow;
        state.hrow += 1;
        let off = state.doff;
        let eoff = state.doff + data.len() as u64;
        assert_eq!(off, row.off);
        assert_eq!(eoff, row.eoff);
        let next_doff = ((eoff + 0x200 - 1) / 0x200) * 0x200;
        state.doff = next_doff;
        shared.doff[rank] = next_doff;
        drop(state);
        /*println!("DEBUG:  CellSplit::nonblocking_put_unsafe: row:    rank={} hctr={} hrow={} key={:?} ty={:?} o={} eo={}",
            rank, hctr, hrow, row.key, row.ty, row.off, row.eoff);*/
        self.pool.put_row_unsafe(rank, states[rank].clone(), hctr, hrow, row, data.as_ptr(), data.len());
        (off, eoff)
      }
      _ => unreachable!(),
    }
  }
}
