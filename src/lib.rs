extern crate byteorder;
//extern crate libc;
extern crate rustc_serialize;
extern crate smol_str;

use byteorder::{ReadBytesExt, LittleEndian as LE};
use rustc_serialize::*;
use rustc_serialize::json::{JsonLines, JsonEncoder};
use rustc_serialize::utf8::{len_utf8};
use smol_str::{SmolStr};

use std::collections::{BTreeMap};
//use std::convert::{TryFrom, TryInto};
use std::fs::{File, OpenOptions, remove_file};
use std::io::{Read, Write, Seek, SeekFrom, Cursor, Error as IoError};
use std::path::{PathBuf, Path};
//use std::ptr::{copy_nonoverlapping};
use std::str::{FromStr};

//pub mod mmap;
pub mod safetensor;

#[derive(Clone, Debug)]
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
  //pub hash: HashVal,
  pub off:  u64,
  pub eoff: u64,
}

//pub type Version = u64;

/*#[derive(Clone, Copy, Debug, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct Version {
  pub rst: u64,
  pub up: i32,
}*/

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct CellType {
  pub shape: Vec<i64>,
  pub dtype: Dtype,
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
pub enum Dtype {
  F64,
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

#[derive(Clone, Copy, Debug)]
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
  Blake3,
}

impl FromStr for HashFun {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<HashFun, SmolStr> {
    Ok(match s {
      "blake3" => HashFun::Blake3,
      _ => return Err(s.into())
    })
  }
}

impl HashFun {
  pub fn to_str(&self) -> &'static str {
    match self {
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

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct HashVal {
  pub fun: HashFun,
  pub val: String,
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

struct AppendState {
  file: File,
  hoff: u64,
  hcur: u64,
  //htmp: Vec<u8>,
  htmp: String,
  hrow: u32,
  sync: bool,
  //splt: bool,
  doff: u64,
  //eoff: u64,
}

impl Drop for AppendState {
  fn drop(&mut self) {
    if self.hcur > self.hoff + self.htmp.len() as u64 {
      panic!("bug");
    } else if self.hcur == self.hoff + self.htmp.len() as u64 {
      return;
    }
    assert!(self.htmp.len() < 0x1000);
    assert!(self.hcur == self.hoff);
    self.file.seek(SeekFrom::Start(self.hoff)).unwrap();
    self.file.write_all(self.htmp.as_bytes()).unwrap();
    self.hcur += self.htmp.len() as u64;
    assert!(self.hcur < self.hoff + 0x1000);
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
  Append(Vec<AppendState>),
  Index(Vec<IndexState>),
  Bot,
}

pub struct CellSplit {
  path: Vec<PathBuf>,
  mode: Mode,
  //rank: u32,
}

/*impl Drop for CellSplit {
}*/

impl CellSplit {
  pub fn new1<P: AsRef<Path>>(path: P) -> CellSplit {
    let path = path.as_ref();
    CellSplit{
      path: vec![PathBuf::from(path)],
      mode: Mode::Top,
    }
  }

  pub fn new2<P0: AsRef<Path>, P1: AsRef<Path>>(p0: P0, p1: P1) -> CellSplit {
    CellSplit{
      path: vec![PathBuf::from(p0.as_ref()), PathBuf::from(p1.as_ref())],
      mode: Mode::Top,
    }
  }

  pub fn new<P: AsRef<Path>>(mut roots: Vec<PathBuf>, prefix: P) -> CellSplit {
    let prefix = prefix.as_ref();
    let mut path = roots;
    for p in path.iter_mut() {
      *p = p.join(prefix);
    }
    CellSplit{
      path,
      mode: Mode::Top,
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
            //htmp: Vec::new(),
            htmp: String::new(),
            hrow: 0,
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
            //state.splt = true;
          }
          states.push(state);
        }
        self.mode = Mode::Append(states);
      }
      &Mode::Append(_) => {}
      &Mode::Index(_) => panic!("bug"),
      _ => panic!("bug")
    }
  }

  pub fn _index(&mut self) {
    match &self.mode {
      &Mode::Top |
      &Mode::Append(_) => {
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

  pub fn headers(&mut self) -> Vec<Header> {
    self._index();
    match &self.mode {
      &Mode::Index(ref states) => {
        // FIXME
        states[0].head.clone()
      }
      _ => panic!("bug")
    }
  }

  pub fn get<K: AsRef<str>>(&mut self, key: K) -> (CellType, CellRepr, u64, u64) {
    self._index();
    match &self.mode {
      &Mode::Index(ref states) => {
        let key = key.as_ref();
        // FIXME
        match states[0].key.get(key) {
          None => panic!("bug"),
          Some(val) => {
            (val.ty.clone(), val.rep, val.off, val.eoff)
          }
        }
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

  pub fn put<K: AsRef<str>, T: Into<CellType>, R: Into<CellRepr>>(&mut self, key: K, ty: T, rep: R, data: &[u8]) -> (u64, u64) {
    self._append();
    match &mut self.mode {
      &mut Mode::Append(ref mut states) => {
        let rank = if states.len() == 0 {
          panic!("bug");
        } else if states.len() == 1 {
          0
        } else {
          let mut min_doff = None;
          for (rank, state) in states.iter().enumerate() {
            match min_doff {
              None => {
                min_doff = Some((rank, state.doff));
              }
              Some((_, o_doff)) => if state.doff < o_doff {
                min_doff = Some((rank, state.doff));
              }
            }
          }
          min_doff.unwrap().0
        };
        let state = &mut states[rank];
        let key = key.as_ref();
        let mut row = Row{
          key: key.into(),
          ty: ty.into(),
          rep: rep.into(),
          off: state.doff,
          eoff: state.doff + data.len() as u64,
        };
        assert!(state.hcur == state.hoff);
        let olen = state.htmp.len();
        assert!(olen + 40 < 0x1000);
        let mut retry = false;
        loop {
          {
            let mut enc = JsonEncoder::new(&mut state.htmp);
            row.encode(&mut enc).unwrap();
          }
          state.htmp.push('\n');
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
            state.file.seek(SeekFrom::Start(state.hoff)).unwrap();
            state.file.write_all(state.htmp.as_bytes()).unwrap();
            state.htmp.clear();
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
        state.file.seek(SeekFrom::Start(state.doff)).unwrap();
        state.file.write_all(data).unwrap();
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
        let off = state.doff;
        let eoff = state.doff + data.len() as u64;
        assert_eq!(off, row.off);
        assert_eq!(eoff, row.eoff);
        state.doff = ((eoff + 0x200 - 1) / 0x200) * 0x200;
        (off, eoff)
      }
      _ => unreachable!()
    }
  }
}

/*pub struct CellSplitFile {
  // TODO
}

impl CellSplitFile {
  /*pub fn iter_headers(&self) -> impl Iterator<Item=Header> {
    unimplemented!();
  }*/

  pub fn latest_version(&self) -> Version {
    unimplemented!();
  }

  pub fn get_latest<S: AsRef<str>>(&self, key: S) -> Option<Row> {
    unimplemented!();
  }

  pub fn get<S: AsRef<str>>(&self, key: S, ver: Version) -> Option<Row> {
    unimplemented!();
  }
}*/
