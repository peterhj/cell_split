#[cfg(feature = "nightly")]
extern crate blake2;
extern crate byteorder;
extern crate rustc_serialize;
extern crate smol_str;

#[cfg(feature = "nightly")]
pub use crate::impl_::{CellSplit};

use rustc_serialize::*;
use rustc_serialize::hex::{FromHex, ToHex};
use rustc_serialize::json::{JsonLines};
use rustc_serialize::utf8::{len_utf8};
use smol_str::{SmolStr};

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::io::{Read, Write, Cursor, Error as IoError};
use std::str::{FromStr};

#[cfg(feature = "nightly")]
pub mod impl_;
//pub mod mmap;
pub mod safetensor;
#[cfg(feature = "nightly")]
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
#[non_exhaustive]
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
#[non_exhaustive]
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
      _ => unimplemented!()
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
      _ => unimplemented!()
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
#[non_exhaustive]
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
#[non_exhaustive]
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
