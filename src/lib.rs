extern crate byteorder;
extern crate rustc_serialize;
extern crate smol_str;

use rustc_serialize::*;
use rustc_serialize::utf8::{len_utf8};
use smol_str::{SmolStr};

//use std::convert::{TryFrom, TryInto};
use std::str::{FromStr};

pub mod safetensor;

#[derive(Clone, Debug)]
pub struct Header {
  pub rows: Vec<Row>,
  pub next: Option<NextHeader>,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct NextHeader {
  // TODO
  pub off: u64,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
pub struct Row {
  // TODO
  pub key: SmolStr,
  pub ver: Version,
  pub type_: CellType,
  pub repr_: CellRepr,
  pub off: u64,
  pub eoff: u64,
}

pub type Version = u64;

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

pub struct SplitFile {
  // TODO
}

impl SplitFile {
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
}
