extern crate rustc_serialize;
extern crate smol_str;

use rustc_serialize::*;
use smol_str::{SmolStr};

//use std::convert::{TryFrom, TryInto};
use std::str::{FromStr};

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

#[derive(Clone, Copy, Debug, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct Version {
  pub rst: u64,
  pub up: i32,
}

#[derive(Clone, Debug, RustcDecodable, RustcEncodable)]
//#[derive(Clone, Debug)]
pub struct CellType {
  pub shape: Vec<i64>,
  pub dtype: Dtype,
}

// TODO: parsing Futhark-style type.

#[derive(Clone, Copy, Debug)]
pub enum Dtype {
  F64,
  F32,
  // TODO
  F16,
}

impl FromStr for Dtype {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<Dtype, SmolStr> {
    Ok(match s {
      "f64" => Dtype::F64,
      "f32" => Dtype::F32,
      // TODO
      "f16" => Dtype::F16,
      _ => return Err(s.into())
    })
  }
}

impl Dtype {
  pub fn to_str(&self) -> &'static str {
    match self {
      &Dtype::F64 => "f64",
      &Dtype::F32 => "f32",
      // TODO
      &Dtype::F16 => "f16",
    }
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

  pub fn get_latest<S: AsRef<str>>(&self, key: S) -> Option<Row> {
    unimplemented!();
  }

  pub fn get<S: AsRef<str>>(&self, key: S, ver: Version) -> Option<Row> {
    unimplemented!();
  }
}
