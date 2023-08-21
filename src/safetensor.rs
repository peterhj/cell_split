//use crate::{Dtype};

use byteorder::{ReadBytesExt, WriteBytesExt, LittleEndian as LE};
use rustc_serialize::*;
use rustc_serialize::json::{Config, Json, ParserError, DecoderError, decode_from_json, encode_to_bytes};
use smol_str::{SmolStr};

use std::collections::{BTreeMap};
use std::convert::{TryInto};
use std::io::{Read, Write, Cursor, Error as IoError};
use std::mem::{size_of};
use std::str::{FromStr};

// TODO

#[derive(Debug)]
pub enum Error {
  Io(IoError),
  ParseJson(ParserError),
  DecodeJson(DecoderError),
}

impl From<IoError> for Error {
  fn from(e: IoError) -> Error {
    Error::Io(e)
  }
}

impl From<ParserError> for Error {
  fn from(e: ParserError) -> Error {
    Error::ParseJson(e)
  }
}

impl From<DecoderError> for Error {
  fn from(e: DecoderError) -> Error {
    Error::DecodeJson(e)
  }
}

#[derive(Clone, Copy, Debug)]
pub enum TensorDtype {
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
  Bool,
  F16,
  Bf16,
}

impl FromStr for TensorDtype {
  type Err = SmolStr;

  fn from_str(s: &str) -> Result<TensorDtype, SmolStr> {
    Ok(match s {
      "F64" => TensorDtype::F64,
      "F32" => TensorDtype::F32,
      "I64" => TensorDtype::I64,
      "I32" => TensorDtype::I32,
      "I16" => TensorDtype::I16,
      "I8" => TensorDtype::I8,
      "U64" => TensorDtype::U64,
      "U32" => TensorDtype::U32,
      "U16" => TensorDtype::U16,
      "U8" => TensorDtype::U8,
      "BOOL" => TensorDtype::Bool,
      "F16" => TensorDtype::F16,
      "BF16" => TensorDtype::Bf16,
      _ => return Err(s.into())
    })
  }
}

impl TensorDtype {
  pub fn to_str(&self) -> &'static str {
    match self {
      &TensorDtype::F64 => "F64",
      &TensorDtype::F32 => "F32",
      &TensorDtype::I64 => "I64",
      &TensorDtype::I32 => "I32",
      &TensorDtype::I16 => "I16",
      &TensorDtype::I8  => "I8",
      &TensorDtype::U64 => "U64",
      &TensorDtype::U32 => "U32",
      &TensorDtype::U16 => "U16",
      &TensorDtype::U8  => "U8",
      &TensorDtype::Bool => "BOOL",
      &TensorDtype::F16 => "F16",
      &TensorDtype::Bf16 => "BF16",
    }
  }

  pub fn size_bytes(&self) -> Option<u64> {
    Some(match self {
      &TensorDtype::F64 => 8,
      &TensorDtype::F32 => 4,
      &TensorDtype::I64 => 8,
      &TensorDtype::I32 => 4,
      &TensorDtype::I16  => 2,
      &TensorDtype::I8  => 1,
      &TensorDtype::U64 => 8,
      &TensorDtype::U32 => 4,
      &TensorDtype::U16  => 2,
      &TensorDtype::U8  => 1,
      &TensorDtype::Bool => 1,
      &TensorDtype::F16 => 2,
      &TensorDtype::Bf16 => 2,
    })
  }
}

impl Decodable for TensorDtype {
  fn decode<D: Decoder>(d: &mut D) -> Result<TensorDtype, D::Error> {
    TensorDtype::from_str(d.read_str()?.as_str()).map_err(|_| d.error("invalid dtype"))
  }
}

impl Encodable for TensorDtype {
  fn encode<E: Encoder>(&self, e: &mut E) -> Result<(), E::Error> {
    e.emit_str(self.to_str())
  }
}

#[derive(Clone, RustcDecodable, Debug)]
pub struct TensorEntry {
  pub shape: Vec<i64>,
  pub dtype: TensorDtype,
  pub data_offsets: Vec<u64>,
  //pub data_offsets: [u64; 2],
}

#[derive(Debug)]
pub struct TensorsDict {
  pub buf_start: u64,
  pub entries: BTreeMap<SmolStr, TensorEntry>,
  //pub metadata: Option<HashMap<SmolStr, Json>>,
}

pub fn fixup_toplevel_metadata(mut j: Json) -> (Json, Option<Json>) {
  let mut metadata = None;
  match j {
    Json::Object(mut kvs) => {
      match kvs.remove("__metadata__") {
        None => {}
        Some(v) => {
          metadata = Some(v);
        }
      }
      j = Json::Object(kvs);
    }
    _ => {}
  }
  (j, metadata)
}

impl TensorsDict {
  pub fn from_reader<R: Read>(mut reader: R) -> Result<TensorsDict, Error> {
    println!("DEBUG: TensorsDict::from_reader");
    let magic = reader.read_u64::<LE>()?;
    println!("DEBUG: TensorsDict::from_reader: magic=0x{:016x}", magic);
    let buf_start = (size_of::<u64>() as u64) + magic;
    println!("DEBUG: TensorsDict::from_reader: buf start={}", buf_start);
    let h_sz: usize = magic.try_into().unwrap();
    println!("DEBUG: TensorsDict::from_reader: header sz={}", h_sz);
    let mut hbuf = Vec::with_capacity(h_sz);
    hbuf.resize(h_sz, 0);
    reader.read_exact(&mut hbuf)?;
    let mut cfg = Config::default();
    cfg.allow_trailing = true;
    cfg.eof_on_trailing_spaces = true;
    let hjson = cfg.from_reader(Cursor::new(hbuf)).build()?;
    let (hjson, _) = fixup_toplevel_metadata(hjson);
    let entries = decode_from_json(hjson)?;
    Ok(TensorsDict{buf_start, entries})
  }

  /*pub fn write_header<W: Write>(&self, mut writer: W) -> Result<(), Error> {
    unimplemented!();
  }*/
}

/*#[derive(Debug)]
pub struct TensorsDictBuilder {
  pub end_offset: u64,
  pub entries: Vec<TensorEntry>,
  pub keys: BTreeMap<SmolStr, usize>,
}

impl TensorsDictBuilder {
  pub fn new() -> TensorsDictBuilder {
    TensorsDictBuilder{
      end_offset: 0,
      entries: Vec::new(),
      keys: BTreeMap::new(),
    }
  }

  pub fn insert<K: AsRef<str>>(&mut self, key: K, shape: Vec<i64>, dtype: TensorDtype) -> Result<(), ()> {
    let key = key.as_ref();
    match self.keys.get(key) {
      None => {}
      Some(_) => {
        return Err(());
      }
    }
    let mut flat_sz = dtype.size_bytes().unwrap();
    for &s in shape.iter().rev() {
      if s < 0 {
        return Err(());
      }
      flat_sz *= s as u64;
    }
    let next_off = self.end_offset;
    let next_end_off = self.end_offset + flat_sz;
    let entry = TensorEntry{shape, dtype, offsets: [next_off, next_end_off]};
    self.keys.insert(key.into(), self.entries.len());
    self.entries.push(entry);
    self.end_offset = next_end_off;
    Ok(())
  }

  pub fn finalize(self) -> Result<TensorsDict, Error> {
    let mut entries = BTreeMap::new();
    for (key, &idx) in self.keys.iter() {
      let e = self.entries[idx].clone();
      entries.insert(key.into(), e);
    }
    let hbuf = encode_to_bytes(&entries)?;
    let h_sz = hbuf.len();
    let buf_start = (size_of::<u64>() as u64) + h_sz as u64;
    Ok(TensorsDict{buf_start, entries})
  }
}*/
