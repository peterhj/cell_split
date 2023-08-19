use crate::{Dtype};

use byteorder::{ReadBytesExt, LittleEndian as LE};
use rustc_serialize::json::{Json, ParserError, DecoderError, decode_from_json};
use smol_str::{SmolStr};

use std::collections::{BTreeMap};
use std::convert::{TryInto};
use std::io::{Read, Cursor, Error as IoError};
use std::mem::{size_of};

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

#[derive(Clone, RustcDecodable, Debug)]
pub struct TensorEntry {
  pub shape: Vec<i64>,
  pub dtype: Dtype,
  pub offsets: [u64; 2],
}

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
    let magic = reader.read_u64::<LE>()?;
    let buf_start = (size_of::<u64>() as u64) + magic;
    let h_sz: usize = magic.try_into().unwrap();
    let mut hbuf = Vec::with_capacity(h_sz);
    hbuf.resize(h_sz, 0);
    reader.read_exact(&mut hbuf)?;
    let hjson = Json::from_reader(Cursor::new(hbuf))?;
    let (hjson, _) = fixup_toplevel_metadata(hjson);
    let entries = decode_from_json(hjson)?;
    Ok(TensorsDict{buf_start, entries})
  }
}

pub struct TensorsDictBuilder {
  //pub buf_start: Option<u64>,
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

  pub fn insert<K: AsRef<str>>(&mut self, key: K, shape: Vec<i64>, dtype: Dtype) -> Result<(), ()> {
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
}
