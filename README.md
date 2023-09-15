`cell_split` is a library for serialization and deserialization
of numerical arrays (or "cells"). It is currently designed to
interoperate well with [cacti](https://github.com/peterhj/cacti),
though standalone use is also possible but not the primary aim.

The `cell_split` serialized format adheres to a few design goals:

1.  It implements a simple append-only key-value store, where
    records adhere to JSON Lines (one JSON line per record).
2.  It is easy to `mmap`; in practice, this means that record
    data are block-aligned.
3.  It may be split across multiple backing files for I/O
    parallelism and load balancing.

This library also contains a simple read-only implementation of
[safetensors](https://github.com/huggingface/safetensors).
