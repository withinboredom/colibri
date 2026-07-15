#!/usr/bin/env python3
"""Shared reader/constants for Colibri's native model container."""

from __future__ import annotations

import struct
from pathlib import Path


MAGIC = b"COLINAT1"
VERSION = 1
ALIGNMENT = 4096
HEADER = struct.Struct("<8sIIQQQQQQ")
TENSOR = struct.Struct("<IIQQQIIQ")
EXPERT = struct.Struct("<IIQQ" + "Q" * 12 + "II")

DTYPE_TO_CODE = {"BF16": 0, "F16": 1, "F32": 2, "U8": 3, "I8": 3}
CODE_TO_DTYPE = {0: "BF16", 1: "F16", 2: "F32", 3: "U8"}


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & -alignment


def read_native(path, *, include_tensors=True):
    """Read and bounds-check a native container's indexes."""
    path = Path(path)
    actual_size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"short native header: {path}")
        (magic, version, flags, file_size, tensor_count, expert_count,
         tensor_index, expert_index, data_start) = HEADER.unpack(raw)
        if magic != MAGIC:
            raise ValueError(f"invalid native model magic: {path}")
        if version != VERSION:
            raise ValueError(f"unsupported native model version {version}: {path}")
        if file_size != actual_size:
            raise ValueError(f"native model size mismatch: header={file_size}, file={actual_size}")
        if data_start < HEADER.size or tensor_index < data_start or expert_index < tensor_index:
            raise ValueError(f"invalid native model indexes: {path}")

        tensors = []
        tensor_names = set()
        stream.seek(tensor_index)
        for _ in range(tensor_count):
            fixed = stream.read(TENSOR.size)
            if len(fixed) != TENSOR.size:
                raise ValueError(f"truncated native tensor index: {path}")
            name_len, dtype, offset, nbytes, numel, crc32, tensor_flags, _reserved = TENSOR.unpack(fixed)
            if not name_len or name_len > 1 << 20 or dtype not in CODE_TO_DTYPE:
                raise ValueError(f"invalid native tensor entry: {path}")
            name_raw = stream.read(align_up(name_len, 8))
            if len(name_raw) != align_up(name_len, 8):
                raise ValueError(f"truncated native tensor name: {path}")
            if offset < data_start or nbytes > file_size or offset > file_size - nbytes:
                raise ValueError(f"native tensor is outside the file: {path}")
            if include_tensors:
                try:
                    name = name_raw[:name_len].decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"invalid native tensor name: {path}") from error
                if "\0" in name or name in tensor_names:
                    raise ValueError(f"invalid or duplicate native tensor name: {path}")
                tensor_names.add(name)
                tensors.append({"name": name, "dtype": dtype, "offset": offset,
                                "nbytes": nbytes, "numel": numel, "crc32": crc32,
                                "flags": tensor_flags})
        if stream.tell() != expert_index:
            raise ValueError(f"native expert index offset mismatch: {path}")

        experts = []
        expert_layers = set()
        for _ in range(expert_count):
            raw = stream.read(EXPERT.size)
            if len(raw) != EXPERT.size:
                raise ValueError(f"truncated native expert index: {path}")
            values = EXPERT.unpack(raw)
            layer, n_experts, base, stride = values[:4]
            offsets = list(values[4:10])
            lengths = list(values[10:16])
            fmt, group_size = values[16:18]
            if layer in expert_layers:
                raise ValueError(f"duplicate native expert layer {layer}: {path}")
            expert_layers.add(layer)
            if not n_experts or not stride or base % ALIGNMENT or stride % ALIGNMENT:
                raise ValueError(f"invalid native expert layout for layer {layer}: {path}")
            if base < data_start or n_experts > (file_size - base) // stride:
                raise ValueError(f"native expert records are outside the file: {path}")
            for offset, length in zip(offsets, lengths):
                if offset > stride or length > stride - offset:
                    raise ValueError(f"native expert field is outside its record: {path}")
            scale_start = offsets[3]
            if scale_start % ALIGNMENT or any(offset + length > scale_start
                                              for offset, length in zip(offsets[:3], lengths[:3])):
                raise ValueError(f"invalid native expert weight region: {path}")
            if any(offset < scale_start for offset in offsets[3:]):
                raise ValueError(f"invalid native expert scale region: {path}")
            experts.append({"layer": layer, "n_experts": n_experts, "base": base,
                            "stride": stride, "offsets": offsets, "lengths": lengths,
                            "fmt": fmt, "group_size": group_size})

    return {"path": str(path), "flags": flags, "file_size": file_size,
            "tensor_count": tensor_count, "expert_count": expert_count,
            "data_start": data_start, "tensors": tensors, "experts": experts}
