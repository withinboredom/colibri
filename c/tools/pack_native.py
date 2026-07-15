#!/usr/bin/env python3
"""Pack quantized safetensors into Colibri's io_uring-native model layout.

Expert records are 4 KiB aligned and laid out exactly as the runtime consumes
them.  One io_uring READV therefore fills the final weight and scale buffers.
All non-expert tensors are included too, allowing the source safetensors to be
removed after a full checksum verification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_format import (ALIGNMENT, DTYPE_TO_CODE, EXPERT, HEADER, MAGIC,
                           TENSOR, VERSION, align_up, read_native)


EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(\.qs)?$")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
COPY_CHUNK = 8 << 20


def read_sources(model: Path):
    tensors = {}
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards in {model}")
    for shard in shards:
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            raw = stream.read(8)
            if len(raw) != 8:
                raise ValueError(f"short safetensors header: {shard}")
            header_len = int.from_bytes(raw, "little")
            if header_len < 2 or header_len > size - 8:
                raise ValueError(f"invalid safetensors header length: {shard}")
            header = json.loads(stream.read(header_len))
            if not isinstance(header, dict):
                raise ValueError(f"safetensors header is not an object: {shard}")
        data_start = 8 + header_len
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or "\0" in name or not isinstance(meta, dict):
                raise ValueError(f"invalid tensor metadata in {shard}")
            if name in tensors:
                raise ValueError(f"duplicate tensor {name} in {shard}")
            dtype = meta.get("dtype")
            if dtype not in DTYPE_TO_CODE:
                raise ValueError(f"unsupported dtype {dtype} for {name}")
            offsets = meta.get("data_offsets")
            shape = meta.get("shape")
            if not isinstance(offsets, list) or len(offsets) != 2 or not isinstance(shape, list):
                raise ValueError(f"malformed tensor metadata for {name}")
            start, end = map(int, offsets)
            if start < 0 or end < start or data_start + end > size:
                raise ValueError(f"tensor {name} is outside {shard}")
            numel = 1
            for dimension in shape:
                dimension = int(dimension)
                if dimension < 0:
                    raise ValueError(f"negative shape for {name}")
                numel *= dimension
            tensors[name] = {"name": name, "source": shard,
                             "source_offset": data_start + start,
                             "nbytes": end - start, "numel": numel,
                             "dtype": DTYPE_TO_CODE[dtype]}
    return shards, tensors


def expert_key(name):
    match = EXPERT_RE.match(name)
    if not match:
        return None
    layer, expert, projection, scale = match.groups()
    return int(layer), int(expert), PROJECTIONS.index(projection) + (3 if scale else 0)


def quant_format(config, layer, fields):
    hidden = int(config["hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    outputs = (intermediate, intermediate, hidden)
    inputs = (hidden, hidden, intermediate)
    formats = []
    groups = []
    for index in range(3):
        weight_bytes = fields[index]["nbytes"]
        scale_bytes = fields[index + 3]["nbytes"]
        output, input_ = outputs[index], inputs[index]
        if weight_bytes == output * input_:
            if scale_bytes != output * 4:
                raise ValueError(f"unsupported int8 scale layout in layer {layer}")
            fmt, group = 1, 0
        elif weight_bytes == output * ((input_ + 1) // 2):
            if scale_bytes == output * ((input_ + 127) // 128) * 4:
                fmt, group = 4, 128
            elif scale_bytes == output * 4:
                fmt, group = 2, 0
            else:
                raise ValueError(f"unsupported int4 scale layout in layer {layer}")
        elif weight_bytes == output * ((input_ + 3) // 4) and scale_bytes == output * 4:
            fmt, group = 3, 0
        else:
            raise ValueError(f"unsupported expert layout in layer {layer}")
        formats.append(fmt)
        groups.append(group)
    if len(set(formats)) != 1 or len(set(groups)) != 1:
        raise ValueError(f"mixed expert formats in layer {layer}")
    return formats[0], groups[0]


def plan_layout(model: Path, tensors):
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    configured_experts = int(config["n_routed_experts"])
    grouped = {}
    normal = []
    for tensor in tensors.values():
        key = expert_key(tensor["name"])
        if key is None:
            normal.append(tensor)
        else:
            layer, expert, field = key
            grouped.setdefault(layer, {}).setdefault(expert, {})[field] = tensor

    cursor = ALIGNMENT
    for tensor in sorted(normal, key=lambda item: item["name"]):
        cursor = align_up(cursor, 64)
        tensor["offset"] = cursor
        cursor += tensor["nbytes"]

    layouts = []
    cursor = align_up(cursor)
    for layer in sorted(grouped):
        experts = grouped[layer]
        expected = set(range(configured_experts))
        if set(experts) != expected:
            missing = sorted(expected - set(experts))
            raise ValueError(f"layer {layer} is incomplete; missing experts {missing[:8]}")
        template = experts[0]
        if set(template) != set(range(6)):
            raise ValueError(f"layer {layer} expert 0 does not contain six fields")
        lengths = [template[index]["nbytes"] for index in range(6)]
        for expert, fields in experts.items():
            if set(fields) != set(range(6)):
                raise ValueError(f"layer {layer} expert {expert} does not contain six fields")
            if [fields[index]["nbytes"] for index in range(6)] != lengths:
                raise ValueError(f"layer {layer} expert {expert} has a different layout")
        fmt, group_size = quant_format(config, layer, template)
        offsets = [0] * 6
        for index in range(1, 3):
            offsets[index] = offsets[index - 1] + lengths[index - 1]
        weight_span = align_up(sum(lengths[:3]))
        offsets[3] = weight_span
        offsets[4] = offsets[3] + lengths[3]
        offsets[5] = offsets[4] + lengths[4]
        stride = weight_span + align_up(sum(lengths[3:]))
        base = align_up(cursor)
        for expert, fields in experts.items():
            for field, tensor in fields.items():
                tensor["offset"] = base + expert * stride + offsets[field]
        layouts.append({"layer": layer, "n_experts": configured_experts,
                        "base": base, "stride": stride, "offsets": offsets,
                        "lengths": lengths, "fmt": fmt, "group_size": group_size})
        cursor = base + configured_experts * stride
    return sorted(normal, key=lambda item: item["offset"]), grouped, layouts, cursor


def copy_tensor(source_handles, destination, tensor):
    source = source_handles.get(tensor["source"])
    if source is None:
        source = tensor["source"].open("rb", buffering=0)
        source_handles[tensor["source"]] = source
    source.seek(tensor["source_offset"])
    destination.seek(tensor["offset"])
    remaining = tensor["nbytes"]
    checksum = 0
    while remaining:
        chunk = source.read(min(COPY_CHUNK, remaining))
        if not chunk:
            raise OSError(f"short read while copying {tensor['name']}")
        destination.write(chunk)
        checksum = zlib.crc32(chunk, checksum)
        remaining -= len(chunk)
    tensor["crc32"] = checksum & 0xFFFFFFFF


def write_index(destination, tensors, layouts, tensor_index):
    destination.seek(tensor_index)
    for tensor in sorted(tensors.values(), key=lambda item: item["name"]):
        name = tensor["name"].encode("utf-8")
        destination.write(TENSOR.pack(len(name), tensor["dtype"], tensor["offset"],
                                      tensor["nbytes"], tensor["numel"], tensor["crc32"], 0, 0))
        destination.write(name)
        destination.write(b"\0" * (align_up(len(name), 8) - len(name)))
    expert_index = destination.tell()
    for layout in layouts:
        destination.write(EXPERT.pack(layout["layer"], layout["n_experts"],
                                      layout["base"], layout["stride"],
                                      *layout["offsets"], *layout["lengths"],
                                      layout["fmt"], layout["group_size"]))
    return expert_index, destination.tell()


def verify_payload(path: Path):
    native = read_native(path)
    with path.open("rb", buffering=0) as stream:
        for number, tensor in enumerate(native["tensors"], 1):
            stream.seek(tensor["offset"])
            remaining = tensor["nbytes"]
            checksum = 0
            while remaining:
                chunk = stream.read(min(COPY_CHUNK, remaining))
                if not chunk:
                    raise ValueError(f"short native tensor payload: {tensor['name']}")
                checksum = zlib.crc32(chunk, checksum)
                remaining -= len(chunk)
            if checksum & 0xFFFFFFFF != tensor["crc32"]:
                raise ValueError(f"checksum mismatch for {tensor['name']}")
            if number % 10000 == 0:
                print(f"verified {number}/{native['tensor_count']} tensors", flush=True)
    return native


def fsync_directory(path: Path):
    try:
        directory_fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def pack(model: Path, output: Path, *, delete_source=False, quick=False, force=False):
    model = model.resolve()
    output = output.resolve()
    temporary = output.with_name(output.name + ".tmp")
    if output.exists() and not force:
        raise ValueError(f"output already exists: {output} (use --force to replace)")
    if temporary.exists():
        if not force:
            raise ValueError(f"temporary output exists: {temporary} (use --force to restart)")
        temporary.unlink()
    if delete_source and quick:
        raise ValueError("--delete-source requires full payload verification")

    shards, tensors = read_sources(model)
    normal, grouped, layouts, data_end = plan_layout(model, tensors)
    print(f"packing {len(tensors)} tensors from {len(shards)} shards", flush=True)
    handles = {}
    try:
        with temporary.open("w+b", buffering=0) as destination:
            destination.truncate(ALIGNMENT)
            for tensor in normal:
                copy_tensor(handles, destination, tensor)
            for layout in layouts:
                print(f"packing layer {layout['layer']}: {layout['n_experts']} experts, "
                      f"{layout['stride'] / 1e6:.2f} MB/record", flush=True)
                for expert in range(layout["n_experts"]):
                    for field in range(6):
                        copy_tensor(handles, destination, grouped[layout["layer"]][expert][field])
            tensor_index = align_up(data_end, 8)
            expert_index, file_size = write_index(destination, tensors, layouts, tensor_index)
            destination.truncate(file_size)
            destination.seek(0)
            destination.write(HEADER.pack(MAGIC, VERSION, 0, file_size, len(tensors),
                                          len(layouts), tensor_index, expert_index, ALIGNMENT))
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    finally:
        for handle in handles.values():
            handle.close()

    if quick:
        native = read_native(temporary)
    else:
        print("verifying native payload checksums", flush=True)
        native = verify_payload(temporary)
    os.replace(temporary, output)
    fsync_directory(output.parent)
    if delete_source:
        for shard in shards:
            shard.unlink()
        fsync_directory(output.parent)
        print(f"removed {len(shards)} verified source shards", flush=True)
    print(f"native model ready: {output} ({native['file_size'] / 1e9:.2f} GB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model directory containing quantized safetensors")
    parser.add_argument("--output", default=None, help="output path (default: <model>/model.coli)")
    parser.add_argument("--delete-source", action="store_true",
                        help="delete safetensors only after full checksum verification")
    parser.add_argument("--quick", action="store_true", help="skip the full output checksum pass")
    parser.add_argument("--force", action="store_true", help="replace an existing native image")
    arguments = parser.parse_args()
    model = Path(arguments.model)
    output = Path(arguments.output) if arguments.output else model / "model.coli"
    try:
        pack(model, output, delete_source=arguments.delete_source,
             quick=arguments.quick, force=arguments.force)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
