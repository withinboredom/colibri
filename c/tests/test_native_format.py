import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
sys.path.insert(0, str(C_DIR / "tools"))

from native_format import read_native
from pack_native import pack
from resource_plan import analyze_model


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [start, len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)


class NativeFormatTests(unittest.TestCase):
    def make_model(self, root):
        model = Path(root)
        (model / "config.json").write_text(json.dumps({
            "hidden_size": 4,
            "moe_intermediate_size": 3,
            "n_routed_experts": 2,
            "num_hidden_layers": 2,
        }))
        tensors = [("dense.weight", "F32", [2], struct.pack("<2f", 1.0, 2.0))]
        expected = {}
        for expert in range(2):
            for projection, output, input_ in (("gate_proj", 3, 4),
                                                ("up_proj", 3, 4),
                                                ("down_proj", 4, 3)):
                name = f"model.layers.1.mlp.experts.{expert}.{projection}.weight"
                weights = bytes((expert * 32 + index + 1) & 0xFF for index in range(output * input_))
                scales = struct.pack(f"<{output}f", *[expert + index + 0.5 for index in range(output)])
                tensors.append((name, "U8", [len(weights)], weights))
                tensors.append((name + ".qs", "F32", [output], scales))
                expected[name] = weights
                expected[name + ".qs"] = scales
        write_safetensors(model / "out-00000.safetensors", tensors)
        return model, expected

    def test_pack_builds_fixed_stride_expert_records(self):
        with tempfile.TemporaryDirectory() as directory:
            model, expected = self.make_model(directory)
            output = model / "model.coli"
            pack(model, output, quick=False)
            native = read_native(output)
            self.assertEqual(native["tensor_count"], 13)
            self.assertEqual(native["expert_count"], 1)
            layout = native["experts"][0]
            self.assertEqual(layout["layer"], 1)
            self.assertEqual(layout["n_experts"], 2)
            self.assertEqual(layout["stride"], 8192)
            self.assertEqual(layout["offsets"][3], 4096)
            by_name = {tensor["name"]: tensor for tensor in native["tensors"]}
            with output.open("rb") as stream:
                for name, data in expected.items():
                    stream.seek(by_name[name]["offset"])
                    self.assertEqual(stream.read(len(data)), data)

    def test_delete_source_requires_and_runs_full_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            model, _expected = self.make_model(directory)
            output = model / "model.coli"
            with self.assertRaises(ValueError):
                pack(model, output, delete_source=True, quick=True)
            pack(model, output, delete_source=True, quick=False)
            self.assertTrue(output.is_file())
            self.assertEqual(list(model.glob("*.safetensors")), [])
            self.assertEqual(read_native(output)["tensor_count"], 13)
            info = analyze_model(model)
            self.assertEqual(info["shards"], 1)
            self.assertEqual(info["expert_count"], 2)


if __name__ == "__main__":
    unittest.main()
