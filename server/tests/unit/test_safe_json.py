"""Unit tests for SafeJSONEncoder / safe_json_dumps."""
from decimal import Decimal

import numpy as np

from server import SafeJSONEncoder, safe_json_dumps


class TestSafeJSONEncoder:
    def test_numpy_integer(self):
        assert safe_json_dumps({"n": np.int64(42)}) == '{"n": 42}'

    def test_numpy_float(self):
        data = safe_json_dumps({"n": np.float64(1.5)})
        assert '"n": 1.5' in data

    def test_numpy_array(self):
        assert safe_json_dumps({"a": np.array([1, 2, 3])}) == '{"a": [1, 2, 3]}'

    def test_decimal(self):
        assert safe_json_dumps({"d": Decimal("3.14")}) == '{"d": 3.14}'

    def test_bytes(self):
        assert safe_json_dumps({"b": b"hello"}) == '{"b": "hello"}'

    def test_nested_mixed(self):
        payload = {
            "score": np.float32(0.9),
            "ids": np.array([1, 2]),
            "price": Decimal("10.5"),
        }
        encoded = safe_json_dumps(payload)
        assert "0.89" in encoded  # float32 may not serialize as exact 0.9
        assert "[1, 2]" in encoded
        assert "10.5" in encoded

    def test_encoder_rejects_unknown_via_default(self):
        class Weird:
            pass

        encoder = SafeJSONEncoder()
        try:
            encoder.default(Weird())
            assert False, "expected TypeError"
        except TypeError:
            pass
