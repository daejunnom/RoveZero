import unittest

from rz_data.errors import DataError
from rz_data.serialization import digest, loads


class SerializationTests(unittest.TestCase):
    def test_duplicate_keys_are_rejected_even_nested(self):
        with self.assertRaises(DataError) as caught:
            loads('{"state":{"revision":1,"revision":2}}')
        self.assertEqual(caught.exception.code, "DuplicateKey")

    def test_non_finite_and_numeric_overflow_are_rejected(self):
        for data in ('NaN', 'Infinity', '-Infinity', '1e999'):
            with self.subTest(data=data), self.assertRaises(DataError):
                loads(data)

    def test_digest_is_independent_of_object_key_order(self):
        self.assertEqual(digest({"a": 1, "b": "한글"}),
                         digest({"b": "한글", "a": 1}))

    def test_digest_preserves_array_order(self):
        self.assertNotEqual(digest(["e2e4", "d2d4"]),
                            digest(["d2d4", "e2e4"]))

    def test_invalid_utf8_and_surrogate_are_rejected(self):
        for data in (b'"\xff"', '"\\ud800"'):
            with self.subTest(data=data), self.assertRaises(DataError):
                loads(data)
