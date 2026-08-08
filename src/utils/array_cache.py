"""Compact, mutation-safe fingerprints for cached NumPy model weights."""

from __future__ import annotations

import hashlib
import weakref

import numpy as np


class ArrayFingerprints:
    """Hash an array once, then reuse its key by weak object identity.

    Model weight vectors are immutable after training/loading. ``bind`` enforces
    that invariant, computes the content key, and retains only a weak reference;
    repeated encode/decode calls therefore avoid hashing megabytes of weights.
    """

    def __init__(self):
        self._by_identity = {}

    @staticmethod
    def _content_key(weights):
        array = np.ascontiguousarray(np.asarray(weights, dtype=np.float32)).ravel()
        return array.size, hashlib.sha256(memoryview(array).cast("B")).digest()

    def key(self, weights):
        if isinstance(weights, np.ndarray):
            entry = self._by_identity.get(id(weights))
            if entry is not None and entry[0]() is weights:
                return entry[1]
        return self._content_key(weights)

    def bind(self, weights):
        key = self._content_key(weights)
        if isinstance(weights, np.ndarray):
            weights.setflags(write=False)
            object_id = id(weights)

            def discard(reference, *, expected_id=object_id):
                entry = self._by_identity.get(expected_id)
                if entry is not None and entry[0] is reference:
                    self._by_identity.pop(expected_id, None)

            self._by_identity[object_id] = (weakref.ref(weights, discard), key)
        return key
