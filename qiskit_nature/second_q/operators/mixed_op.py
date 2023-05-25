from __future__ import annotations

from .sparse_label_op import SparseLabelOp
from .fermionic_op import FermionicOp
from .spin_op import SpinOp
from typing import cast

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping

from qiskit.quantum_info.operators.mixins import (
    AdjointMixin,
    GroupMixin,
    LinearMixin,
    TolerancesMixin,
)

from copy import deepcopy
import numpy as np

class MixedOp(LinearMixin):
    def __init__(self, data=dict[tuple, list[tuple[SparseLabelOp, float]]]):
        self.data = data

    def __repr__(self) -> str:
        out_str = ""
        for key, oplist in self.data.items():
            for hspace in key:
                out_str += hspace
            out_str += " : "
            for op_tuple in oplist:
                out_str += "["
                for op in op_tuple:
                    out_str += f"{repr(op)} "
                out_str += "]"
            out_str += "\n"

        return out_str

    def copy(self):
        return deepcopy(self)

    def keys_no_duplicate(self):
        all_keys = ()
        for key in self.data.keys():
            all_keys += key
        return tuple(set(all_keys))

    def compose(self, other: MixedOp) -> MixedOp:
        new_data = {}
        for key1, op_tuple1 in self.data.items():
            for key2, op_tuple2 in other.data.items():
                new_tuple = []
                for op1 in op_tuple1:
                    for op2 in op_tuple2:
                        new_tuple.append((op1[0] * op2[0],) + op1[1:] + op2[1:])
                        #TODO: Add simplification if the same hilbert space appears twice.
                new_data[key1 + key2] = new_tuple
        return MixedOp(new_data)

    def _add(self, other: MixedOp, qargs: None = None) -> MixedOp:
        new_data = deepcopy(self.data)
        for key in other.data.keys():
            if key in new_data.keys():
                new_data[key] += other.data[key]
            else:
                new_data[key] = other.data[key]
        return MixedOp(new_data)

    def _multiply(self, other: float):
        new_data = deepcopy(self.data)
        for key, list_op in new_data.items():
            for k, op_k in enumerate(list_op):
                new_data[key][k] = (op_k[0] * other,) + op_k[1:]
        return MixedOp(new_data)
