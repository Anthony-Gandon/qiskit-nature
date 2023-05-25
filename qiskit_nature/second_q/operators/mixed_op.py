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
    def __init__(self, data=list[dict[str, tuple[SparseLabelOp, float]]]):
        self.data = data

    def __repr__(self) -> str:
        out_str = ""
        for d in self.data:
            for key, item in d.items():
                out_str += key + ": "
                out_str += f"{repr(item[0])} / coeff: {item[1]}"
                out_str += "\n"
            out_str += "\n"

        return out_str

    def __len__(self):
        return len(self.ops[FermionicOp]) + len(self.ops[SpinOp])

    def compose(self, other: MixedOp) -> MixedOp:
        def single_compose(d1, d2):
            return {**d1, **d2}

        new_data = []
        for d1 in self.data:
            for d2 in other.data:
                new_data.append(single_compose(d1, d2))
        return MixedOp(new_data)

    def _add(self, other: MixedOp, qargs: None = None) -> MixedOp:
        new_data = self.data + other.data

        return MixedOp(new_data)

    def _multiply(self, other: float):
        new_data = []
        for d in self.data:
            new_data.append({})
            for key, value in d.items():
                new_data[-1][key] = [value[0], value[1] * other]
        return MixedOp(new_data)