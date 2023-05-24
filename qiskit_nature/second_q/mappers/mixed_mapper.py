# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Spin Mapper."""

from __future__ import annotations

from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp

from qiskit_nature.second_q.operators import SpinOp, MixedOp, MixedOp2

from .qubit_mapper import ListOrDictType, QubitMapper


class MixedMapper2(QubitMapper):
    """Mapper of Mixed Operator to Qubit Operator"""

    def _map_single(self, position_ii, mapper_ii, coef_ii, operator_ii):
        print((coef_ii, operator_ii))
        print("\t", (position_ii, type(mapper_ii).__name__, operator_ii.register_length))

    def map(
        self,
        mixed_op: MixedOp,
        ordering: str,
        mappers: list[QubitMapper],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp:
        
        all_keys = list(mixed_op.data.keys())
        print(all_keys)

        for key, operator_list in mixed_op.data.items():
            for operator_tuple in operator_list:
                for index, op in enumerate(operator_tuple[1:]):
                    position_ii = ordering[key[index]]
                    mapper_ii = mappers[key[index]]
                    self._map_single(position_ii, mapper_ii, operator_tuple[0], op)




class MixedMapper(QubitMapper):
    """Mapper of Spin Operator to Qubit Operator"""

    def map(
        self,
        mixed_op: MixedOp,
        ordering: str,
        mappers: list[QubitMapper],
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp:

        # Run through the sum of terms
        for op in mixed_op.data:
            # Run through the hilbert spaces
            for key, item in op.items():
                position_ii = ordering[key]
                mapper_ii = mappers[str(type(item[0]))]
                coef_ii = item[0]
                operator_ii = item[1]

                print((position_ii, mapper_ii, coef_ii, operator_ii))




