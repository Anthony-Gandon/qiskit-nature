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
    """Mapper of Spin Operator to Qubit Operator"""

    def _map_single(self, position_ii, mapper_ii, coef_ii, operator_ii):
        print((position_ii, mapper_ii, coef_ii, operator_ii))

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

        for key, item in mixed_op.data.items():
            operator_length = len(item)
            print(operator_length)
            for ii in range(operator_length):
                position_ii = ordering[key[ii]]
                mapper_ii = mappers[str(type(item[ii][0]))]
                coef_ii = item[ii][0]
                operator_ii = item[ii][1]

                self._map_single(position_ii, mapper_ii, coef_ii, operator_ii)




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




