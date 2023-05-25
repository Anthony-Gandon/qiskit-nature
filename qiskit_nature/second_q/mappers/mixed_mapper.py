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

from qiskit_nature.second_q.operators import SpinOp, MixedOp

from .qubit_mapper import ListOrDictType, QubitMapper


class MixedMapper(QubitMapper):
    """Mapper of Mixed Operator to Qubit Operator"""

    def __init__(
        self,
        hilbert_spaces: dict[str:int],
        mappers: dict[str, QubitMapper],
    ):
        """TODO"""
        super().__init__()
        self.hilbert_spaces = hilbert_spaces
        self.mappers = mappers

    def _map_tuple_product(self, key, operator_tuple):
        """(5.0, Fop1, Fop2, I, I, Sop3)"""
        coefficient = operator_tuple[0]
        dict_mapped_op = {}
        print(self.hilbert_spaces.items())
        for key_temp, length in self.hilbert_spaces.items():
            dict_mapped_op[key_temp] = SparsePauliOp("I" * length, coeffs=coefficient)

        print("dict_mapped_op", dict_mapped_op)
        for index, op in enumerate(operator_tuple[1:]):
            key_char = key[index]
            mapper = self.mappers[key_char]
            dict_mapped_op[key_char] = mapper.map(op)
            # We can manage here the cases where multiple operators live in the same
            # Hilbert Space.
            # This could replace a simplify method.

        print("dict_mapped_op", dict_mapped_op)
        list_op = list(dict_mapped_op.values())
        product_op = list_op[0]
        for op in list_op[1:]:
            product_op = product_op.tensor(op)

        print("product_op", product_op)
        return product_op

    def _map_list_sum(self, key, operator_list):
        """[operator_tuple1, operator_tuple2, ...]"""
        final_op = 0
        for operator_tuple in operator_list:
            prod = self._map_tuple_product(key, operator_tuple)
            final_op += prod

        return final_op

    def map(
        self,
        mixed_op: MixedOp2,
        *,
        register_length: int | None = None,
    ) -> SparsePauliOp:
        """A MixedOp is a dict of list (to represent the sum of operators) of
        tuple (to represent the sum of operators).
        We run through the sum and then map the products with the necessary
        padding.
        """
        result = []

        for key, operator_list in mixed_op.data.items():
            result.append(self._map_list_sum(key, operator_list))

        return result
