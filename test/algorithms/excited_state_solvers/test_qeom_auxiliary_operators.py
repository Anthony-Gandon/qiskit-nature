# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Numerical qEOM excited states calculation """

import unittest

from qiskit.algorithms.optimizers import SLSQP
from qiskit.opflow import PauliExpectation, MatrixExpectation
from test import QiskitNatureTestCase
import numpy as np
from qiskit import BasicAer, Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver, NumPyEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver, VQEUCCFactory
from qiskit_nature.algorithms.excited_states_solvers import QEOM
from typing import List, Union, Optional, Tuple, Dict, cast
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from qiskit.opflow import PauliExpectation, AerPauliExpectation, MatrixExpectation
import numpy as np
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import SLSQP

from qiskit_nature import settings

settings.dict_aux_operators = True
from qiskit_nature.drivers import UnitsType
from qiskit_nature.drivers.second_quantization import PySCFDriver
from qiskit_nature.mappers.second_quantization import (
    BravyiKitaevMapper,
    JordanWignerMapper,
    ParityMapper,
)
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.algorithms import (
    GroundStateEigensolver,
    VQEUCCFactory,
    NumPyEigensolverFactory,
    ExcitedStatesEigensolver,
    QEOM,
)
import qiskit_nature.optionals as _optionals


class TestNumericalQEOMESCCalculation(QiskitNatureTestCase):
    """Test Numerical qEOM excited states calculation"""

    @unittest.skipIf(not _optionals.HAS_PYSCF, "pyscf not available.")
    def setUp(self):
        super().setUp()

        # algorithm_globals.random_seed = 8
        self.driver = PySCFDriver(
            atom="H .0 .0 .0; H .0 .0 0.75",
            unit=UnitsType.ANGSTROM,
            charge=0,
            spin=0,
            basis="sto3g",
        )

        self.reference_energies = [
            -1.8427016,
            -1.8427016 + 0.5943372,
            -1.8427016 + 0.95788352,
            -1.8427016 + 1.5969296,
        ]
        self.qubit_converter = QubitConverter(JordanWignerMapper())
        self.electronic_structure_problem = ElectronicStructureProblem(self.driver)

        self.qubit_converter = QubitConverter(
            mapper=JordanWignerMapper(), two_qubit_reduction=False, z2symmetry_reduction=None
        )

        self.optimizer = SLSQP(maxiter=100, disp=False)

    def test_statevector_simulator_matrix_expectation(self):
        backend = Aer.get_backend("statevector_simulator")
        quantum_instance = QuantumInstance(backend=backend)
        expectation = MatrixExpectation()

        vqe_solver = VQEUCCFactory(
            quantum_instance=quantum_instance,
            optimizer=self.optimizer,
            initial_point=None,
            expectation=expectation,
        )

        gsc1 = GroundStateEigensolver(qubit_converter=self.qubit_converter, solver=vqe_solver)

        qeom_solver = QEOM(gsc1, "sd")
        results = qeom_solver.solve(self.electronic_structure_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_energies[idx], energy, places=4)

    def test_statevector_simulator_pauli_expectation(self):
        backend = Aer.get_backend("statevector_simulator")
        quantum_instance = QuantumInstance(backend=backend, shots=10000)
        expectation = PauliExpectation()

        vqe_solver = VQEUCCFactory(
            quantum_instance=quantum_instance,
            optimizer=self.optimizer,
            initial_point=None,
            expectation=expectation,
        )

        gsc1 = GroundStateEigensolver(qubit_converter=self.qubit_converter, solver=vqe_solver)

        qeom_solver = QEOM(gsc1, "sd")
        results = qeom_solver.solve(self.electronic_structure_problem)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_energies[idx], energy, places=4)

    def test_qasm_simulator_pauli_expectation(self):
        backend = Aer.get_backend("qasm_simulator")
        quantum_instance = QuantumInstance(backend=backend)
        expectation = AerPauliExpectation()

        vqe_solver = VQEUCCFactory(
            quantum_instance=quantum_instance,
            optimizer=self.optimizer,
            initial_point=None,
            expectation=expectation,
        )

        gsc1 = GroundStateEigensolver(qubit_converter=self.qubit_converter, solver=vqe_solver)

        qeom_solver = QEOM(gsc1, "sd")
        results = qeom_solver.solve(self.electronic_structure_problem)

        print(results.computed_energies)

        for idx, energy in enumerate(self.reference_energies):
            self.assertAlmostEqual(results.computed_energies[idx], energy, places=4)


if __name__ == "__main__":
    unittest.main()
