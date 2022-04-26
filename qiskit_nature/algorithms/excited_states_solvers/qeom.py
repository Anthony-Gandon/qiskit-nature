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

"""The calculation of excited states via the qEOM algorithm"""

from typing import List, Union, Optional, Tuple, Dict, cast
import itertools
import logging
import sys

import numpy as np
from scipy import linalg
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import AlgorithmResult
from qiskit.opflow.expectations.expectation_base import ExpectationBase
from qiskit.opflow import (
    Z2Symmetries,
    commutator,
    double_commutator,
    PauliSumOp,
    StateFn,
    MatrixExpectation,
)
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.opflow import PauliOp
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.state_fns import OperatorStateFn

from qiskit.algorithms import eval_observables
from qiskit.algorithms import EigensolverResult

from qiskit_nature import ListOrDictType
from qiskit_nature.operators.second_quantization import SecondQuantizedOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult
from .excited_states_solver import ExcitedStatesSolver
from ..ground_state_solvers import GroundStateSolver

logger = logging.getLogger(__name__)


class QEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm"""

    def __init__(
            self,
            ground_state_solver: GroundStateSolver,
            excitations: Union[str, List[List[int]]] = "sd",
    ) -> None:
        """
        Args:
            ground_state_solver: a GroundStateSolver object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise a list of custom excitations can directly be provided.
        """
        self._gsc = ground_state_solver
        self.excitations = excitations

        self._untapered_qubit_op_main: PauliSumOp = None

    @property
    def excitations(self) -> Union[str, List[List[int]]]:
        """Returns the excitations to be included in the eom pseudo-eigenvalue problem."""
        return self._excitations

    @excitations.setter
    def excitations(self, excitations: Union[str, List[List[int]]]) -> None:
        """The excitations to be included in the eom pseudo-eigenvalue problem. If a string then
        all excitations of given type will be used. Otherwise a list of custom excitations can
        directly be provided."""
        if isinstance(excitations, str) and excitations not in ["s", "d", "sd"]:
            raise ValueError(
                "Excitation type must be s (singles), d (doubles) or sd (singles and doubles)"
            )
        self._excitations = excitations

    def solve(
            self,
            problem: BaseProblem,
            aux_operators: Optional[ListOrDictType[SecondQuantizedOp]] = None,
    ) -> EigenstateResult:
        """Run the excited-states calculation.

        Construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """

        if aux_operators is not None:
            logger.warning(
                "With qEOM the auxiliary operators can currently only be "
                "evaluated on the ground state."
            )

        # 1. Prepare the excitation operators
        second_q_ops = problem.second_q_ops()

        aux_second_q_ops: ListOrDictType[SecondQuantizedOp]
        if isinstance(second_q_ops, list):
            main_second_q_op = second_q_ops[0]
            aux_second_q_ops = second_q_ops[1:]
        elif isinstance(second_q_ops, dict):
            name = problem.main_property_name
            main_second_q_op = second_q_ops.pop(name, None)
            if main_second_q_op is None:
                raise ValueError(
                    f"The main `SecondQuantizedOp` associated with the {name} property cannot be "
                    "`None`."
                )
            aux_second_q_ops = second_q_ops

        _ = self._gsc.qubit_converter.convert(
            main_second_q_op,
            num_particles=problem.num_particles,
            sector_locator=problem.symmetry_sector_locator,
        )

        # 2. Prepare matrix operators
        self._untapered_qubit_op_main = self._gsc.qubit_converter.convert_only(
            main_second_q_op,
            num_particles=problem.num_particles,
        )
        matrix_operators_dict, size = self._prepare_matrix_operators(problem)

        if isinstance(aux_operators, Dict) and aux_operators is not None:
            matrix_operators_dict.update(aux_operators)  # IN PLACE

        if isinstance(aux_operators, List) and aux_operators is not None:
            matrix_operators_dict = np.append(
                aux_operators, matrix_operators_dict.values()
            )  # IN PLACE

        # 3. Run ground state calculation
        groundstate_result = self._gsc.solve(problem, matrix_operators_dict)
        measurement_results = groundstate_result.aux_operator_eigenvalues[0]

        # 4. Post-process ground_state_result to construct eom matrices
        (
            m_mat,
            v_mat,
            q_mat,
            w_mat,
            m_mat_std,
            v_mat_std,
            q_mat_std,
            w_mat_std,
        ) = self._build_eom_matrices(measurement_results, size)

        # 5. solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self._compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs
        qeom_result.excitation_energies = energy_gaps
        qeom_result.m_matrix = m_mat
        qeom_result.v_matrix = v_mat
        qeom_result.q_matrix = q_mat
        qeom_result.w_matrix = w_mat
        qeom_result.m_matrix_std = m_mat_std
        qeom_result.v_matrix_std = v_mat_std
        qeom_result.q_matrix_std = q_mat_std
        qeom_result.w_matrix_std = w_mat_std

        eigenstate_result = EigenstateResult()
        eigenstate_result.eigenstates = groundstate_result.eigenstates
        eigenstate_result.aux_operator_eigenvalues = groundstate_result.aux_operator_eigenvalues
        eigenstate_result.raw_result = qeom_result

        eigenstate_result.eigenenergies = np.append(
            groundstate_result.eigenenergies,
            np.asarray([groundstate_result.eigenenergies[0] + gap for gap in energy_gaps]),
        )

        result = problem.interpret(eigenstate_result)

        return result

    def _build_all_commutators(
            self, hopping_operators: dict, type_of_commutativities: dict, size: int
    ) -> dict:
        """Building all commutators for Q, W, M, V matrices.

        Args:
            hopping_operators: all hopping operators based on excitations_list,
                key is the string of single/double excitation;
                value is corresponding operator.
            type_of_commutativities: if tapering is used, it records the commutativities of
                hopping operators with the
                Z2 symmetries found in the original operator.
            size: the number of excitations (size of the qEOM pseudo-eigenvalue problem)

        Returns:
            a dictionary that contains the operators for each matrix element
        """

        all_matrix_operators = {}

        mus, nus = np.triu_indices(size)

        def _build_one_sector(available_hopping_ops, untapered_op, z2_symmetries):

            to_be_computed_list = []
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]
                left_op = available_hopping_ops.get(f"E_{m_u}")
                right_op_1 = available_hopping_ops.get(f"E_{n_u}")
                right_op_2 = available_hopping_ops.get(f"Edag_{n_u}")
                to_be_computed_list.append((m_u, n_u, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(
                self._build_commutator_routine,
                to_be_computed_list,
                task_args=(untapered_op, z2_symmetries),
                num_processes=algorithm_globals.num_processes,
            )
            for result in results:
                m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result

                if q_mat_op is not None:
                    all_matrix_operators[f"q_{m_u}_{n_u}"] = q_mat_op
                if w_mat_op is not None:
                    all_matrix_operators[f"w_{m_u}_{n_u}"] = w_mat_op
                if m_mat_op is not None:
                    all_matrix_operators[f"m_{m_u}_{n_u}"] = m_mat_op
                if v_mat_op is not None:
                    all_matrix_operators[f"v_{m_u}_{n_u}"] = v_mat_op

        try:
            z2_symmetries = self._gsc.qubit_converter.z2symmetries
        except AttributeError:
            z2_symmetries = Z2Symmetries([], [], [])

        if not z2_symmetries.is_empty():
            combinations = itertools.product([1, -1], repeat=len(z2_symmetries.symmetries))
            for targeted_tapering_values in combinations:
                logger.info(
                    "In sector: (%s)",
                    ",".join([str(x) for x in targeted_tapering_values]),
                )
                # remove the excited operators which are not suitable for the sector

                available_hopping_ops = {}
                targeted_sector = np.asarray(targeted_tapering_values) == 1
                for key, value in type_of_commutativities.items():
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = hopping_operators[key]
                # untapered_qubit_op is a PauliSumOp and should not be exposed.
                _build_one_sector(
                    available_hopping_ops, self._untapered_qubit_op_main, z2_symmetries
                )

        else:
            # untapered_qubit_op is a PauliSumOp and should not be exposed.
            _build_one_sector(hopping_operators, self._untapered_qubit_op_main, z2_symmetries)

        return all_matrix_operators

    @staticmethod
    def _build_commutator_routine(
            params: List, operator: PauliSumOp, z2_symmetries: Z2Symmetries
    ) -> Tuple[int, int, PauliSumOp, PauliSumOp, PauliSumOp, PauliSumOp]:
        """Numerically computes the commutator / double commutator between operators.

        Args:
            params: list containing the indices of matrix element and the corresponding
                excitation operators
            operator: the hamiltonian
            z2_symmetries: z2_symmetries in case of tapering

        Returns:
            The indices of the matrix element and the corresponding qubit
            operator for each of the EOM matrices
        """
        m_u, n_u, left_op, right_op_1, right_op_2 = params
        if left_op is None or right_op_1 is None and right_op_2 is None:
            q_mat_op = None
            w_mat_op = None
            m_mat_op = None
            v_mat_op = None
        else:

            if right_op_1 is not None:
                # The sign which we use in the case of the double commutator is arbitrary. In
                # theory, one would choose this according to the nature of the problem (i.e.
                # whether it is fermionic or bosonic), but in practice, always choosing the
                # anti-commutator has proven to be more robust.
                q_mat_op = double_commutator(left_op, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = commutator(left_op, right_op_1)
                q_mat_op = None if len(q_mat_op) == 0 else q_mat_op
                w_mat_op = None if len(w_mat_op) == 0 else w_mat_op
            else:
                q_mat_op = None
                w_mat_op = None

            if right_op_2 is not None:
                # For explanations on the choice of commutation relation, please refer to the
                # comments above.
                m_mat_op = double_commutator(left_op, operator, right_op_2, sign=False)
                v_mat_op = commutator(left_op, right_op_2)
                m_mat_op = None if len(m_mat_op) == 0 else m_mat_op
                v_mat_op = None if len(v_mat_op) == 0 else v_mat_op
            else:
                m_mat_op = None
                v_mat_op = None

            if not z2_symmetries.is_empty():
                if q_mat_op is not None and len(q_mat_op) > 0:
                    q_mat_op = z2_symmetries.taper(q_mat_op)
                if w_mat_op is not None and len(w_mat_op) > 0:
                    w_mat_op = z2_symmetries.taper(w_mat_op)
                if m_mat_op is not None and len(m_mat_op) > 0:
                    m_mat_op = z2_symmetries.taper(m_mat_op)
                if v_mat_op is not None and len(v_mat_op) > 0:
                    v_mat_op = z2_symmetries.taper(v_mat_op)

        return m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op

    def _build_eom_matrices(
            self, gs_results: Dict[str, List[float]], size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
        """Constructs the M, V, Q and W matrices from the results on the ground state

        Args:
            gs_results: a ground state result object
            size: size of eigenvalue problem

        Returns:
            the matrices and their standard deviation
        """

        mus, nus = np.triu_indices(size)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0.0, 0.0, 0.0, 0.0

        # evaluate results
        for idx, _ in enumerate(mus):
            m_u = mus[idx]
            n_u = nus[idx]

            q_mat[m_u][n_u] = (
                gs_results[f"q_{m_u}_{n_u}"][0]
                if gs_results.get(f"q_{m_u}_{n_u}") is not None
                else q_mat[m_u][n_u]
            )
            w_mat[m_u][n_u] = (
                gs_results[f"w_{m_u}_{n_u}"][0]
                if gs_results.get(f"w_{m_u}_{n_u}") is not None
                else w_mat[m_u][n_u]
            )
            m_mat[m_u][n_u] = (
                gs_results[f"m_{m_u}_{n_u}"][0]
                if gs_results.get(f"m_{m_u}_{n_u}") is not None
                else m_mat[m_u][n_u]
            )
            v_mat[m_u][n_u] = (
                gs_results[f"v_{m_u}_{n_u}"][0]
                if gs_results.get(f"v_{m_u}_{n_u}") is not None
                else v_mat[m_u][n_u]
            )

            q_mat_std += (
                gs_results[f"q_{m_u}_{n_u}_std"][0]
                if gs_results.get(f"q_{m_u}_{n_u}_std") is not None
                else 0
            )
            w_mat_std += (
                gs_results[f"w_{m_u}_{n_u}_std"][0]
                if gs_results.get(f"w_{m_u}_{n_u}_std") is not None
                else 0
            )
            m_mat_std += (
                gs_results[f"m_{m_u}_{n_u}_std"][0]
                if gs_results.get(f"m_{m_u}_{n_u}_std") is not None
                else 0
            )
            v_mat_std += (
                gs_results[f"v_{m_u}_{n_u}_std"][0]
                if gs_results.get(f"v_{m_u}_{n_u}_std") is not None
                else 0
            )

        # these matrices are numpy arrays and therefore have the ``shape`` attribute
        q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
        w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

        q_mat = np.real(q_mat)
        w_mat = np.real(w_mat)
        m_mat = np.real(m_mat)
        v_mat = np.real(v_mat)

        q_mat_std = q_mat_std / float(size ** 2)
        w_mat_std = w_mat_std / float(size ** 2)
        m_mat_std = m_mat_std / float(size ** 2)
        v_mat_std = v_mat_std / float(size ** 2)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nW:=========================\n%s", w_mat)
        logger.debug("\nM:=========================\n%s", m_mat)
        logger.debug("\nV:=========================\n%s", v_mat)

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    @staticmethod
    def _compute_excitation_energies(
            m_mat: np.ndarray,
            v_mat: np.ndarray,
            q_mat: np.ndarray,
            w_mat: np.ndarray,
            pinv_sqrt_metric: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat : M matrices
            v_mat : V matrices
            q_mat : Q matrices
            w_mat : W matrices

        Returns:
            1-D vector stores all energy gap to reference state
            2-D array storing the X and Y expansion coefficients
        """
        logger.debug("Diagonalizing qeom matrices for excited states...")
        a_mat = np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.matrixlib.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])

        if pinv_sqrt_metric is None:
            res = linalg.eig(a_mat, b_mat)
            eigenvectors_fixed = res[1]
        else:
            a_mat_new_metric = a_mat @ pinv_sqrt_metric
            b_mat_new_metric = b_mat @ pinv_sqrt_metric
            res = linalg.eig(a_mat_new_metric, b_mat_new_metric)
            eigenvectors_fixed = pinv_sqrt_metric @ res[1]

        # temp = eigenvectors_fixed.T.conjugate() @ metric @ eigenvectors_fixed
        # print("Test Normalisation: ", np.diag(temp))

        # convert nan value into 0
        res[0][np.where(np.isnan(res[0]))] = 0.0
        # Only the positive eigenvalues are physical. We need to take care
        # though of very small values
        # should an excited state approach ground state. Here the small values
        # may be both negative or
        # positive. We should take just one of these pairs as zero. So to get the values we want we
        # sort the real parts and then take the upper half of the sorted values.
        # Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug("... %s", res[0])
        order = np.argsort(np.real(res[0]))
        w = np.real(res[0])[order]
        print(order)
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w[len(w) // 2:])
        # w[np.abs(w) < 1e-06] = 0
        excitation_energies_gap = w
        expansion_coefs = eigenvectors_fixed[:, order[:len(order) // 2]]
        # expansion_coefs = eigenvectors_fixed[:, order]
        return excitation_energies_gap, expansion_coefs

    @staticmethod
    def _compute_excited_operators_Odag_n(
            hopping_operators: Dict[str, PauliSumOp],
            hopping_operators_eval,
            expansion_coefs: np.ndarray,
    ) -> Dict[str, Dict[str, PauliSumOp]]:
        """

        Construct the excited states |n>

        Args:
            hopping_operators:
            hopping_operators_eval:
            expansion_coefs:
            size:
            aux_operators:
            num_qubits:

        Returns:

        """
        size = len(hopping_operators) // 2
        print('size', size)
        # Creates all the On and On^\dag operators
        general_excitation_operators = {}  # O(n)^\dag for n = 1,2,3,...,size
        general_excitation_operators_eval = {}  # O(n)^\dag for n = 1,2,3,...,size

        # operator_indices = list(itertools.chain(range(-size, 0), range(1, size+1)))
        operator_indices = list(range(1, size + 1))

        for n in range(0, len(operator_indices)):
            general_excitation_operators[f"Odag_{operator_indices[n]}"] = 0
            general_excitation_operators_eval[f"Odag_{operator_indices[n]}"] = 0
            for mu in range(0, size):
                de_excitation_op = hopping_operators.get(f"E_{mu}")
                excitation_op = hopping_operators.get(f"Edag_{mu}")
                de_excitation_eval = hopping_operators_eval.get(f"E_{mu}")[0]
                excitation_eval = hopping_operators_eval.get(f"Edag_{mu}")[0]

                general_excitation_operators[f"Odag_{operator_indices[n]}"] += (
                        complex(expansion_coefs[mu, n]) * de_excitation_op
                        - complex(expansion_coefs[mu + size, n]) * excitation_op
                )
                general_excitation_operators_eval[f"Odag_{operator_indices[n]}"] += (
                        complex(expansion_coefs[mu, n]) * de_excitation_eval
                        - complex(expansion_coefs[mu + size, n]) * excitation_eval
                )

            num_qubits = excitation_op.num_qubits
            general_excitation_operators[f"Odag_{operator_indices[n]}"] = (
                    (
                            general_excitation_operators[f"Odag_{operator_indices[n]}"]
                            - PauliOp(
                        Pauli("I" * num_qubits), general_excitation_operators_eval[f"Odag_{operator_indices[n]}"]
                    )
                    )
                    / (1 - general_excitation_operators_eval[f"Odag_{operator_indices[n]}"] ** 2) ** 0.5
            ).reduce()

            # general_excitation_operators[f"Odag_{n + 1}"] = general_excitation_operators[f"Odag_{n + 1}"].reduce()

        return general_excitation_operators

    def _compute_metric(self, hopping_operator_products_eval, size, normalize=True):

        # for i in range(size):
        #     print(f"Edag_{i}E_{i}", hopping_operator_products_eval.get(f"Edag_{i}E_{i}")[0])
        #     print(f"E_{i}Edag_{i}", hopping_operator_products_eval.get(f"E_{i}Edag_{i}")[0])

        A_mat = np.zeros((size, size), dtype=complex)
        B_mat = np.zeros((size, size), dtype=complex)
        C_mat = np.zeros((size, size), dtype=complex)
        D_mat = np.zeros((size, size), dtype=complex)
        for id1 in range(size):
            for id2 in range(size):
                A_mat[id1, id2] = hopping_operator_products_eval.get(f"Edag_{id1}E_{id2}")[0]
                B_mat[id1, id2] = -hopping_operator_products_eval.get(f"Edag_{id1}Edag_{id2}")[0]
                C_mat[id1, id2] = -hopping_operator_products_eval.get(f"E_{id1}E_{id2}")[0]
                D_mat[id1, id2] = hopping_operator_products_eval.get(f"E_{id1}Edag_{id2}")[0]

        # print("A_mat", A_mat)
        # print("B_mat", B_mat)
        # print("C_mat", C_mat)
        # print("D_mat", D_mat)
        # print()

        metric = np.matrixlib.bmat([[A_mat, B_mat], [C_mat, D_mat]])

        # print(np.around(np.real(metric), 3))
        if normalize:
            print("is_sym : ", (metric == metric.T).all())
            eigenvalue_metric = np.sort(linalg.eig(metric)[0])
            print("cholesky: ", eigenvalue_metric)
            minimum_eigenval = np.abs(eigenvalue_metric[0])
            sqrt_metric = linalg.cholesky(metric + 10 * minimum_eigenval * np.eye(2 * size))
            pinv_sqrt_metric = linalg.pinv(sqrt_metric)

        else:
            pinv_sqrt_metric = np.eye(2 * size)

        return pinv_sqrt_metric, metric

    def _conversion_aux_operators(
            self,
            problem: BaseProblem,
            aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ):
        # get the operator and auxiliary operators, and transform the provided auxiliary operators
        # note that ``aux_ops`` contains not only the transformed ``aux_operators`` passed by the
        # user but also additional ones from the transformation
        second_q_ops = problem.second_q_ops()
        aux_second_q_ops: ListOrDictType[SecondQuantizedOp]
        print(second_q_ops.keys())
        if isinstance(second_q_ops, list):
            main_second_q_op = second_q_ops[0]
            aux_second_q_ops = second_q_ops
        elif isinstance(second_q_ops, dict):
            name = problem.main_property_name
            main_second_q_op = second_q_ops.get(name, None)
            if main_second_q_op is None:
                raise ValueError(
                    f"The main `SecondQuantizedOp` associated with the {name} property cannot be "
                    "`None`."
                )
            aux_second_q_ops = second_q_ops

        self._untapered_qubit_op_main = self._gsc.qubit_converter.convert_only(
            main_second_q_op,
            num_particles=problem.num_particles,
        )
        num_qubits = self._untapered_qubit_op_main.num_qubits

        aux_ops = self._gsc.qubit_converter.convert_match(aux_second_q_ops)

        if aux_operators is not None:
            wrapped_aux_operators: ListOrDict[Union[SecondQuantizedOp, PauliSumOp]] = ListOrDict(
                aux_operators
            )
            for name_aux, aux_op in iter(wrapped_aux_operators):
                if isinstance(aux_op, SecondQuantizedOp):
                    converted_aux_op = self._qubit_converter.convert_match(aux_op, True)
                else:
                    converted_aux_op = aux_op
                if isinstance(aux_ops, list):
                    aux_ops.append(converted_aux_op)
                elif isinstance(aux_ops, dict):
                    if name_aux in aux_ops.keys():
                        raise QiskitNatureError(
                            f"The key '{name_aux}' is already taken by an internally constructed "
                            "auxiliary operator! Please use a different name for your custom "
                            "operator."
                        )
                    aux_ops[name_aux] = converted_aux_op

        # if the eigensolver does not support auxiliary operators, reset them
        if not self._gsc._solver.supports_aux_operators():
            aux_ops = None

        return aux_ops, num_qubits

    def _eval_additional_observables(
            self, hopping_operators, matrix_operators_dict, quantum_instance, circuit_state, expectation
    ):

        hopping_operator_products = {}
        for idx_left, op_left in hopping_operators.items():
            for idx_right, op_right in hopping_operators.items():
                hopping_operator_products[idx_left + idx_right] = (op_left @ op_right).reduce()

        hopping_operators_eval = eval_observables(
            quantum_instance, circuit_state, hopping_operators, expectation
        )

        hopping_operator_products_eval = eval_observables(
            quantum_instance, circuit_state, hopping_operator_products, expectation
        )

        matrix_operators_eval = eval_observables(
            quantum_instance, circuit_state, matrix_operators_dict, expectation
        )
        matrix_operators_eval = cast(Dict[str, List[float]], matrix_operators_eval)

        return hopping_operators_eval, hopping_operator_products_eval, matrix_operators_eval

    def _eval_all_aux_ops(
            self,
            excited_operators_Odag_n,
            aux_operators,
            groundstate,
            quantum_instance,
            expectation,
    ):
        # Creates all the On @ Aux @ On^\dag operators
        general_on_aux_on_dag_operators = {}
        for on_str, on_dag in excited_operators_Odag_n.items():
            if isinstance(aux_operators, dict):
                listordict_aux_op_on = {"Identity": (on_dag.adjoint() @ on_dag).reduce()}
                for aux_str, aux_op in aux_operators.items():
                    listordict_aux_op_on[aux_str] = (on_dag.adjoint() @ aux_op @ on_dag).reduce()

            if isinstance(aux_operators, list):
                listordict_aux_op_on = [(on_dag.adjoint() @ on_dag).reduce()]
                for aux_str, aux_op in enumerate(aux_operators):
                    listordict_aux_op_on.append((on_dag.adjoint() @ aux_op @ on_dag).reduce())

            general_on_aux_on_dag_operators[on_str] = listordict_aux_op_on

        aux_operator_eigenvalues_excited_states = []  # groundstate_result.aux_operator_eigenvalues
        excited_energies = []

        for index_n, on_aux_on_dag_operator in enumerate(general_on_aux_on_dag_operators.values()):
            not_normalized_eigenvalues = eval_observables(
                quantum_instance, groundstate, on_aux_on_dag_operator, expectation
            )

            if isinstance(not_normalized_eigenvalues, dict):
                # normalisation_value = not_normalized_eigenvalues.pop("Identity", (1.0, 0.0))
                eigenenergies_check = not_normalized_eigenvalues.pop("ElectronicEnergy", None)
                # print(index_n, normalisation_value)
                print(index_n, eigenenergies_check)
                aux_operator_eigenvalues_excited_states.append({})
                for op_name, op_eigenval in not_normalized_eigenvalues.items():
                    aux_operator_eigenvalues_excited_states[-1][op_name] = (
                        op_eigenval[0],  # / normalisation_value[0],
                        op_eigenval[1],
                    )

            if isinstance(not_normalized_eigenvalues, list):
                normalisation_value = not_normalized_eigenvalues.pop(0)
                eigenenergies_check = not_normalized_eigenvalues.pop(0)

                aux_operator_eigenvalues_excited_states.append([])
                for op_name, op_eigenval in enumerate(not_normalized_eigenvalues):
                    aux_operator_eigenvalues_excited_states[index_n + 1].append(
                        (op_eigenval[0] / normalisation_value[0], op_eigenval[1])
                    )
            excited_energies.append(eigenenergies_check)

        return excited_energies, aux_operator_eigenvalues_excited_states

    def _prepare_matrix_operators(self, problem, construct_true_eigenstates=False) -> Tuple[dict, int]:
        """Construct the excitation operators for each matrix element.

        Returns:
            a dictionary of all matrix elements operators and the number of excitations
            (or the size of the qEOM pseudo-eigenvalue problem)
        """
        data = problem.hopping_qeom_ops(self._gsc.qubit_converter, self._excitations)
        hopping_operators, type_of_commutativities, excitation_indices = data

        size = int(len(list(excitation_indices.keys())) // 2)

        hopping_operators_norm = {}
        for idx, op in hopping_operators.items():
            if not idx.startswith("Edag"):
                hopping_operators_norm[idx] = (
                        op * 1 / len(op.coeffs)
                )  # (2*idx.startswith("Edag")-1)
                hopping_operators_norm["Edag" + idx[1:]] = (
                        op.adjoint() * 1 / len(op.coeffs)
                )  # (2*idx.startswith("Edag")-1)

        eom_matrix_operators = self._build_all_commutators(
            hopping_operators_norm, type_of_commutativities, size
        )

        if construct_true_eigenstates:
            hopping_operator_products = {}
            for idx_left, op_left in hopping_operators_norm.items():
                for idx_right, op_right in hopping_operators_norm.items():
                    hopping_operator_products[idx_left + idx_right] = (op_left @ op_right).reduce()

            eom_matrix_operators.update(hopping_operators_norm.update(hopping_operator_products))

        return eom_matrix_operators, size

    def _prepare_all_operators(
            self,
            problem: BaseProblem,
            quantum_instance,
            groundstate,
            expectation,
            aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ):
        aux_operators_converted, num_qubits = self._conversion_aux_operators(problem, aux_operators)

        matrix_operators_dict, size, hopping_operators = self._prepare_matrix_operators(problem)
        print("***1***")
        (
            hopping_operators_eval,
            hopping_operator_products_eval,
            matrix_operators_eval,
        ) = self._eval_additional_observables(
            hopping_operators, matrix_operators_dict, quantum_instance, groundstate, expectation
        )

        pinv_sqrt_new_metric, metric = self._compute_metric(hopping_operator_products_eval, size)

        return (
            matrix_operators_eval,
            aux_operators_converted,
            size,
            num_qubits,
            pinv_sqrt_new_metric,
            metric,
            hopping_operators,
            hopping_operators_eval,
        )


class QEOMResult(AlgorithmResult):
    """The results class for the QEOM algorithm."""

    def __init__(self) -> None:
        super().__init__()
        self._ground_state_raw_result = None
        self._excitation_energies: Optional[np.ndarray] = None
        self._expansion_coefficients: Optional[np.ndarray] = None
        self._m_matrix: Optional[np.ndarray] = None
        self._v_matrix: Optional[np.ndarray] = None
        self._q_matrix: Optional[np.ndarray] = None
        self._w_matrix: Optional[np.ndarray] = None
        self._v_matrix_std: float = 0.0
        self._q_matrix_std: float = 0.0
        self._w_matrix_std: float = 0.0

    @property
    def ground_state_raw_result(self):
        """returns ground state raw result"""
        return self._ground_state_raw_result

    @ground_state_raw_result.setter
    def ground_state_raw_result(self, value) -> None:
        """sets ground state raw result"""
        self._ground_state_raw_result = value

    @property
    def excitation_energies(self) -> Optional[np.ndarray]:
        """returns the excitation energies (energy gaps)"""
        return self._excitation_energies

    @excitation_energies.setter
    def excitation_energies(self, value: np.ndarray) -> None:
        """sets the excitation energies (energy gaps)"""
        self._excitation_energies = value

    @property
    def expansion_coefficients(self) -> Optional[np.ndarray]:
        """returns the X and Y expansion coefficients"""
        return self._expansion_coefficients

    @expansion_coefficients.setter
    def expansion_coefficients(self, value: np.ndarray) -> None:
        """sets the X and Y expansion coefficients"""
        self._expansion_coefficients = value

    @property
    def m_matrix(self) -> Optional[np.ndarray]:
        """returns the M matrix"""
        return self._m_matrix

    @m_matrix.setter
    def m_matrix(self, value: np.ndarray) -> None:
        """sets the M matrix"""
        self._m_matrix = value

    @property
    def v_matrix(self) -> Optional[np.ndarray]:
        """returns the V matrix"""
        return self._v_matrix

    @v_matrix.setter
    def v_matrix(self, value: np.ndarray) -> None:
        """sets the V matrix"""
        self._v_matrix = value

    @property
    def q_matrix(self) -> Optional[np.ndarray]:
        """returns the Q matrix"""
        return self._q_matrix

    @q_matrix.setter
    def q_matrix(self, value: np.ndarray) -> None:
        """sets the Q matrix"""
        self._q_matrix = value

    @property
    def w_matrix(self) -> Optional[np.ndarray]:
        """returns the W matrix"""
        return self._w_matrix

    @w_matrix.setter
    def w_matrix(self, value: np.ndarray) -> None:
        """sets the W matrix"""
        self._w_matrix = value

    @property
    def m_matrix_std(self) -> float:
        """returns the M matrix standard deviation"""
        return self._m_matrix_std

    @m_matrix_std.setter
    def m_matrix_std(self, value: float) -> None:
        """sets the M matrix standard deviation"""
        self._m_matrix_std = value

    @property
    def v_matrix_std(self) -> float:
        """returns the V matrix standard deviation"""
        return self._v_matrix_std

    @v_matrix_std.setter
    def v_matrix_std(self, value: float) -> None:
        """sets the V matrix standard deviation"""
        self._v_matrix_std = value

    @property
    def q_matrix_std(self) -> float:
        """returns the Q matrix standard deviation"""
        return self._q_matrix_std

    @q_matrix_std.setter
    def q_matrix_std(self, value: float) -> None:
        """sets the Q matrix standard deviation"""
        self._q_matrix_std = value

    @property
    def w_matrix_std(self) -> float:
        """returns the W matrix standard deviation"""
        return self._w_matrix_std

    @w_matrix_std.setter
    def w_matrix_std(self, value: float) -> None:
        """sets the W matrix standard deviation"""
        self._w_matrix_std = value
