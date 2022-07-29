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

from typing import List, Union, Optional, Tuple, Dict
import itertools
import logging
import sys

import numpy as np
from scipy import linalg

from qiskit import QuantumCircuit
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import EigensolverResult
from qiskit.providers import Backend
from qiskit.opflow import (
    Z2Symmetries,
    commutator,
    double_commutator,
    PauliSumOp,
    StateFn,
    ExpectationBase,
    anti_commutator,
)
from qiskit.quantum_info import Pauli, Statevector

from qiskit.opflow import PauliOp

from qiskit.algorithms import eval_observables

from qiskit_nature.converters.second_quantization.utils import ListOrDict
from qiskit_nature import ListOrDictType
from qiskit_nature.operators.second_quantization import SecondQuantizedOp, FermionicOp
from qiskit_nature.problems.second_quantization import BaseProblem
from qiskit_nature.results import EigenstateResult
from .excited_states_solver import ExcitedStatesSolver
from ..ground_state_solvers import GroundStateSolver

logger = logging.getLogger(__name__)


class VQEEOM(ExcitedStatesSolver):
    """The calculation of excited states via the qEOM algorithm"""

    def __init__(
        self,
        ground_state_solver: GroundStateSolver,
        excitations: Union[str, List[List[int]]] = "sd",
        eval_aux_excited_states: Optional[bool] = False,
        quantum_instance: Optional[QuantumInstance] = None,
        expectation: Optional[ExpectationBase] = None,
        transition_amplitude_pairs: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            ground_state_solver: a GroundStateSolver object. The qEOM algorithm
                will use this ground state to compute the EOM matrix elements.
            excitations: The excitations to be included in the eom pseudo-eigenvalue problem.
                If a string ('s', 'd' or 'sd') then all excitations of the given type will be used.
                Otherwise, a list of custom excitations can directly be provided.
            eval_aux_excited_states: If True, the auxiliary operators are evaluated on the
                excited states. (NB: Requires extra measurements)
            quantum_instance: Must be provided if eval_aux_excited_states is True.
            expectation: Must be provided if eval_aux_excited_states is True.
            transition_amplitude_pairs: Specifies the transition amplitudes to evaluate.
        """
        self._gsc = ground_state_solver
        self.excitations = excitations
        self._untapered_qubit_op_main: PauliSumOp = None
        self._pre_tap_qubit_op_main: PauliSumOp = None

        self._eval_aux_excited_states: bool = eval_aux_excited_states
        self._quantum_instance: Optional[Union[QuantumInstance, Backend]] = quantum_instance
        self._expectation: Optional[ExpectationBase] = expectation
        self._transition_amplitude_pairs: Dict[List] = transition_amplitude_pairs

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

    @property
    def eval_aux_excited_states(self) -> bool:
        """Returns the eval_aux_excited_states boolean."""
        return self._eval_aux_excited_states

    @eval_aux_excited_states.setter
    def eval_aux_excited_states(self, eval_aux_excited_states: Optional[bool]) -> bool:
        """Sets the eval_aux_excited_states boolean."""
        self._eval_aux_excited_states = eval_aux_excited_states

    @property
    def quantum_instance(self) -> QuantumInstance:
        """Returns the Quantum instance."""
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Optional[QuantumInstance]):
        """Sets the Quantum instance."""
        self._quantum_instance = quantum_instance

    @property
    def expectation(self) -> ExpectationBase:
        """Returns the expectation."""
        return self._expectation

    @expectation.setter
    def expectation(self, expectation: Optional[ExpectationBase]):
        """Sets the expectation."""
        self._expectation = expectation

    @property
    def transition_amplitude_pairs(self) -> Dict:
        """Returns the transition amplitudes specifications."""
        return self._transition_amplitude_pairs

    @transition_amplitude_pairs.setter
    def transition_amplitude_pairs(self, transition_amplitude_pairs: Optional[Dict]):
        """Sets the eval_aux_excited_states boolean."""
        self._transition_amplitude_pairs = transition_amplitude_pairs

    @property
    def solver(self):
        return self._gsc.solver

    @property
    def qubit_converter(self):
        return self._gsc.qubit_converter

    def get_qubit_operators(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[Union[SecondQuantizedOp, PauliSumOp]]] = None,
    ) -> Tuple[PauliSumOp, Optional[ListOrDictType[PauliSumOp]]]:
        return self._gsc.get_qubit_operators(problem, aux_operators)

    def solve(
        self,
        problem: BaseProblem,
        aux_operators: Optional[ListOrDictType[SecondQuantizedOp]] = None,
    ) -> EigenstateResult:

        pre_tap_aux_ops = self._prepare_second_q_ops(problem, aux_operators)

        # 2. Prepare the matrix operators.
        # If a z2symmetry is provided, the operators are the partially tapered.
        (
            matrix_operators_dict,
            hopping_ops_pre_tap,
            size,
        ) = self._prepare_matrix_operators(problem)

        # Custom auxiliary operators
        pre_tap_aux_ops_custom = ListOrDict()
        for aux_name, aux_op in pre_tap_aux_ops:
            if aux_name not in problem.second_q_ops().keys():
                pre_tap_aux_ops_custom[aux_name] = aux_op

        tapered_aux_ops_custom = self.qubit_converter._symmetry_reduce_no_clifford(
            pre_tap_aux_ops_custom, True
        )

        # 3. Run ground state calculation with all the matrix elements as auxiliaries
        groundstate_result = self._gsc.solve(problem, tapered_aux_ops_custom)

        # Tries to retrieve the ansatz as a circuit and sets its parameters to the one
        # given by the GroundStateEigenSolver.
        # If the GrounsStateEigenSolver is not VQE then we take the eigenstate output.
        # This is needed for example for StateVector simulations with NumpyMinimumEigenSolver.
        if self.solver is not None and hasattr(self.solver, "ansatz"):
            bound_ansatz = self.solver.ansatz.assign_parameters(
                groundstate_result.raw_result.optimal_point
            )
        else:
            bound_ansatz = groundstate_result.eigenstates[0]

        (
            energies,
            expansion_coefs,
            circuit_dict_diag,
            circuit_dict_antidiag,
        ) = self._circuit_preparation(bound_ansatz)

        # 8. Prepares results if self._eval_aux_excited_states is False
        excited_eigenenergies = np.asarray(energies)
        eigenenergies = np.append(groundstate_result.eigenenergies, excited_eigenenergies)

        if not isinstance(groundstate_result.eigenstates[0], StateFn):
            eigenstates = [StateFn(groundstate_result.eigenstates[0])]
        else:
            eigenstates = [groundstate_result.eigenstates[0]]

        aux_operator_eigenvalues = groundstate_result.aux_operator_eigenvalues
        transition_amplitudes = []

        # 9. Evaluation of auxiliaries on excited states if self._eval_aux_excited_states is True
        if self._eval_aux_excited_states:
            aux_operator_eigenvalues_excited_states = self._eval_all_aux_ops(
                expansion_coefs, pre_tap_aux_ops, circuit_dict_diag, circuit_dict_antidiag, size
            )

            print(aux_operator_eigenvalues)
            for aux_op_str, aux_op in aux_operator_eigenvalues[0].items():
                if aux_op_str in aux_operator_eigenvalues_excited_states.keys():
                    aux_operator_eigenvalues_excited_states[aux_op_str] = np.append(
                        np.array([aux_operator_eigenvalues[0][aux_op_str][0]]),
                        aux_operator_eigenvalues_excited_states[aux_op_str],
                    )

            aux_operator_eigenvalues = aux_operator_eigenvalues_excited_states.copy()
            print(aux_operator_eigenvalues)
        # 10. Refactor the results
        qeom_result = QEOMResult()
        qeom_result.ground_state_raw_result = groundstate_result.raw_result
        qeom_result.expansion_coefficients = expansion_coefs

        qeom_result.eigenstates = eigenstates
        qeom_result.eigenvalues = eigenenergies
        qeom_result.transition_amplitudes = transition_amplitudes
        qeom_result.aux_operator_eigenvalues = aux_operator_eigenvalues

        result = problem.interpret(qeom_result)
        return result

    def _build_all_commutators(
        self,
        convert_hopping_ops: dict,
        type_of_commutativities: dict,
        size: int,
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
            z2_symmetries = self.qubit_converter.z2symmetries
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
                # print("targeted_sector", targeted_sector)
                for key, value in type_of_commutativities.items():
                    value = np.asarray(value)
                    # print(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = convert_hopping_ops[key]
                # untapered_qubit_op is a PauliSumOp and should not be exposed.
                # print("available", available_hopping_ops)

                _build_one_sector(available_hopping_ops, self._pre_tap_qubit_op_main, z2_symmetries)

        else:
            # untapered_qubit_op is a PauliSumOp and should not be exposed.
            _build_one_sector(convert_hopping_ops, self._untapered_qubit_op_main, z2_symmetries)

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
                q_mat_op = -double_commutator(left_op, operator, right_op_1, sign=False)
                # In the case of the single commutator, we are always interested in the energy
                # difference of two states. Thus, regardless of the problem's nature, we will
                # always use the commutator.
                w_mat_op = -commutator(left_op, right_op_1)
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
                    q_mat_op = z2_symmetries.taper_no_clifford(q_mat_op)
                if w_mat_op is not None and len(w_mat_op) > 0:
                    w_mat_op = z2_symmetries.taper_no_clifford(w_mat_op)
                if m_mat_op is not None and len(m_mat_op) > 0:
                    m_mat_op = z2_symmetries.taper_no_clifford(m_mat_op)
                if v_mat_op is not None and len(v_mat_op) > 0:
                    v_mat_op = z2_symmetries.taper_no_clifford(v_mat_op)

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

        if isinstance(gs_results, List):
            all_matrix_operators = np.concatenate(
                [
                    [
                        f"q_{mus[uu]}_{nus[uu]}",
                        f"w_{mus[uu]}_{nus[uu]}",
                        f"m_{mus[uu]}_{nus[uu]}",
                        f"v_{mus[uu]}_{nus[uu]}",
                    ]
                    for uu, _ in enumerate(mus)
                ]
            )
            gs_results = dict(zip(all_matrix_operators, gs_results[-len(all_matrix_operators) :]))

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
        w_mat = w_mat - w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T.conj() - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T.conj() - np.identity(v_mat.shape[0]) * v_mat

        # print(np.real(q_mat))
        # print(np.real(w_mat))
        # print(np.real(m_mat))
        # print(np.real(v_mat))

        q_mat = np.real(q_mat)
        w_mat = np.real(w_mat)
        m_mat = np.real(m_mat)
        v_mat = np.real(v_mat)

        q_mat_std = q_mat_std / float(size**2)
        w_mat_std = w_mat_std / float(size**2)
        m_mat_std = m_mat_std / float(size**2)
        v_mat_std = v_mat_std / float(size**2)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nW:=========================\n%s", w_mat)
        logger.debug("\nM:=========================\n%s", m_mat)
        logger.debug("\nV:=========================\n%s", v_mat)

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    def _compute_excitation_energies(
        self, m_mat: np.ndarray, v_mat: np.ndarray, q_mat: np.ndarray, w_mat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        a_mat = np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T]])
        b_mat = np.matrixlib.bmat([[v_mat, w_mat], [w_mat.T.conj(), -v_mat.T]])
        res = linalg.eig(a_mat, b_mat)
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
        logger.debug("Without filtering +/- %s", np.real(res[0]))
        order = np.argsort(np.real(res[0]))  # [:len(res[0])//2:][::-1]
        w = np.real(res[0])[order]
        logger.debug("Order real parts %s", order)
        logger.debug("Sorted real parts %s", w)
        w = np.abs(w)
        w[np.abs(w) < 1e-06] = 0
        excitation_energies_gap = w
        expansion_coefs = res[1][:, order]
        commutator_metric = expansion_coefs.T.conjugate() @ b_mat @ expansion_coefs

        logger.debug("Build commutator metric %s", np.abs(commutator_metric))

        return excitation_energies_gap, expansion_coefs, commutator_metric

    def _eval_all_aux_ops(
        self, expansion_coefs, pre_tap_aux_ops, circuit_dict_diag, circuit_dict_antidiag, size
    ):

        # Creates all the On @ Aux @ On^\dag operators
        results = {}
        for idx, circuit in circuit_dict_diag.items():
            value = eval_observables(
                self.quantum_instance, circuit, pre_tap_aux_ops, self.expectation
            )
            results[idx] = {}
            for idval, val in value.items():
                results[idx][idval] = val[0]

        for idx_1, excitation_1 in circuit_dict_antidiag.items():
            if idx_1[0] < idx_1[1]:
                # print(idx_1, idx_2)
                value = eval_observables(
                    self.quantum_instance,
                    circuit_dict_antidiag[(idx_1[0], idx_1[1])],
                    pre_tap_aux_ops,
                    self.expectation,
                )
                print("value", value)
                results[idx_1] = {}
                for idval, val in value.items():
                    results[idx_1][idval] = (
                        val[0]
                        - 0.5 * results[(idx_1[0], idx_1[0])][idval]
                        - 0.5 * results[(idx_1[1], idx_1[1])][idval]
                    )

        # for id in results.keys():
        #     print(f"results + {id}", results[id])
        total = {}
        for aux_op_name in pre_tap_aux_ops.keys():
            mss_mat = np.zeros((size, size), dtype=complex)
            msd_mat = np.zeros((size, size), dtype=complex)
            mds_mat = np.zeros((size, size), dtype=complex)
            mdd_mat = np.zeros((size, size), dtype=complex)

            for k in range(size):
                for l in range(size):
                    temp = results.get((f"E_{k}", f"E_{l}"), results.get((f"E_{l}", f"E_{k}"), 0.0))
                    if isinstance(temp, Dict) and len(temp) != 0:
                        mss_mat[k, l] = temp[aux_op_name]
                    else:
                        mss_mat[k, l] = 0.0

                    temp2 = results.get(
                        (f"Edag_{k}", f"Edag_{l}"), results.get((f"Edag_{l}", f"Edag_{k}"), 0.0)
                    )
                    if isinstance(temp2, Dict) and len(temp) != 0:
                        mdd_mat[k, l] = temp2[aux_op_name]
                    else:
                        mdd_mat[k, l] = 0.0

            a_mat = np.matrixlib.bmat([[mss_mat, -msd_mat], [-mds_mat, mdd_mat]])

            total[aux_op_name] = np.diag(expansion_coefs.T @ a_mat @ expansion_coefs)
        return total

    def _compute_transition_amplitudes(
        self,
        excited_operators_n,
        pre_tap_aux_ops,
        bound_ansatz,
    ):
        # Creates all the On @ Aux @ Om^\dag operators
        on_aux_om_dag_ops = ListOrDict()
        transition_amplitudes = {}
        if self._transition_amplitude_pairs is not None:
            for aux_op_name in self._transition_amplitude_pairs["names"]:
                op = pre_tap_aux_ops[aux_op_name]
                for pair in self._transition_amplitude_pairs["indices"]:
                    str_transition = aux_op_name + "_" + str(pair[0]) + "_" + str(pair[1])

                    if pair[0] == 0:
                        on_op = op
                    else:
                        left_op = excited_operators_n["Odag_" + str(pair[0])].adjoint()
                        on_op = left_op @ op

                    if pair[1] == 0:
                        on_op_om = on_op
                    else:
                        right_op = excited_operators_n["Odag_" + str(pair[1])]
                        on_op_om = on_op @ right_op

                    on_aux_om_dag_ops[str_transition] = on_op_om.reduce()

            on_aux_om_dag_operators_tapered = self.qubit_converter._symmetry_reduce_no_clifford(
                on_aux_om_dag_ops, True
            )

            transition_amplitudes = eval_observables(
                self._quantum_instance,
                bound_ansatz,
                on_aux_om_dag_operators_tapered,
                self._expectation,
            )

        return transition_amplitudes

    @staticmethod
    def _construct_excited_operators_n(
        hopping_operators: Dict[str, PauliSumOp],
        hopping_operators_eval,
        expansion_coefs: np.ndarray,
        size,
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
        # Creates all the On and On^\dag operators
        general_excitation_operators = {}  # O(n)^\dag for n = 1,2,3,...,size
        general_excitation_operators_eval = {}  # O(n)^\dag for n = 1,2,3,...,size
        # operator_indices = list(range(1, size + 1))
        operator_indices = list(itertools.chain(range(-size, 0), range(1, size + 1)))
        alpha = np.zeros(len(operator_indices), dtype=complex)
        gamma_square = np.zeros(len(operator_indices), dtype=complex)

        for n in range(0, len(operator_indices)):
            general_excitation_operators[f"Odag_{operator_indices[n]}"] = 0
            general_excitation_operators_eval[f"Odag_{operator_indices[n]}"] = 0
            for mu in range(0, size):
                de_excitation_op = hopping_operators.get(f"E_{mu}", 0)
                excitation_op = hopping_operators.get(f"Edag_{mu}", 0)
                de_excitation_eval = hopping_operators_eval.get(f"E_{mu}", [0, 0])[0]
                excitation_eval = hopping_operators_eval.get(f"Edag_{mu}", [0, 0])[0]

                general_excitation_operators[f"Odag_{operator_indices[n]}"] += (
                    complex(expansion_coefs[mu, n]) * de_excitation_op
                    - complex(expansion_coefs[mu + size, n]) * excitation_op
                )
                general_excitation_operators_eval[f"Odag_{operator_indices[n]}"] += (
                    complex(expansion_coefs[mu, n]) * de_excitation_eval
                    - complex(expansion_coefs[mu + size, n]) * excitation_eval
                )

        for n in range(0, len(operator_indices)):
            num_qubits = excitation_op.num_qubits
            alpha[n] = general_excitation_operators_eval[f"Odag_{operator_indices[n]}"]
            gamma_square[n] = 1

            general_excitation_operators[f"Odag_{operator_indices[n]}"] = (
                (
                    general_excitation_operators[f"Odag_{operator_indices[n]}"]
                    - PauliOp(
                        Pauli("I" * num_qubits),
                        alpha[n],
                    )
                )
                / np.sqrt(gamma_square[n])
            ).reduce()

        print("alpha :", alpha)
        print("gamma :", gamma_square)

        return general_excitation_operators, alpha, gamma_square

    def _prepare_second_q_ops(self, problem, aux_operators):
        second_q_ops = problem.second_q_ops()
        if isinstance(second_q_ops, list):
            main_second_q_op: SecondQuantizedOp = second_q_ops[0]
            aux_second_q_ops: ListOrDictType[SecondQuantizedOp] = second_q_ops[1:]
        elif isinstance(second_q_ops, dict):
            name = problem.main_property_name
            main_second_q_op: SecondQuantizedOp = second_q_ops.pop(name, None)

            if main_second_q_op is None:
                raise ValueError(
                    f"The main `SecondQuantizedOp` associated with the {name} property cannot be "
                    "`None`."
                )

            aux_second_q_ops: ListOrDictType[SecondQuantizedOp] = second_q_ops

        # Applies z2symmetries to the Hamiltonian
        # Sets _num_particle and _z2symmetries for the qubit_converter

        self._untapered_qubit_op_main = self.qubit_converter.convert_only(
            main_second_q_op, num_particles=problem.num_particles
        )

        tapered_qubit_op_main, z2symmetries = self.qubit_converter.find_taper_op(
            self._untapered_qubit_op_main, problem.symmetry_sector_locator
        )

        # Update the num_particles of the qubit converter to prepare the call of convert_match
        # on the auxiliaries.
        # The z2symmetries are deliberately not set at this stage because we do not want to
        # taper auxiliaries
        self.qubit_converter.force_match(
            num_particles=problem.num_particles, z2symmetries=Z2Symmetries([], [], [])
        )

        # Apply the same Mapping and Two Qubit Reduction as for the Hamiltonian
        untapered_aux_second_q_ops = self.qubit_converter.convert_match(aux_second_q_ops)
        if aux_operators is not None:
            if isinstance(aux_operators, ListOrDict):
                aux_operators = dict(aux_operators)
            untapered_aux_ops = self.qubit_converter.convert_match(aux_operators)
            untapered_aux_ops = ListOrDict(untapered_aux_ops)
            for aux_str, untapered_aux_op in iter(untapered_aux_ops):
                untapered_aux_second_q_ops[aux_str] = untapered_aux_op

        # Setup the z2symmetries that will be used to taper the qeom matrix element later
        self.qubit_converter.force_match(z2symmetries=z2symmetries)

        # Pre-calculation of the tapering, must come after force_match()
        self._pre_tap_qubit_op_main = self.qubit_converter._convert_clifford(
            self._untapered_qubit_op_main
        )
        pre_tap_aux_ops = self.qubit_converter._convert_clifford(
            ListOrDict(untapered_aux_second_q_ops)
        )
        return pre_tap_aux_ops

    def _prepare_matrix_operators(self, problem) -> Tuple[Dict, Dict, int]:
        """Construct the excitation operators for each matrix element.

        Returns:
            a dictionary of all matrix elements operators and the number of excitations
            (or the size of the qEOM pseudo-eigenvalue problem)
        """
        data = problem.hopping_qeom_ops(self.qubit_converter, self._excitations)
        hopping_operators, type_of_commutativities, excitation_indices = data

        size = int(len(list(excitation_indices.keys())) // 2)

        hopping_operators_norm = ListOrDict()
        for idx, hopping in hopping_operators.items():
            if not idx.startswith("Edag"):
                coeff = 1 / len(hopping.coeffs)
                hopping_operators_norm[idx] = hopping * coeff
                hopping_operators_norm["Edag" + idx[1:]] = hopping.adjoint() * coeff

        reduced_ops = self.qubit_converter.two_qubit_reduce(hopping_operators_norm)
        hopping_ops_pre_tap = self.qubit_converter._convert_clifford(reduced_ops)

        eom_matrix_operators = self._build_all_commutators(
            hopping_ops_pre_tap,
            type_of_commutativities,
            size,
        )

        return eom_matrix_operators, hopping_ops_pre_tap, size

    def _circuit_preparation(self, bound_ansatz):

        num_qubit = bound_ansatz.num_qubits
        excitation_list = bound_ansatz._get_excitation_list()
        size = len(excitation_list)
        excitation_dict = {
            "E_0": excitation_list[0],
            "E_1": excitation_list[1],
            "E_2": excitation_list[2],
        }
        excitation_dict_dag = {
            "Edag_0": tuple(reversed(excitation_list[0])),
            "Edag_1": tuple(reversed(excitation_list[1])),
            "Edag_2": tuple(reversed(excitation_list[2])),
        }

        # Diagonal elements
        def prepare_excitation_operators(excitation_tuple):
            occ = excitation_tuple[0]
            unocc = excitation_tuple[1]
            circuit = QuantumCircuit(num_qubit)

            for index_occ in occ:
                circuit.x(index_occ)
            for index_unocc in unocc:
                circuit.x(index_unocc)
            return circuit

        circuit_dict_diag = {}
        for idx, excitation in excitation_dict.items():
            circuit_dict_diag[(idx, idx)] = prepare_excitation_operators(excitation) + bound_ansatz
        for idx, excitation in excitation_dict_dag.items():
            circuit_dict_diag[(idx, idx)] = prepare_excitation_operators(excitation) + bound_ansatz

        # Trace for testing the computation of the eigenstates
        # for circ in circuit_dict_diag.values():
        #     psi = Statevector.from_instruction(circ).data
        #     print(circ)
        #     print(np.real(psi))

        results = {}
        for idx, circuit in circuit_dict_diag.items():
            value = eval_observables(
                self.quantum_instance,
                circuit,
                {"H": self._untapered_qubit_op_main},
                self.expectation,
            )
            results[idx] = value["H"][0]
        # print(results)

        # Anti-diagonal elements
        def prepare_excitation_operators_superposition(excitation_tuple1, excitation_tuple2):
            circuit = prepare_excitation_operators(excitation_tuple1)
            occ1 = excitation_tuple1[0]
            occ2 = excitation_tuple2[0]
            unocc1 = excitation_tuple1[1]
            unocc2 = excitation_tuple2[1]
            diff1 = [exc for exc in occ1 if exc not in occ2] + [
                exc for exc in occ2 if exc not in occ1
            ]
            diff2 = [exc for exc in unocc1 if exc not in unocc2] + [
                exc for exc in unocc2 if exc not in unocc1
            ]
            ref_pos = [exc for exc in unocc2 if exc not in unocc1][0]
            # print(ref_pos)
            # print("diff1", diff1)
            # print("diff2", diff2)
            circuit.h(ref_pos)
            for index_occ in diff1:
                if index_occ != ref_pos:
                    circuit.cnot(ref_pos, index_occ)
            for index_unocc in diff2:
                if index_unocc != ref_pos:
                    circuit.cnot(ref_pos, index_unocc)

            return circuit

        # print(excitation_dict)

        circuit_dict_antidiag = {}
        for idx_1, excitation_1 in excitation_dict.items():
            for idx_2, excitation_2 in excitation_dict.items():
                if idx_1 < idx_2:
                    # print(idx_1, idx_2)
                    circuit_dict_antidiag[(idx_1, idx_2)] = (
                        prepare_excitation_operators_superposition(excitation_1, excitation_2)
                        + bound_ansatz
                    )
        for idx_1, excitation_1 in excitation_dict_dag.items():
            for idx_2, excitation_2 in excitation_dict_dag.items():
                if idx_1 < idx_2:
                    circuit_dict_antidiag[(idx_1, idx_2)] = (
                        prepare_excitation_operators_superposition(excitation_1, excitation_2)
                        + bound_ansatz
                    )

        # for idx, circ in circuit_dict_antidiag.items():
        #     psi = Statevector.from_instruction(circ).data
        #     print(idx)
        #     print(circ)
        #     print(np.real(psi))

        for idx_1, excitation_1 in excitation_dict.items():
            for idx_2, excitation_2 in excitation_dict.items():
                if idx_1 < idx_2:
                    # print(idx_1, idx_2)
                    value = eval_observables(
                        self.quantum_instance,
                        circuit_dict_antidiag[(idx_1, idx_2)],
                        {"H": self._untapered_qubit_op_main},
                        self.expectation,
                    )

                    results[(idx_1, idx_2)] = (
                        value["H"][0]
                        - 0.5 * results[(idx_1, idx_1)]
                        - 0.5 * results[(idx_2, idx_2)]
                    )

        for idx_1, excitation_1 in excitation_dict_dag.items():
            for idx_2, excitation_2 in excitation_dict_dag.items():
                if idx_1 < idx_2:
                    value = eval_observables(
                        self.quantum_instance,
                        circuit_dict_antidiag[(idx_1, idx_2)],
                        {"H": self._untapered_qubit_op_main},
                        self.expectation,
                    )
                    results[(idx_1, idx_2)] = (
                        value["H"][0]
                        - 0.5 * results[(idx_1, idx_1)]
                        - 0.5 * results[(idx_2, idx_2)]
                    )

        # print(results)

        mss_mat = np.zeros((size, size), dtype=complex)
        msd_mat = np.zeros((size, size), dtype=complex)
        mds_mat = np.zeros((size, size), dtype=complex)
        mdd_mat = np.zeros((size, size), dtype=complex)

        for k in range(size):
            for l in range(size):
                mss_mat[k, l] = results.get(
                    (f"E_{k}", f"E_{l}"), results.get((f"E_{l}", f"E_{k}"), 0.0)
                )
                mdd_mat[k, l] = results.get(
                    (f"Edag_{k}", f"Edag_{l}"), results.get((f"Edag_{l}", f"Edag_{k}"), 0.0)
                )

        a_mat = np.matrixlib.bmat([[mss_mat, -msd_mat], [-mds_mat, mdd_mat]])
        res = linalg.eig(a_mat)

        # print(a_mat)
        # print("eigenvalues", np.real(res[0]))
        # print(res[1])
        return res[0], res[1], circuit_dict_diag, circuit_dict_antidiag


class QEOMResult(EigensolverResult):
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
        self._eigenvalues = None
        self._eigenstates = None
        self._aux_operator_eigenvalues = None
        self._alpha: Optional[np.ndarray] = None
        self._gamma_square: Optional[np.ndarray] = None
        self._transition_amplitudes: Optional[np.ndarray] = None

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

    @property
    def eigenvalues(self) -> Optional[np.ndarray]:
        """returns eigen values"""
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, value: np.ndarray) -> None:
        """set eigen values"""
        self._eigenvalues = value

    @property
    def eigenstates(self) -> Optional[np.ndarray]:
        """return eigen states"""
        return self._eigenstates

    @eigenstates.setter
    def eigenstates(self, value: np.ndarray) -> None:
        """set eigen states"""
        self._eigenstates = value

    @property
    def aux_operator_eigenvalues(self) -> Optional[List[ListOrDict[Tuple[complex, complex]]]]:
        """Return aux operator expectation values.

        These values are in fact tuples formatted as (mean, standard deviation).
        """
        return self._aux_operator_eigenvalues

    @aux_operator_eigenvalues.setter
    def aux_operator_eigenvalues(self, value: List[ListOrDict[Tuple[complex, complex]]]) -> None:
        """set aux operator eigen values"""
        self._aux_operator_eigenvalues = value

    @property
    def alpha(self) -> Optional[np.ndarray]:
        """returns the correction coefficients alpha_n"""
        return self._alpha

    @alpha.setter
    def alpha(self, value: np.ndarray) -> None:
        """sets the correction coefficients alpha_n"""
        self._alpha = value

    @property
    def gamma_square(self) -> Optional[np.ndarray]:
        """returns the correction coefficients gamma_square_n"""
        return self._gamma_square

    @gamma_square.setter
    def gamma_square(self, value: np.ndarray) -> None:
        """sets the correction coefficients gamma_square_n"""
        self._gamma_square = value

    @property
    def transition_amplitudes(self) -> Optional[np.ndarray]:
        """returns the transition amplitudes"""
        return self._transition_amplitudes

    @transition_amplitudes.setter
    def transition_amplitudes(self, value: np.ndarray) -> None:
        """sets the transition_amplitudes"""
        self._transition_amplitudes = value
