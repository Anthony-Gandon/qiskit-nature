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
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import EigensolverResult
from qiskit.providers import Backend
from qiskit.opflow import Z2Symmetries, commutator, double_commutator, PauliSumOp, StateFn, \
    ExpectationBase, anti_commutator
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliOp

from qiskit.algorithms import eval_observables

from qiskit_nature.converters.second_quantization.utils import ListOrDict
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
        self._partially_tapered_qubit_op_main: PauliSumOp = None

        self._eval_aux_excited_states: bool = False
        self._quantum_instance: Optional[Union[QuantumInstance, Backend]] = None
        self._expectation: Optional[ExpectationBase] = None
        self._transition_amplitude_pairs: Dict[List] = None

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

    def set_eval_aux_ops_args(self, eval_aux_ops_args:Dict):
        """
        Set up various parameters to define if and how the auxiliary operators should be evaluated
        on excited states.
        Args:
            eval_aux_ops_args: Dictionnary of parameters
                eval_aux_excited_states: If True, the auxiliary operators are evaluated on the
                excited states. (NB: Requires extra measurements)
                quantum_instance: Must be provided if evaluate_aux_qeom is True
                expectation: Must be provided if evaluate_aux_qeom is True
                transition_amplitude_pairs: Specifies the transition amplitudes to evaluate
        """
        self._eval_aux_excited_states = eval_aux_ops_args.get("eval_aux_excited_states",None)
        self._quantum_instance = eval_aux_ops_args.get("quantum_instance", None)
        self._expectation = eval_aux_ops_args.get("expectation", None)
        self._transition_amplitude_pairs = eval_aux_ops_args.get("transition_amplitude_pairs", None)

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
        """Run the excited-states calculation.

        Construct and solves the EOM pseudo-eigenvalue problem to obtain the excitation energies
        and the excitation operators expansion coefficients.

        Args:
            problem: a class encoding a problem to be solved.
            aux_operators: Additional auxiliary operators to evaluate.
            eval_aux_excited_states:
            transition_amplitude_pairs:

        Returns:
            An interpreted :class:`~.EigenstateResult`. For more information see also
            :meth:`~.BaseProblem.interpret`.
        """

        # 1. Prepare the auxiliary operators
        # Prepares self._untapered_qubit_op_main
        # Prepares self._partially_tapered_qubit_op_main to speed up the tapering
        # Note that the tapering must be delayed until the building of the commutators
        partially_tapered_aux_ops = self._prepare_second_q_ops(problem, aux_operators)

        # 2. Prepare the matrix operators.
        # if self._eval_aux_excited_states==False,
        #   matrix_operators_dict = {aux}
        # if self._eval_aux_excited_states==False,
        #   matrix_operators_dict = {aux, E_\alpha, E_\alpha E_\beta}
        # If a z2symmetry is provided, the operators are the partially tapered.
        (
            matrix_operators_dict,
            hopping_ops_partially_tapered,
            size,
        ) = self._prepare_matrix_operators(problem)

        partially_tapered_aux_ops_user = ListOrDict()
        for aux_name, aux_op in partially_tapered_aux_ops:
            if aux_name not in problem.second_q_ops().keys():
                partially_tapered_aux_ops_user[aux_name] = aux_op
        tapered_aux_ops = self._gsc.qubit_converter._symmetry_reduce_no_clifford(
            partially_tapered_aux_ops_user, True
        )
        matrix_operators_dict.update(tapered_aux_ops)

        # 3. Run ground state calculation with all the matrix elements as auxiliaries
        groundstate_result = self._gsc.solve(problem, matrix_operators_dict)
        measurement_results = groundstate_result.aux_operator_eigenvalues[0]

        # Tries to retrieve the ansatz as a circuit and set its parameters to the one
        # given by the GroundStateEigenSolver.
        # If the GrounsStateEigenSolver is not VQE then we take the eigenstate output.
        # This is needed for example for StateVector simulations with NumpyMinimumEigenSolver.
        if self._gsc.solver is not None and hasattr(self._gsc.solver, "ansatz"):
            bound_ansatz = self._gsc.solver.ansatz.assign_parameters(
                groundstate_result.raw_result.optimal_point
            )
        else:
            bound_ansatz = groundstate_result.eigenstates[0]

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

        # 5. Solve pseudo-eigenvalue problem
        energy_gaps, expansion_coefs = self._compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        # 6. Computes the product metric which is needed for the evaluation of auxiliary operators.
        product_metric = self._compute_metric(measurement_results, expansion_coefs, size)
        print("Product Metric")
        print(np.real(product_metric))
        # 7. Reconstructs the excitation operators On^\dag = |n><0| from the coefficients, the basis
        # operators, and the corrections.
        # If self._construct_true_eigenstates=True, then product_metric and measurement_results
        # contains all the terms necessary for the correction of On^\dag
        excitation_operators_partially_tapered, alpha, gamma_square = self._construct_excited_operators_n(
            hopping_ops_partially_tapered,
            measurement_results,
            expansion_coefs,
            size,
            product_metric,
        )

        # 8. Prepares results if self._eval_aux_excited_states is False
        excited_eigenenergies = np.asarray(
            [groundstate_result.eigenenergies[0] + gap for gap in energy_gaps]
        )
        eigenenergies = np.append(groundstate_result.eigenenergies, excited_eigenenergies)
        recalculated_excited_energies = np.zeros_like(eigenenergies)

        if not isinstance(groundstate_result.eigenstates[0], StateFn):
            eigenstates = [StateFn(groundstate_result.eigenstates[0])]
        else:
            eigenstates = [groundstate_result.eigenstates[0]]

        aux_operator_eigenvalues = groundstate_result.aux_operator_eigenvalues
        transition_amplitudes = []

        # 9. Evaluation of auxiliaries on excited states if self._eval_aux_excited_states is True
        if self._eval_aux_excited_states:
            (
                aux_operator_eigenvalues_excited_states
            ) = self._eval_all_aux_ops(
                excitation_operators_partially_tapered,
                partially_tapered_aux_ops,
                bound_ansatz,
            )

            (
                transition_amplitudes
            ) = self._compute_transition_amplitudes(
                excitation_operators_partially_tapered,
                partially_tapered_aux_ops,
                bound_ansatz,
            )

            # Updates the excited states
            for excitation_op_n in excitation_operators_partially_tapered.values():
                eigenstates.append((excitation_op_n @ eigenstates[0]).eval())

            # Updates the eigenvalues of auxiliaries on excited states
            aux_operator_eigenvalues = (
                aux_operator_eigenvalues + aux_operator_eigenvalues_excited_states
            )

        # 10. Refactor the results
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

        qeom_result.eigenstates = eigenstates
        qeom_result.eigenvalues = eigenenergies
        qeom_result.transition_amplitudes = transition_amplitudes
        qeom_result.aux_operator_eigenvalues = aux_operator_eigenvalues
        qeom_result.alpha = alpha
        qeom_result.gamma_square = gamma_square

        result = problem.interpret(qeom_result)
        return result, excitation_operators_partially_tapered, hopping_ops_partially_tapered, expansion_coefs

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

        # print(z2_symmetries)

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

                _build_one_sector(
                    available_hopping_ops, self._partially_tapered_qubit_op_main, z2_symmetries
                )

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
        w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
        m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
        v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

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
        a_mat = np.matrixlib.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.matrixlib.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])

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
        order = np.argsort(np.real(res[0]))
        w = np.real(res[0])[order]
        logger.debug("Order real parts %s", order)
        logger.debug("Sorted real parts %s", w)
        #w = np.abs(w[len(w) // 2 :])
        w = np.abs(w)
        w[np.abs(w) < 1e-06] = 0
        excitation_energies_gap = w
        # expansion_coefs = res[1][:, order[len(order) // 2 :]]
        # expansion_coefs = res[1][:, order[: len(order) // 2]]
        # expansion_coefs = expansion_coefs[:, ::-1]
        expansion_coefs = res[1][:, order]
        commutator_metric = expansion_coefs.T.conjugate() @ b_mat @ expansion_coefs
        print(commutator_metric)
        return excitation_energies_gap, expansion_coefs

    def _eval_all_aux_ops(
        self,
        excited_operators_n,
        partially_tapered_aux_ops,
        bound_ansatz,
    ):

        # Creates all the On @ Aux @ On^\dag operators
        general_on_aux_on_dag_operators = {}
        for on_str, on_dag in excited_operators_n.items():
            listordict_aux_op_on = (
                ListOrDict()
            )  # {"Identity": (on_dag.adjoint() @ on_dag).reduce()}
            for aux_str, aux_op in iter(partially_tapered_aux_ops):
                listordict_aux_op_on[aux_str] = (on_dag.adjoint() @ aux_op @ on_dag).reduce()
            if not self.qubit_converter.z2symmetries.is_empty():
                listordict_aux_op_on_tapered = (
                    self.qubit_converter._symmetry_reduce_no_clifford(
                        listordict_aux_op_on, True
                    )
                )
            else:
                listordict_aux_op_on_tapered = ListOrDict(listordict_aux_op_on)

            general_on_aux_on_dag_operators[on_str] = listordict_aux_op_on_tapered

        aux_operator_eigenvalues_excited_states = []  # Eigenvalues of Aux_ops
        for index_n, on_aux_on_dag_operator in enumerate(general_on_aux_on_dag_operators.values()):
            if len(on_aux_on_dag_operator) != 0:
                not_normalized_eigenvalues = eval_observables(
                    self._quantum_instance, bound_ansatz, on_aux_on_dag_operator, self._expectation
                )
            else:
                not_normalized_eigenvalues = {}

            if isinstance(not_normalized_eigenvalues, dict):

                # TODO Fix the calculation of the variance + the expectation value

                aux_operator_eigenvalues_excited_states.append({})
                for op_name, op_eigenval in not_normalized_eigenvalues.items():
                    aux_operator_eigenvalues_excited_states[-1][op_name] = (
                        op_eigenval[0],
                        op_eigenval[1],
                    )

            elif isinstance(not_normalized_eigenvalues, list):
                aux_operator_eigenvalues_excited_states.append([])
                for op_name, op_eigenval in enumerate(not_normalized_eigenvalues):
                    aux_operator_eigenvalues_excited_states[index_n].append(
                        (op_eigenval[0], op_eigenval[1])
                    )

        return aux_operator_eigenvalues_excited_states

    def _compute_transition_amplitudes(
        self,
        excited_operators_n,
        partially_tapered_aux_ops,
        bound_ansatz,
    ):
        # Creates all the On @ Aux @ Om^\dag operators
        on_aux_om_dag_ops = ListOrDict()
        transition_amplitudes = {}
        if self._transition_amplitude_pairs is not None:
            for aux_op_name in self._transition_amplitude_pairs['names']:
                op = partially_tapered_aux_ops[aux_op_name]
                for pair in self._transition_amplitude_pairs['indices']:
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

                    on_aux_om_dag_ops[str_transition] =  on_op_om.reduce()

            on_aux_om_dag_operators_tapered = (
                self.qubit_converter._symmetry_reduce_no_clifford(
                    on_aux_om_dag_ops, True
                )
            )

            transition_amplitudes = eval_observables(
                self._quantum_instance,
                bound_ansatz,
                on_aux_om_dag_operators_tapered,
                self._expectation
            )

        return transition_amplitudes

    @staticmethod
    def _construct_excited_operators_n(
        hopping_operators: Dict[str, PauliSumOp],
        hopping_operators_eval,
        expansion_coefs: np.ndarray,
        size,
        product_metric,
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
        operator_indices = list(range(1, size + 1))
        # operator_indices = list(itertools.chain(range(-size, 0), range(1, size+1)))
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
            gamma_square[n] = product_metric[n, n] - alpha[n] ** 2

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

    def _compute_metric(self, measurement_results, expansion_coefs, size):
        A_mat = np.zeros((size, size), dtype=complex)
        B_mat = np.zeros((size, size), dtype=complex)
        C_mat = np.zeros((size, size), dtype=complex)
        D_mat = np.zeros((size, size), dtype=complex)
        for id1 in range(size):
            for id2 in range(size):
                A_mat[id1, id2] = measurement_results.get(f"Edag_{id1}E_{id2}", [0, 0])[0]
                B_mat[id1, id2] = -measurement_results.get(f"Edag_{id1}Edag_{id2}", [0, 0])[0]
                C_mat[id1, id2] = -measurement_results.get(f"E_{id1}E_{id2}", [0, 0])[0]
                D_mat[id1, id2] = measurement_results.get(f"E_{id1}Edag_{id2}", [0, 0])[0]

        metric = np.matrixlib.bmat([[A_mat, B_mat], [C_mat, D_mat]])

        if not np.any(metric):
            metric = np.eye(2 * size, dtype=complex)
        product_metric = expansion_coefs.T.conjugate() @ metric @ expansion_coefs
        return product_metric

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
        self._partially_tapered_qubit_op_main = self.qubit_converter._convert_clifford(
            self._untapered_qubit_op_main
        )
        partially_tapered_aux_ops = self.qubit_converter._convert_clifford(
            ListOrDict(untapered_aux_second_q_ops)
        )
        return partially_tapered_aux_ops

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
                hopping_operators_norm["Edag" + idx[1:]] = (
                    hopping.adjoint() * coeff
                )

        reduced_ops = self.qubit_converter.two_qubit_reduce(hopping_operators_norm)
        hopping_ops_partially_tapered = self.qubit_converter._convert_clifford(reduced_ops)

        eom_matrix_operators = self._build_all_commutators(
            hopping_ops_partially_tapered,
            type_of_commutativities,
            size,
        )

        if self._eval_aux_excited_states:
            hopping_operator_products = ListOrDict()
            for idx_left, op_left in hopping_ops_partially_tapered.items():
                for idx_right, op_right in hopping_ops_partially_tapered.items():
                    hopping_operator_products[idx_left + idx_right] = (op_left @ op_right).reduce()

            hopping_ops_partially_tapered.update(hopping_operator_products)

            tapered_hopping_ops = self.qubit_converter._symmetry_reduce_no_clifford(
                hopping_ops_partially_tapered, True
            )
            eom_matrix_operators.update(tapered_hopping_ops)

        return eom_matrix_operators, hopping_ops_partially_tapered, size


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
