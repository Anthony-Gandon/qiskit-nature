# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Active-Space Reduction interface."""

from __future__ import annotations

import logging

from copy import deepcopy
from typing import cast

import numpy as np

from qiskit_nature import QiskitNatureError
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy, Hamiltonian
from qiskit_nature.second_q.operators import ElectronicIntegrals
from qiskit_nature.second_q.problems import BaseProblem, ElectronicStructureProblem
from qiskit_nature.second_q.properties import (
    AngularMomentum,
    ElectronicDensity,
    ElectronicDipoleMoment,
    Magnetization,
    ParticleNumber,
)
from qiskit_nature.second_q.properties.bases import ElectronicBasis

from .base_transformer import BaseTransformer
from .basis_transformer import BasisTransformer

LOGGER = logging.getLogger(__name__)


class ActiveSpaceTransformer(BaseTransformer):
    r"""The Active-Space reduction.

    The reduction is done by computing the inactive Fock operator which is defined as
    :math:`F^I_{pq} = h_{pq} + \sum_i 2 g_{iipq} - g_{iqpi}` and the inactive energy which is
    given by :math:`E^I = \sum_j h_{jj} + F ^I_{jj}`, where :math:`i` and :math:`j` iterate over
    the inactive orbitals.
    By using the inactive Fock operator in place of the one-electron integrals, `h1`, the
    description of the active space contains an effective potential generated by the inactive
    electrons. Therefore, this method permits the exclusion of non-core electrons while
    retaining a high-quality description of the system.

    For more details on the computation of the inactive Fock operator refer to
    https://arxiv.org/abs/2009.01872.

    The active space can be configured in one of the following ways through the initializer:
        - when only `num_electrons` and `num_spatial_orbitals` are specified, these integers
          indicate the number of active electrons and orbitals, respectively. The active space will
          then be chosen around the Fermi level resulting in a unique choice for any pair of
          numbers.  Nonetheless, the following criteria must be met:

            #. the remaining number of inactive electrons must be a positive, even number

            #. the number of active orbitals must not exceed the total number of orbitals minus the
               number of orbitals occupied by the inactive electrons

        - when, `num_electrons` is a tuple, this must indicate the number of alpha- and beta-spin
          electrons, respectively. The same requirements as listed before must be met.
        - finally, it is possible to select a custom set of active orbitals via their indices using
          `active_orbitals`. This allows selecting an active space which is not placed around the
          Fermi level as described in the first case, above. When using this keyword argument, the
          following criteria must be met *in addition* to the ones listed above:

            #. the length of `active_orbitals` must be equal to `num_spatial_orbitals`. Note, that
               we do **not** infer the number of active orbitals from this list of indices!

            #. the sum of electrons present in `active_orbitals` must be equal to `num_electrons`.

    References:
        - *M. Rossmannek, P. Barkoutsos, P. Ollitrault, and I. Tavernelli, arXiv:2009.01872
          (2020).*
    """

    def __init__(
        self,
        num_electrons: int | tuple[int, int],
        num_spatial_orbitals: int,
        active_orbitals: list[int] | None = None,
    ):
        """
        Args:
            num_electrons: The number of active electrons. If this is a tuple, it represents the
               number of alpha- and beta-spin electrons, respectively. If this is a number, it is
               interpreted as the total number of active electrons, should be even, and implies that
               the number of alpha and beta electrons equals half of this value, respectively.
            num_spatial_orbitals: The number of active orbitals.
            active_orbitals: A list of indices specifying the spatial orbitals of the active
                space. This argument must match with the remaining arguments and should only be used
                to enforce an active space that is not chosen purely around the Fermi level.

        Raises:
            QiskitNatureError: if an invalid configuration is provided.
        """
        self._num_electrons = num_electrons
        self._num_spatial_orbitals = num_spatial_orbitals
        self._active_orbitals = active_orbitals

        try:
            self._check_configuration()
        except QiskitNatureError as exc:
            raise QiskitNatureError("Incorrect Active-Space configuration.") from exc

        self._mo_occ_total: np.ndarray = None
        self._active_orbs_indices: list[int] = None
        self._transform_active: BasisTransformer = None
        self._density_active: ElectronicIntegrals = None
        self._density_total: ElectronicIntegrals = None

    def _check_configuration(self):
        if isinstance(self._num_electrons, (int, np.integer)):
            if self._num_electrons % 2 != 0:
                raise QiskitNatureError(
                    "The number of active electrons must be even! Otherwise you must specify them "
                    "as a tuple, not as:",
                    str(self._num_electrons),
                )
            if self._num_electrons < 0:
                raise QiskitNatureError(
                    "The number of active electrons cannot be negative, not:",
                    str(self._num_electrons),
                )
        elif isinstance(self._num_electrons, tuple):
            if not all(
                isinstance(n_elec, (int, np.integer)) and n_elec >= 0
                for n_elec in self._num_electrons
            ):
                raise QiskitNatureError(
                    "Neither the number of alpha, nor the number of beta electrons can be "
                    "negative, not:",
                    str(self._num_electrons),
                )
        else:
            raise QiskitNatureError(
                "The number of active electrons must be an int, or a tuple thereof, not:",
                str(self._num_electrons),
            )

        if isinstance(self._num_spatial_orbitals, (int, np.integer)):
            if self._num_spatial_orbitals < 0:
                raise QiskitNatureError(
                    "The number of active orbitals cannot be negative, not:",
                    str(self._num_spatial_orbitals),
                )
        else:
            raise QiskitNatureError(
                "The number of active orbitals must be an int, not:",
                str(self._num_spatial_orbitals),
            )

    def transform(self, problem: BaseProblem) -> BaseProblem:
        """Transforms one :class:`~qiskit_nature.second_q.problems.BaseProblem` into another.

        Args:
            problem: the problem to be transformed.

        Raises:
            NotImplementedError: when an unsupported problem type is provided.
            NotImplementedError: when the ``ElectronicStructureProblem`` is not in the
                :attr:`qiskit_nature.second_q.properties.bases.ElectronicBasis.MO` basis.
            QiskitNatureError: If the provided ``ElectronicStructureProblem`` does not contain a
                               ``ParticleNumber``, if more electrons or orbitals are requested than
                               are available, or if the amount of selected active orbital indices
                               does not match the total number of active orbitals.

        Returns:
            A new `BaseProblem` instance.
        """
        if isinstance(problem, ElectronicStructureProblem):
            return self._transform_electronic_structure_problem(problem)
        else:
            raise NotImplementedError(
                f"The problem of type, {type(problem)}, is not supported by this transformer."
            )

    def _transform_electronic_structure_problem(
        self, problem: ElectronicStructureProblem
    ) -> ElectronicStructureProblem:

        if problem.basis != ElectronicBasis.MO:
            raise NotImplementedError(
                f"Transformation of an ElectronicStructureProblem in the {problem.basis} basis is "
                "not supported by this transformer. Please convert it to the ElectronicBasis.MO"
                " basis first, for example by using a BasisTransformer."
            )

        particle_number = problem.properties.particle_number
        if particle_number is None:
            raise QiskitNatureError(
                "The provided ElectronicStructureProblem does not contain a `ParticleNumber` "
                "property, which is required by this transformer!"
            )

        # get spatial orbital occupation numbers
        occupation_alpha = particle_number.occupation_alpha
        occupation_beta = particle_number.occupation_beta
        self._mo_occ_total = occupation_alpha + occupation_beta

        # determine the active space
        self._active_orbs_indices = self._determine_active_space(problem)

        # initialize size-reducing basis transformation
        coeff_alpha = np.zeros((particle_number.num_spin_orbitals // 2, self._num_spatial_orbitals))
        coeff_alpha[self._active_orbs_indices, range(self._num_spatial_orbitals)] = 1.0
        coeff_beta = np.zeros((particle_number.num_spin_orbitals // 2, self._num_spatial_orbitals))
        coeff_beta[self._active_orbs_indices, range(self._num_spatial_orbitals)] = 1.0
        self._transform_active = BasisTransformer(
            ElectronicBasis.MO,
            ElectronicBasis.MO,
            ElectronicIntegrals.from_raw_integrals(coeff_alpha, h1_b=coeff_beta, validate=False),
        )

        self._density_total = ElectronicIntegrals.from_raw_integrals(
            np.diag(occupation_alpha), h1_b=np.diag(occupation_beta)
        )

        self._density_active = self._transform_active.transform_electronic_integrals(
            self._density_total
        )
        self._density_active.beta_alpha = None
        self._density_active = self._transform_active.invert().transform_electronic_integrals(
            self._density_active
        )
        self._density_active.beta_alpha = None

        electronic_energy = cast(ElectronicEnergy, self.transform_hamiltonian(problem.hamiltonian))

        # construct new ElectronicStructureProblem
        new_problem = ElectronicStructureProblem(electronic_energy)
        new_problem.basis = ElectronicBasis.MO
        new_problem.molecule = problem.molecule

        for prop in problem.properties:
            if isinstance(prop, ElectronicDipoleMoment):
                new_problem.properties.electronic_dipole_moment = (
                    self._transform_electronic_dipole_moment(prop)
                )
            elif isinstance(prop, ParticleNumber):
                active_occ_alpha = prop.occupation_alpha[self._active_orbs_indices]
                active_occ_beta = prop.occupation_beta[self._active_orbs_indices]
                particle_number = ParticleNumber(
                    len(self._active_orbs_indices) * 2,
                    (int(sum(active_occ_alpha)), int(sum(active_occ_beta))),
                    active_occ_alpha,
                    active_occ_beta,
                )
                new_problem.properties.particle_number = particle_number
            elif isinstance(prop, (AngularMomentum, Magnetization)):
                new_problem.properties.add(prop.__class__(len(self._active_orbs_indices) * 2))
            elif isinstance(prop, ElectronicDensity):
                transformed = self._transform_active.transform_electronic_integrals(prop)
                new_problem.properties.electronic_density = ElectronicDensity(
                    transformed.alpha, transformed.beta, transformed.beta_alpha
                )

            else:
                LOGGER.warning("Encountered an unsupported property of type '%s'.", type(prop))

        return new_problem

    def _determine_active_space(self, problem: ElectronicStructureProblem) -> list[int]:
        """Determines the active and inactive orbital indices.

        Args:
            problem: the ElectronicStructureProblem to be transformed.

        Returns:
            The list of active and inactive orbital indices.
        """
        particle_number = problem.properties.particle_number
        if isinstance(self._num_electrons, tuple):
            num_alpha, num_beta = self._num_electrons
        elif isinstance(self._num_electrons, (int, np.integer)):
            num_alpha = num_beta = self._num_electrons // 2

        # compute number of inactive electrons
        nelec_total = particle_number._num_alpha + particle_number._num_beta
        nelec_inactive = nelec_total - num_alpha - num_beta

        self._validate_num_electrons(nelec_inactive)
        self._validate_num_orbitals(nelec_inactive, particle_number)

        # determine active and inactive orbital indices
        if self._active_orbitals is None:
            norbs_inactive = nelec_inactive // 2
            active_orbs_idxs = list(
                range(norbs_inactive, norbs_inactive + self._num_spatial_orbitals)
            )
            return active_orbs_idxs

        return self._active_orbitals

    def _validate_num_electrons(self, nelec_inactive: int) -> None:
        """Validates the number of electrons.

        Args:
            nelec_inactive: the computed number of inactive electrons.

        Raises:
            QiskitNatureError: if the number of inactive electrons is either negative or odd.
        """
        if nelec_inactive < 0:
            raise QiskitNatureError("More electrons requested than available.")
        if nelec_inactive % 2 != 0:
            raise QiskitNatureError("The number of inactive electrons must be even.")

    def _validate_num_orbitals(self, nelec_inactive: int, particle_number: ParticleNumber) -> None:
        """Validates the number of orbitals.

        Args:
            nelec_inactive: the computed number of inactive electrons.
            particle_number: the `ParticleNumber` containing system size information.

        Raises:
            QiskitNatureError: if more orbitals were requested than are available in total or if the
                               number of selected orbitals mismatches the specified number of active
                               orbitals.
        """
        if self._active_orbitals is None:
            norbs_inactive = nelec_inactive // 2
            if (
                norbs_inactive + self._num_spatial_orbitals
                > particle_number._num_spin_orbitals // 2
            ):
                raise QiskitNatureError("More orbitals requested than available.")
        else:
            if self._num_spatial_orbitals != len(self._active_orbitals):
                raise QiskitNatureError(
                    "The number of selected active orbital indices does not "
                    "match the specified number of active orbitals."
                )
            if max(self._active_orbitals) >= particle_number._num_spin_orbitals // 2:
                raise QiskitNatureError("More orbitals requested than available.")
            expected_num_electrons = (
                self._num_electrons
                if isinstance(self._num_electrons, (int, np.integer))
                else sum(self._num_electrons)
            )
            if sum(self._mo_occ_total[self._active_orbitals]) != expected_num_electrons:
                raise QiskitNatureError(
                    "The number of electrons in the selected active orbitals "
                    "does not match the specified number of active electrons."
                )

    def transform_hamiltonian(self, hamiltonian: Hamiltonian) -> Hamiltonian:
        """Transforms one :class:`~qiskit_nature.second_q.hamiltonians.Hamiltonian` into another.

        Args:
            hamiltonian: the hamiltonian to be transformed.

        Raises:
            NotImplementedError: when an unsupported hamiltonian type is provided.
            NotImplementedError: when called standalone.

        Returns:
            A new `Hamiltonian` instance.
        """
        if isinstance(hamiltonian, ElectronicEnergy):
            # TODO: implement the standalone usage of this method
            # See also: https://github.com/Qiskit/qiskit-nature/issues/847
            if self._transform_active is None:
                raise NotImplementedError(
                    "This transformer does not yet support the standalone use of the "
                    "transform_hamiltonian method. See also "
                    "https://github.com/Qiskit/qiskit-nature/issues/847"
                )
            return self._transform_electronic_energy(hamiltonian)
        else:
            raise NotImplementedError(
                f"The hamiltonian of type, {type(hamiltonian)}, is not supported by this "
                "transformer."
            )

    def _transform_electronic_energy(self, hamiltonian: ElectronicEnergy) -> ElectronicEnergy:
        total_fock_operator = hamiltonian.fock(self._density_total)

        active_fock_operator = (
            hamiltonian.fock(self._density_active) - hamiltonian.electronic_integrals.one_body
        )

        inactive_fock_operator = total_fock_operator - active_fock_operator

        e_inactive: ElectronicIntegrals = cast(
            ElectronicIntegrals,
            0.5
            * ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")},
                total_fock_operator + hamiltonian.electronic_integrals.one_body,
                self._density_total,
            ),
        )
        e_inactive -= ElectronicIntegrals.einsum(
            {"ij,ji": ("+-", "+-", "")}, total_fock_operator, self._density_active
        )
        e_inactive += cast(
            ElectronicIntegrals,
            0.5
            * ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, active_fock_operator, self._density_active
            ),
        )
        e_inactive_sum = sum(e_inactive[key].get("", 0.0) for key in e_inactive)

        new_hamil = ElectronicEnergy(
            self._transform_active.transform_electronic_integrals(
                inactive_fock_operator + hamiltonian.electronic_integrals.two_body
            )
        )
        new_hamil.constants = deepcopy(hamiltonian.constants)
        new_hamil.constants[self.__class__.__name__] = e_inactive_sum

        return new_hamil

    def _transform_electronic_dipole_moment(
        self, dipole_moment: ElectronicDipoleMoment
    ) -> ElectronicDipoleMoment:
        dipoles: list[ElectronicIntegrals] = []
        dip_inactive: list[float] = []
        for dipole in [dipole_moment.x_dipole, dipole_moment.y_dipole, dipole_moment.z_dipole]:
            # In the dipole case, there are no two-body terms. Thus, the inactive Fock operator
            # is unaffected by the density and equals the one-body terms.
            one_body = dipole.one_body

            e_inactive = ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, one_body, self._density_total
            )
            e_inactive -= ElectronicIntegrals.einsum(
                {"ij,ji": ("+-", "+-", "")}, one_body, self._density_active
            )
            dipoles.append(self._transform_active.transform_electronic_integrals(one_body))
            dip_inactive.append(sum(e_inactive[key].get("", 0.0) for key in e_inactive))

        new_dipole_moment = ElectronicDipoleMoment(
            x_dipole=dipoles[0],
            y_dipole=dipoles[1],
            z_dipole=dipoles[2],
        )
        new_dipole_moment.constants = deepcopy(dipole_moment.constants)
        new_dipole_moment.constants[self.__class__.__name__] = (
            dip_inactive[0],
            dip_inactive[1],
            dip_inactive[2],
        )
        new_dipole_moment.reverse_dipole_sign = dipole_moment.reverse_dipole_sign
        new_dipole_moment.nuclear_dipole_moment = dipole_moment.nuclear_dipole_moment

        return new_dipole_moment
