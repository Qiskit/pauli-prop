# This code is a Qiskit project.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Reminder: update the RST file in docs/apidocs when adding new interfaces.
"""Functions for performing Pauli propagation."""

import warnings
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from qiskit.circuit import CircuitInstruction, QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Clifford, Operator, Pauli, PauliList, SparsePauliOp
from pauli_propagation._accelerate import (
    evolve_by_circuit as evolve_by_circuit_r,
)
from pauli_propagation._accelerate import (
    k_largest_products as k_largest_products_r,
)
from qiskit_aer.noise import PauliLindbladError

# We intentionally cast the complex coeffs to real
warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)

_ROTATION_TO_GENERATOR = {
    "rx": Pauli("X"),
    "ry": Pauli("Y"),
    "rz": Pauli("Z"),
    "rxx": Pauli("XX"),
    "ryy": Pauli("YY"),
    "rzz": Pauli("ZZ"),
}

KNOWN_CLIFFS = {
    "cx",
    "cz",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "h",
    "i",
    "x",
    "y",
    "z",
    "cy",
    "ecr",
}


def _commutation_matrix(pl1: PauliList, pl2: PauliList, negate=False):
    a_dot_b = (np.asarray(pl1._x, dtype=np.uint8) @ np.asarray(pl2._z.T, dtype=np.uint8)) & 1
    b_dot_a = (np.asarray(pl1._z, dtype=np.uint8) @ np.asarray(pl2._x.T, dtype=np.uint8)) & 1
    if negate:
        return a_dot_b != b_dot_a
    return a_dot_b == b_dot_a


def evolve_through_cliffords(circuit: QuantumCircuit) -> tuple[Clifford, QuantumCircuit]:
    r"""Evolve (Schrödinger frame) all non-Clifford instructions through all Clifford gates in the circuit.

    This shifts all recognized Clifford gates to the beginning of the circuit and updates the bases of
    Pauli-rotation gates (e.g. ``RxGate``, ``RzzGate``, ``PauliEvolutionGate``) and `PauliLindbladError <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.PauliLindbladError.html#qiskit_aer.noise.PauliLindbladError>`_
    channels. Other operations are not supported. See `Pauli.evolve docs <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Pauli#evolve>`_ for more info about evolution
    of Paulis by Cliffords.

    The effect is similar to going to the Clifford interaction picture in `arXiv:2306.04797 <https://arxiv.org/abs/2306.04797>`_ but
    without mapping all rotation angle magnitudes to be :math:`\leq \pi/4`.

    The function returns two objects representing the Clifford and non-Clifford parts of the circuit.

    Args:
        circuit: The ``QuantumCircuit`` to transform. Can contain only Pauli-rotation gates, ``PauliLindbladError`` (appended to the
            circuit as quantum channels), and recognized Clifford gates.

    Returns:
            * **Clifford** - A single all-qubit `Clifford <https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Clifford>`_ representing the first part of the circuit
            * **QuantumCircuit** - A circuit containing the remaining, transformed part of the circuit

    Raises:
        ValueError: Input circuit contains unsupported gate
    """
    id_pauli = Pauli("I" * circuit.num_qubits)
    net_clifford = Clifford.from_label("I" * circuit.num_qubits)
    non_cliffords = QuantumCircuit.copy_empty_like(circuit)
    for i, circ_inst in enumerate(reversed(circuit)):
        if circ_inst.name == "barrier":
            continue
        qargs = [circuit.find_bit(qubit).index for qubit in circ_inst.qubits]
        if circ_inst.name in KNOWN_CLIFFS:
            net_clifford = net_clifford.dot(circ_inst.operation, qargs)
        elif circ_inst.name in _ROTATION_TO_GENERATOR:
            # Pauli rotation gate:
            pauli = _ROTATION_TO_GENERATOR[circ_inst.name]
            # Expand to full width of the circuit:
            pauli = id_pauli.dot(pauli, qargs=qargs)
            # Evolve by all subsequent Cliffords:
            # For large circuits, faster to evolve by net_clifford than by individual gates
            pauli = pauli.evolve(net_clifford, frame="s")
            pauli_evo_angle = circ_inst.params[0] / 2
            if pauli.phase == 2:
                pauli_evo_angle *= -1
                pauli.phase = 0
            assert pauli.phase == 0  # Paulis should always have real coeffs
            # Reduce to supported qubits only:
            support = np.where(pauli.z | pauli.x)[0].tolist()
            pauli = pauli[support]
            # Collect in non_cliffords circuit as Pauli rotation
            peg = PauliEvolutionGate(pauli, pauli_evo_angle)
            non_cliffords.append(peg, qargs=support, copy=False)
        elif circ_inst.name == "quantum_channel" and hasattr(circ_inst.operation, "_quantum_error"):
            # Pauli-Lindblad channel:
            error = circ_inst.operation._quantum_error
            # Expand to full width of the circuit:
            generators = PauliList([id_pauli] * len(error.generators))
            generators.dot(error.generators, qargs=qargs, inplace=True)
            # Evolve by all subsequent Cliffords:
            ple = PauliLindbladError(generators.evolve(net_clifford, frame="s"), error.rates)
            non_cliffords.append(ple, qargs=range(circuit.num_qubits), copy=False)
        else:
            raise ValueError(
                f"Unsupported gate encountered in circuit data idx {len(circuit) - (i + 1)}: {circ_inst.name}"
            )
    return net_clifford, non_cliffords.reverse_ops()


class RotationGates(NamedTuple):
    """An intermediate minimal representation of a :class:`.QuantumCircuit`.

    During :func:`.propagate_through_circuit` the :class:`.QuantumCircuit` gets converted into a sequence of
    rotation gates, extracting the parameters and acted-upon qubit indices. These data structures
    can be passed to the Rust-accelerated internal function in a straight forward manner.
    """

    gates: list[npt.NDArray[np.bool_]]
    """A ZX-calculus-like representation of the gates."""
    qargs: list[list[int]]
    """The qubit indices acted upon by each gate."""
    thetas: list[float]
    """The rotation angles of all gates."""

    def append_circuit_instruction(
        self,
        inst: CircuitInstruction,
        qargs: list[int],
        num_qubits: int,
        *,
        clifford: Clifford | None = None,
    ) -> None:
        """Parses a circuit instruction and appends its data to the internal lists.

        Args:
            inst: The circuit instruction to parse and append
            qargs: The list of qubit indices of the instruction in the context of its circuit
            num_qubits: The number of qubits of the circuit containing this instruction
            clifford: An optional Clifford through which the provided instruction should be moved.
                The Clifford must act on all qubits in the circuit.

        Raises:
            ValueError: Unsupported gate encountered in circuit
            ValueError: If given, ``clifford`` must act on all qubits in circuit
        """
        if (clifford is not None) and (clifford.num_qubits != num_qubits):
            raise ValueError("Clifford must act on all qubits in circuit.")

        theta = inst.operation.params[0]
        if not isinstance(inst.operation, PauliEvolutionGate):
            if inst.name not in _ROTATION_TO_GENERATOR:
                raise ValueError(f"Encountered unsupported gate: {inst.name}")
            gate = SparsePauliOp.from_operator(Operator(inst.operation))
            rotation_pauli = gate.paulis[np.any((gate.paulis.z | gate.paulis.x), axis=1)]
            assert len(rotation_pauli) == 1
            rotation_pauli = rotation_pauli[0]
        else:
            assert len(inst.operation.operator.paulis) == 1
            rotation_pauli = inst.operation.operator.paulis[0]
            theta *= 2.0

        rotation_pauli = rotation_pauli.apply_layout(qargs, num_qubits=num_qubits)

        if clifford is not None:
            rotation_pauli = rotation_pauli.evolve(clifford, frame="s")
            if rotation_pauli.phase == 2:
                theta *= -1
                rotation_pauli.phase = 0

        assert rotation_pauli.phase == 0

        gate_arr = np.concatenate((rotation_pauli.x, rotation_pauli.z))

        self.gates.append(gate_arr)
        self.qargs.append(qargs)
        self.thetas.append(theta)


def circuit_to_rotation_gates(
    circuit: QuantumCircuit,
) -> RotationGates:
    """Converts the provided circuit to an intermediate representation.

    Args:
        circuit: The circuit to convert. It may contain gates that are Pauli rotations or `PauliLindbladError <https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.PauliLindbladError.html#qiskit_aer.noise.PauliLindbladError>`_ instances.

    Returns:
        The extracted rotation gate data.

    Raises:
        ValueError: when an unsupported gate is encountered in ``circuit``.
    """
    rot_gates = RotationGates([], [], [])
    for data in circuit.data:
        if data.name == "barrier":
            continue
        qargs = [circuit.find_bit(qubit).index for qubit in data.qubits]
        rot_gates.append_circuit_instruction(data, qargs, circuit.num_qubits)
    return rot_gates


def propagate_through_rotation_gates(
    operator: SparsePauliOp,
    rot_gates: RotationGates,
    max_terms: int,
    atol: float,
    frame: str,
) -> tuple[SparsePauliOp, float]:
    """Propagate a sparse Pauli operator, :math:`O`, through a circuit (represented in ``rot_gates``), :math:`U`.

    For Schrödinger propagation: :math:`U O U†`.

    For Heisenberg propagation: :math:`U† O U`.

    In general, the memory and time required for propagating through a circuit grows exponentially with the number of operations in the
    circuit due to the exponential growth in the number of terms of the operator in the Pauli basis. To regulate this exponential
    difficulty, one may truncate small Pauli terms (i.e. set them to zero), resulting in a bias proportional to the magnitudes of the
    truncated terms. After propagating through each operation in the circuit, terms are truncated with respect to two parameters:

    - Only the ``max_terms`` largest Pauli components are kept; any smaller terms will be truncated. This option makes it
      possible to estimate in advance how much time and memory will suffice for the computation.
    - Terms with magnitudes less than ``atol`` are truncated (set to zero).

    .. note::
        This function pre-allocates space in memory for the full-sized operator and operator buffer. It is the caller's
        responsibility to ensure they have enough memory to hold operators containing ``max_terms`` terms. When ``max_terms`` is
        ``None``, the memory and time requirements typically grow exponentially with the number of operations in the circuit.

    Args:
        operator: The operator to propagate
        rot_gates: The circuit represented in the form of :class:`.RotationGates` (see also :func:`.circuit_to_rotation_gates`).
        max_terms: The maximum number of terms the operator may contain as it is propagated
        atol: Terms with coeff magnitudes less than this will not be added to the operator as it is propagated. This parameter is not a
            guarantee on the accuracy of the returned operator.
        frame:
            ``s`` for Schrödinger evolution
            ``h`` for Heisenberg evolution

    Returns:
        The evolved operator

    Raises:
        ValueError: ``frame`` is neither ``h`` nor ``s``.
        ValueError: ``atol`` is negative.
        ValueError: ``max_terms`` is not positive.
    """
    if max_terms < 1:
        raise ValueError("max_terms must be a positive integer.")
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")
    if len(rot_gates.gates) == 0:
        return operator.copy(), 0.0
    if frame not in ["h", "s"]:
        raise ValueError(f"frame must be 'h' or 's', not {frame}.")

    operator = operator.simplify(atol=atol)
    pauli_arr = np.concatenate([operator.paulis.x, operator.paulis.z], axis=1)

    # Lexsort in preparation for rust evolution function
    sorted_ids = np.lexsort(pauli_arr[:, ::-1].T)
    paulis, coeffs, trunc_onenorm = evolve_by_circuit_r(
        pauli_arr[sorted_ids],
        operator.coeffs[sorted_ids].astype(np.float64),
        np.array(rot_gates.gates),
        rot_gates.qargs,
        rot_gates.thetas,
        max_terms,
        atol,
        frame.lower(),
    )
    paulis_x = paulis[:, : operator.num_qubits]
    paulis_z = paulis[:, operator.num_qubits :]

    if len(coeffs) == 0:
        spo_out = SparsePauliOp(
            PauliList(["I" * operator.num_qubits]), [0], ignore_pauli_phase=True, copy=False
        )
    else:
        spo_out = SparsePauliOp(
            PauliList.from_symplectic(paulis_z, paulis_x),
            coeffs,
            ignore_pauli_phase=True,
            copy=False,
        )

    return spo_out, trunc_onenorm


def propagate_through_circuit(
    operator: SparsePauliOp,
    circuit: QuantumCircuit,
    max_terms: int,
    atol: float,
    frame: str,
) -> tuple[SparsePauliOp, float]:
    """Propagate a sparse Pauli operator, :math:`O`, through a circuit, :math:`U`.

    For Schrödinger propagation: :math:`U O U†`.

    For Heisenberg propagation: :math:`U† O U`.

    In general, the memory and time required for propagating through a circuit grows exponentially with the number of operations in the
    circuit due to the exponential growth in the number of terms of the operator in the Pauli basis. To regulate this exponential
    difficulty, one may truncate small Pauli terms (i.e. set them to zero), resulting in a bias proportional to the magnitudes of the
    truncated terms. After propagating through each operation in the circuit, terms are truncated with respect to two parameters:

    - Only the ``max_terms`` largest Pauli components are kept; any smaller terms will be truncated. This option makes it
      possible to estimate in advance how much time and memory will suffice for the computation.
    - Terms with magnitudes less than ``atol`` are truncated (set to zero).

    .. note::
        This function pre-allocates space in memory for the full-sized operator and operator buffer. It is the caller's
        responsibility to ensure they have enough memory to hold operators containing ``max_terms`` terms. When ``max_terms`` is
        ``None``, the memory and time requirements typically grow exponentially with the number of operations in the circuit.

    Args:
        operator: The operator to propagate
        circuit: The circuit through which the operator will be propagated
        max_terms: The maximum number of terms the operator may contain as it is propagated
        atol: Terms with coeff magnitudes less than this will not be added to the operator as it is propagated. This parameter is not a
            guarantee on the accuracy of the returned operator.
        frame:
            ``s`` for Schrödinger evolution
            ``h`` for Heisenberg evolution

    Returns:
        The evolved operator

    Raises:
        ValueError: ``frame`` is neither ``h`` nor ``s``.
        ValueError: ``atol`` is negative.
        ValueError: ``max_terms`` is not positive.
    """
    rot_gates = circuit_to_rotation_gates(circuit)
    return propagate_through_rotation_gates(operator, rot_gates, max_terms, atol, frame)


def propagate_through_operator(
    op1: SparsePauliOp,
    op2: SparsePauliOp,
    max_terms: int | None = None,
    coerce_op1_traceless: bool = False,
    num_leading_terms: int = 0,
    frame: str = "s",
    atol: float = 0.0,
) -> SparsePauliOp:
    """Propagate an operator, `op1` or :math:`O`, through another operator, `op2` or :math:`U`.

    For Schrödinger evolution: :math:`U O U†`.

    For Heisenberg evolution: :math:`U† O U`.

    Evolution is performed in the Pauli basis by summing terms of the form :math:`U_i O_j U_k`
    (neglecting the dagger, see note below). The number of such terms is cubic in operator size (`len(op1) * len(op2)**2`)
    and will generally include many duplicate Paulis.

    Setting `max_terms` produces an approximate result, where only the `max_terms` largest terms (in coefficient magnitude)
    are computed. This can be much faster but results in some error due to truncation of smaller terms.

    It is possible that some Paulis present in the kept terms would have also appeared in the truncated terms. Because such
    truncated terms are never computed, they cannot possibly be merged into the kept terms sharing the same Pauli. Thus, the
    ``n``th-largest term in the approximate result is not guaranteed to equal the nth-largest term in the exact result.
    Likewise, convergence to the exact result with increasing ``max_terms`` can be non-monotonic.

    .. note::

        :math:`O` is assumed to be Hermitian (:math:`O_j' = :math:`O_j†`)

    Args:
        op1: The operator to propagate
        op2: The operator through which to propagate
        max_terms: When not ``None``, produces an approximate result including only the ``max_terms`` largest terms in the
            direct product of the three operators in Pauli space.

            When ``max_terms`` is ``None`` and the number of qubits is < 12, the propagation will be performed in the computational
            basis using matrix multiplication. For systems > 12 qubits, all Pauli terms are computed and summed; however, this is
            usually not a good way to compute exact evolution due to the many duplicate terms present.

        coerce_op1_traceless: A flag denoting whether to remove identity terms from the output operator.
        num_leading_terms: The number of terms in ``op1`` to conjugate by every term in ``op2``. The set of included terms
            is expanded to include its union with the set of terms :math:`U_i O_j U_i†`, for :math:`j < num_leading_terms`.
            This can improve accuracy for the leading components of `O` in the output, at some computational runtime cost.
        frame:
            `s` for Schrödinger evolution
            `h` for Heisenberg evolution
        atol: Terms in the evolved operator with magnitudes below ``atol`` will be truncated

    Returns:
        The transformed operator

    Raises:
        ValueError: ``frame`` is neither ``s`` nor ``h``.
        ValueError: ``max_terms`` is not positive.
    """
    if frame == "h":
        op2 = op2.adjoint()
    elif frame == "s":
        pass
    else:
        raise ValueError(f"Expected frame either 's' or 'h', but got: {frame}")

    num_leads = min(num_leading_terms, len(op1))
    if max_terms is not None:
        # sort terms of each operator (descending by magnitude):
        ordering = np.argsort(np.abs(op1.coeffs))[::-1]
        to_evolve = SparsePauliOp(
            op1.paulis[ordering], op1.coeffs[ordering], ignore_pauli_phase=True, copy=False
        )
        ordering = np.argsort(np.abs(op2.coeffs))[::-1]
        other = SparsePauliOp(
            op2.paulis[ordering], op2.coeffs[ordering], ignore_pauli_phase=True, copy=False
        )

        kept_idx = _k_largest_products(
            to_evolve.coeffs,
            other.coeffs,
            max_terms,
            assume_op1_hermitian=True,
        )

        off_diag = kept_idx[:, 0] != kept_idx[:, 2]
        # axes: [term, operator]

        if num_leads > 0:
            # we can apply these terms with a faster method,
            # so we can remove any present in kept_idx (so they're not processed twice):
            kept_idx = kept_idx[np.logical_or(off_diag, kept_idx[:, 1] > num_leads - 1)]

        a_idx, b_idx, c_idx = kept_idx.T
        paulis_each_term = other.paulis[a_idx] @ to_evolve.paulis[b_idx] @ other.paulis[c_idx]
        coeffs_each_term = (
            other.coeffs[a_idx] * to_evolve.coeffs[b_idx] * other.coeffs[c_idx].conjugate()
        )

        product = SparsePauliOp(
            paulis_each_term, coeffs_each_term, ignore_pauli_phase=False, copy=False
        )

        assert np.all(kept_idx[:, 0] <= kept_idx[:, 2])
        assert np.all(product.paulis.phase == 0)
        off_diag = kept_idx[:, 0] != kept_idx[:, 2]
        product.coeffs[off_diag] = 2 * product.coeffs[off_diag].real

        if num_leads > 0:
            # now apply all with the faster method:
            other_mags = np.abs(other.coeffs) ** 2
            anticomm_matrix = _commutation_matrix(
                to_evolve.paulis[:num_leads], other.paulis, negate=True
            )
            anticomm_matrix = (-1) ** anticomm_matrix
            net_coeffs = anticomm_matrix @ other_mags
            product += SparsePauliOp(
                to_evolve.paulis[:num_leads],
                to_evolve.coeffs[:num_leads] * net_coeffs,
                copy=False,
                ignore_pauli_phase=True,
            )

    else:
        if op1.num_qubits <= 12:
            op2 = op2.to_matrix()
            product_mat = op2 @ op1.to_matrix() @ op2.conj().T
            product = SparsePauliOp.from_operator(product_mat, atol=0.0, rtol=0.0)
        else:
            product = op1.compose(op2)
            product = product.simplify(atol=0.0)
            product = product.dot(op2.adjoint())
        # Chop any numerical noise
        product.coeffs.imag = 0

    if coerce_op1_traceless:
        non_id_mask = np.any(product.paulis.z | product.paulis.x, axis=1)
        if not np.all(non_id_mask):
            product = product[non_id_mask]

    return product.simplify(atol=atol)


def _k_largest_products(
    arr1: np.ndarray, arr2: np.ndarray, k: int, assume_op1_hermitian: bool
) -> np.ndarray:
    """Find indices, `(l, m, n)`, of the ``k`` largest terms.

    The value associated with each index triplet is calculated as:

    ``arr2[l] * arr1[m] * arr2[n]``.

    Returns up to ``k`` triplets of indices ``(l_i, m_i, n_i)``, such that
    ``arr2[l_i] * arr1[m_i] * arr2[n_i]`` is the ``i``th largest element in
    the outer product.

    arr1, arr2 are assumed to already be sorted decreasing by magnitude.
    """
    mags = np.abs(arr1)
    if not np.all(mags[:-1] >= mags[1:]):
        raise ValueError("arr1 should be sorted descending by magnitude.")
    mags = np.abs(arr2)
    if not np.all(mags[:-1] >= mags[1:]):
        raise ValueError("arr2 should be sorted descending by magnitude.")
    arr1 = np.asarray(arr1, dtype=np.complex128)
    arr2 = np.asarray(arr2, dtype=np.complex128)

    ind_out: np.ndarray = np.array(k_largest_products_r(arr1, arr2, k, assume_op1_hermitian))
    return ind_out
