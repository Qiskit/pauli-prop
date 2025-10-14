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

"""Test propagation module."""

from __future__ import annotations

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from pauli_prop import (
    RotationGates,
    circuit_to_rotation_gates,
    evolve_through_cliffords,
    propagate_through_circuit,
    propagate_through_operator,
    propagate_through_rotation_gates,
)
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RZZGate
from qiskit.quantum_info import Clifford, Operator, PauliList, SparsePauliOp


def _pauli_dict(op: SparsePauliOp) -> dict[str, complex]:
    """Helper returning a label -> coefficient mapping."""

    simplified = op.simplify(atol=1e-12)
    paulis = simplified.paulis.to_labels()
    coeffs = simplified.coeffs.astype(complex)
    return {label: coeff for label, coeff in zip(paulis, coeffs, strict=True)}


class TestPropagation(unittest.TestCase):
    """Tests covering the most visible Pauli propagation entry points."""

    def test_propagate_through_circuit_matches_matrix(self):
        """The Rust accelerated evolution should agree with matrix conjugation."""

        circuit = QuantumCircuit(1)
        circuit.ry(math.pi / 3, 0)
        operator = SparsePauliOp.from_list([("Z", 1.0)])

        evolved, trunc_norm = propagate_through_circuit(
            operator, circuit, max_terms=8, atol=1e-12, frame="s"
        )

        unitary = Operator(circuit).data
        expected_matrix = unitary @ operator.to_matrix() @ unitary.conj().T
        expected = SparsePauliOp.from_operator(expected_matrix, atol=1e-12, rtol=0.0)

        self.assertEqual(trunc_norm, 0.0)
        evolved_dict = _pauli_dict(evolved)
        expected_dict = _pauli_dict(expected)
        self.assertSetEqual(set(evolved_dict.keys()), set(expected_dict.keys()))
        evolved_coeffs = np.array([evolved_dict[key] for key in sorted(evolved_dict)])
        expected_coeffs = np.array([expected_dict[key] for key in sorted(expected_dict)])
        assert_allclose(evolved_coeffs, expected_coeffs)

    def test_propagate_through_rotation_gates_heisenberg(self):
        """Heisenberg frame evolution should align with direct calculation."""

        circuit = QuantumCircuit(1)
        circuit.rx(math.pi / 7, 0)
        rot_gates = circuit_to_rotation_gates(circuit)
        operator = SparsePauliOp.from_list([("X", 0.75), ("Z", -0.25)])

        evolved, trunc_norm = propagate_through_rotation_gates(
            operator, rot_gates, max_terms=8, atol=1e-12, frame="h"
        )

        unitary = Operator(circuit).data
        expected_matrix = unitary.conj().T @ operator.to_matrix() @ unitary
        expected = SparsePauliOp.from_operator(expected_matrix, atol=1e-12, rtol=0.0)

        self.assertEqual(trunc_norm, 0.0)
        evolved_dict = _pauli_dict(evolved)
        expected_dict = _pauli_dict(expected)
        self.assertSetEqual(set(evolved_dict.keys()), set(expected_dict.keys()))
        evolved_coeffs = np.array([evolved_dict[key] for key in sorted(evolved_dict)])
        expected_coeffs = np.array([expected_dict[key] for key in sorted(expected_dict)])
        assert_allclose(evolved_coeffs, expected_coeffs)

    def test_rotation_gates_with_clifford(self):
        """Test the RotationGates handling with an extracted Clifford component.

        This is a regression test against https://github.com/Qiskit/pauli-prop/issues/25.
        """
        cnot = QuantumCircuit(2)
        cnot.cx(0, 1)
        clifford = Clifford.from_circuit(cnot)

        num_qubits = 2
        circuit = QuantumCircuit(num_qubits)
        circuit.rx(math.pi / 7, 0)

        rot_gates = RotationGates([], [], [])
        for inst in circuit.data:
            rot_gates.append_circuit_instruction(
                inst, [qb._index for qb in inst.qubits], num_qubits, clifford=clifford
            )

        operator = SparsePauliOp.from_list([("IX", 0.75), ("ZI", -0.25)])

        evolved, trunc_norm = propagate_through_rotation_gates(
            operator, rot_gates, max_terms=8, atol=1e-12, frame="h"
        )

        cliff_op = Operator(clifford).data
        unitary = cliff_op.conj().T @ Operator(circuit).data @ cliff_op
        expected_matrix = unitary.conj().T @ operator.to_matrix() @ unitary
        expected = SparsePauliOp.from_operator(expected_matrix, atol=1e-12, rtol=0.0)

        self.assertEqual(trunc_norm, 0.0)
        evolved_dict = _pauli_dict(evolved)
        expected_dict = _pauli_dict(expected)
        self.assertSetEqual(set(evolved_dict.keys()), set(expected_dict.keys()))
        evolved_coeffs = np.array([evolved_dict[key] for key in sorted(evolved_dict)])
        expected_coeffs = np.array([expected_dict[key] for key in sorted(expected_dict)])
        assert_allclose(evolved_coeffs, expected_coeffs)

    def test_evolve_through_cliffords_decomposition(self):
        """The returned Clifford and remainder circuit should reconstruct the input."""

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.append(RZZGate(math.pi / 5), [0, 1])
        circuit.s(1)

        clifford, non_cliffords = evolve_through_cliffords(circuit)

        reconstructed = Operator(non_cliffords).data @ Operator(clifford.to_circuit()).data
        expected = Operator(circuit).data

        assert_allclose(reconstructed, expected)

        # All instructions in the remainder should be Pauli rotations.
        allowed_ops = {"PauliEvolution", "rzz"}
        for inst in non_cliffords:
            self.assertIn(inst.operation.name, allowed_ops)

    def test_propagate_through_operator_exact_and_traceless(self):
        """Exact propagation should agree with matrix conjugation and support traceless outputs."""

        op1 = SparsePauliOp.from_list([(label, coeff) for label, coeff in [("I", 0.2), ("Z", 0.6)]])
        op2 = SparsePauliOp.from_list([(label, coeff) for label, coeff in [("X", 0.5), ("Z", 1.0)]])

        evolved = propagate_through_operator(op1, op2, max_terms=None, coerce_op1_traceless=True)

        matrix_evolved = op2.to_matrix() @ op1.to_matrix() @ op2.to_matrix().conj().T
        expected = SparsePauliOp.from_operator(matrix_evolved, atol=1e-12, rtol=0.0)
        expected = expected.simplify(atol=1e-12)

        evolved_dict = _pauli_dict(evolved)
        self.assertNotIn("I", evolved_dict)

        expected_dict = _pauli_dict(expected)
        expected_dict.pop("I", None)
        self.assertSetEqual(set(evolved_dict.keys()), set(expected_dict.keys()))
        evolved_coeffs = np.array([evolved_dict[key] for key in sorted(evolved_dict)])
        expected_coeffs = np.array([expected_dict[key] for key in sorted(expected_dict)])
        assert_allclose(evolved_coeffs, expected_coeffs)

    def test_propagate_through_operator_truncation(self):
        """When a cutoff is requested the operator should be truncated accordingly."""

        op1 = SparsePauliOp(PauliList(["X", "Y"]), coeffs=[0.5, 0.5])
        op2 = SparsePauliOp(PauliList(["X", "Y", "Z"]), coeffs=[1.0, 0.8, 0.6])

        truncated = propagate_through_operator(
            op1, op2, max_terms=2, frame="s", atol=0.0, search_step=1
        )

        self.assertLessEqual(len(truncated), 2)

        ordering = np.argsort(np.abs(op1.coeffs))[::-1]
        op1_sorted = SparsePauliOp(
            op1.paulis[ordering], op1.coeffs[ordering], ignore_pauli_phase=True, copy=False
        )
        ordering = np.argsort(np.abs(op2.coeffs))[::-1]
        op2_sorted = SparsePauliOp(
            op2.paulis[ordering], op2.coeffs[ordering], ignore_pauli_phase=True, copy=False
        )

        combos: list[tuple[int, int, int, complex]] = []
        for a_idx in range(len(op2_sorted)):
            for b_idx in range(len(op1_sorted)):
                for c_idx in range(len(op2_sorted)):
                    coeff = (
                        op2_sorted.coeffs[a_idx]
                        * op1_sorted.coeffs[b_idx]
                        * op2_sorted.coeffs[c_idx].conjugate()
                    )
                    combos.append((a_idx, b_idx, c_idx, coeff))
        combos.sort(key=lambda item: abs(item[3]), reverse=True)
        combos = combos[:2]

        paulis = []
        coeffs = []
        for a_idx, b_idx, c_idx, coeff in combos:
            pauli = op2_sorted.paulis[a_idx] @ op1_sorted.paulis[b_idx] @ op2_sorted.paulis[c_idx]
            if a_idx != c_idx:
                coeff = 2 * coeff.real
            coeff *= (-1j) ** pauli.phase
            pauli.phase = 0
            paulis.append(pauli)
            coeffs.append(coeff)

        expected = SparsePauliOp(PauliList(paulis), coeffs, ignore_pauli_phase=True)
        expected = expected.simplify(atol=0.0)

        truncated_dict = _pauli_dict(truncated)
        expected_dict = _pauli_dict(expected)
        self.assertEqual(truncated_dict, expected_dict)
