//! # CPT Engine
//!
//! A library for performing Pauli propagation

use hashbrown::HashSet;
use num_complex::{Complex, ComplexFloat};
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;
use pyo3::Bound;
use rustc_hash::FxHasher;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::BuildHasherDefault;

type FxHashSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;
type Entry = (OrderedFloat<f64>, usize, usize, usize);

const PHASE_MOD: u8 = 4;

/// Find the `k` largest index triplets ,`(l, m, n)`, such that
/// `abs(other[l] @ to_evolve[m] @ other[n])` is maximized.
/// If `assume_hermitian` is True, caller must replace off-diagonal terms `l != n`
/// specified in result with twice their real parts (see code and comments).
#[pyfunction]
fn k_largest_products(
    py: Python,
    to_evolve: PyReadonlyArray1<Complex<f64>>,
    other: PyReadonlyArray1<Complex<f64>>,
    mut k: usize,
    assume_hermitian: bool,
) -> PyResult<Py<PyArray2<i64>>> {
    // Convert np arrays to Rust slices
    let evo_slice = to_evolve.as_slice()?;
    let other_slice = other.as_slice()?;
    let evo_len = evo_slice.len();
    let other_len = other_slice.len();

    let max_possible: usize = other_len.pow(2) * evo_len;
    if k > max_possible {
        k = max_possible;
    }

    // Collect the coeffs into real-valued arrays
    let other_abs: Vec<f64> = other_slice.iter().map(|c| c.abs()).collect();
    let evo_abs: Vec<f64> = evo_slice.iter().map(|c| c.abs()).collect();

    // Create the frontier queue
    let mut frontier: BinaryHeap<Entry> = BinaryHeap::with_capacity(3 * k);

    // Ensure we don't visit a node more than once
    let mut visited: FxHashSet<[u64; 2]> =
        FxHashSet::with_capacity_and_hasher(3 * k, BuildHasherDefault::<FxHasher>::default());

    // Compute the magnitude for (0, 0, 0) and push it onto the queue
    let mag0 = other_abs[0] * evo_abs[0] * other_abs[0];

    frontier.push((OrderedFloat(mag0), 0, 0, 0));

    // Output vector holding the ordered index triplets
    let mut keepers: Vec<[i64; 3]> = Vec::with_capacity(k);

    let mut num_kept = 0;
    while !frontier.is_empty() && num_kept < k {
        // Get largest node and add its indices to the keepers list
        let (_mag, l, m, n) = frontier.pop().unwrap();
        keepers.push([l as i64, m as i64, n as i64]);
        if assume_hermitian && (l != n) {
            num_kept += 2;
        } else {
            num_kept += 1;
        }

        // Add elements from the frontier to the queue.
        // If assume_hermitian is True, we ignore terms in the upper-triangle l > n,
        // with the understanding that the caller will replace lower-triangle terms
        // by twice their real parts.
        if l + 1 < other_len && (!assume_hermitian || (l + 1) <= n) {
            if visited.insert(make_key(l + 1, m, n)) {
                let new_mag = other_abs[l + 1] * evo_abs[m] * other_abs[n];
                frontier.push((OrderedFloat(new_mag), l + 1, m, n));
            }
        }
        if m + 1 < evo_len && (!assume_hermitian || l <= n) {
            if visited.insert(make_key(l, m + 1, n)) {
                let new_mag = other_abs[l] * evo_abs[m + 1] * other_abs[n];
                frontier.push((OrderedFloat(new_mag), l, m + 1, n));
            }
        }
        if n + 1 < other_len && (!assume_hermitian || l <= n + 1) {
            if visited.insert(make_key(l, m, n + 1)) {
                let new_mag = other_abs[l] * evo_abs[m] * other_abs[n + 1];
                frontier.push((OrderedFloat(new_mag), l, m, n + 1));
            }
        }
    }
    // Get a borrowed view of the output array
    let flat: Vec<i64> = keepers.into_iter().flatten().collect();
    let arr2: Array2<i64> =
        Array2::from_shape_vec((flat.len() / 3, 3), flat).expect("Invalid shape");
    let py_arr = PyArray2::from_owned_array(py, arr2);

    // Return an owned, GIL-independent reference to the output array
    Ok(py_arr.to_owned().into())
}

/// Stuff the index triplets into two integers for faster hashing
#[inline(always)]
fn make_key(i: usize, j: usize, k: usize) -> [u64; 2] {
    let a = ((i as u64) << 22) | ((j as u64) >> 20);
    let b = (((j as u64) & ((1 << 20) - 1)) << 44) | (k as u64);
    [a, b]
}

/// Python function for evolving an operator through a circuit.
///
/// The operator consists of Pauli terms over `ceil(num_qubits / 64)` integers, and a
/// real-valued coefficient.
///
/// Each gate in the circuit is represented by `ceil(num_qubits / 64)` integers, a rotation
/// angle, `theta`, and an array of `qargs`.
///
/// `max_terms` controls how big the operator can get during evolution. It is the user's
/// responsibility to ensure they have enough RAM to hold the operator data and the
/// equivalently sized buffer.
///
/// `atol` controls the magnitude a new term's coefficient must be to remain in the operator.
///
/// `frame` can be:
///     `s` for Schrodinger evolution: `U(θ) O U(θ)†`
///     `h` for Heisenberg evolution: `U(θ)† O U(θ)`
#[pyfunction]
fn evolve_by_circuit(
    py: Python<'_>,
    operator: PyReadonlyArray2<bool>,
    coeffs: PyReadonlyArray1<f64>,
    gates: PyReadonlyArray2<bool>,
    qargs: Vec<Vec<usize>>,
    thetas: Vec<f64>,
    max_terms: usize,
    atol: f64,
    frame: char,
) -> PyResult<(Py<PyArray2<bool>>, Py<PyArray1<f64>>, f64)> {
    // Prepare the fields for the CPTOperatorRust struct
    let (_, num_cols) = operator.as_array().dim();
    let num_qubits = num_cols / 2;
    let ints_per_pauli = (2 * num_qubits + 63) / 64;
    let operator = np_to_cpt(operator, max_terms);
    let gates = np_to_cpt(gates, max_terms);
    let coeffs: Vec<f64> = coeffs.as_array().to_owned().into_iter().collect();
    let paulis_buffer: Vec<u64> = Vec::with_capacity(ints_per_pauli * max_terms);
    let coeffs_buffer: Vec<f64> = Vec::with_capacity(max_terms);

    // Instantiate the internal operator
    let mut cpt_op = CPTOperatorRust {
        paulis: operator,
        coeffs: coeffs,
        max_terms: max_terms,
        num_qubits: num_qubits,
        ints_per_pauli: ints_per_pauli,
        atol: atol,
        paulis_buffer: paulis_buffer,
        coeffs_buffer: coeffs_buffer,
    };

    let mut trunc_onenorm = 0.0;
    // Release the GIL and evolve the operator through the circuit
    let num_gates = thetas.len();
    py.detach(|| {
        for i in 0..num_gates {
            let mut id = i;
            if frame == 'h' {
                id = num_gates - 1 - i
            };
            let theta = thetas[id];
            let gate = &gates[ints_per_pauli * id..(id + 1) * ints_per_pauli];
            let qarg = &qargs[id];
            trunc_onenorm += cpt_op.evolve_by_pauli_rotation(gate, theta, qarg, frame);
        }
    });

    // Prepare output numpy arrays and return
    let num_terms = cpt_op.paulis.len() / ints_per_pauli;
    let unpacked_paulis = unpack_pauli_ints_flat(&cpt_op.paulis, num_terms, num_qubits);
    let output_paulis =
        Array2::<bool>::from_shape_vec((num_terms, 2 * num_qubits), unpacked_paulis).unwrap();
    let output_coeffs = Array1::<f64>::from_vec(cpt_op.coeffs);

    Ok((
        output_paulis.into_pyarray(py).to_owned().into(),
        output_coeffs.into_pyarray(py).to_owned().into(),
        trunc_onenorm,
    ))
}

/// Convert an array of Pauli terms (XZ bitstrings in the rows) to bit-packed u64
fn np_to_cpt(pauli_array: PyReadonlyArray2<bool>, max_terms: usize) -> Vec<u64> {
    let pauli_array = pauli_array.as_array();
    let (_num_rows, num_cols) = pauli_array.dim();
    let ints_per_pauli = (num_cols + 63) / 64;

    let mut packed_paulis: Vec<u64> = Vec::with_capacity(ints_per_pauli * max_terms);

    for row in pauli_array.outer_iter() {
        let mut row_iter = row.iter().copied();
        for _ in 0..ints_per_pauli {
            let mut word = 0u64;
            for bit_id in 0..64 {
                if let Some(bit) = row_iter.next() {
                    if bit {
                        word |= 1 << bit_id;
                    }
                }
            }
            packed_paulis.push(word);
        }
    }
    packed_paulis
}

/// Unpack the ints into their flattened Vec<bool> representation
fn unpack_pauli_ints_flat<'py>(
    packed_paulis: &[u64],
    num_terms: usize,
    num_qubits: usize,
) -> Vec<bool> {
    let ipp = (2 * num_qubits + 63) / 64;
    let mut unpacked_paulis = vec![false; num_terms * num_qubits * 2];

    for term_id in 0..num_terms {
        for int_id in 0..ipp {
            let word = packed_paulis[term_id * ipp + int_id];
            let base_bit_id = int_id * 64;
            for bit in 0..64 {
                let bit_id = base_bit_id + bit;
                if bit_id >= 2 * num_qubits {
                    break;
                }
                unpacked_paulis[term_id * 2 * num_qubits + bit_id] = (word >> bit) & 1 == 1;
            }
        }
    }
    unpacked_paulis
}

/// A struct for performing Pauli propagation.
#[derive(Debug)]
pub struct CPTOperatorRust {
    paulis: Vec<u64>,
    coeffs: Vec<f64>,
    max_terms: usize,
    num_qubits: usize,
    ints_per_pauli: usize,
    atol: f64,
    paulis_buffer: Vec<u64>,
    coeffs_buffer: Vec<f64>,
}

impl CPTOperatorRust {
    /// Remove terms with coeff magnitudes < `atol`, keeping, at most, the top-`max_terms` terms.
    fn truncate(&mut self) -> f64 {
        // Get indices sorted wrt coefficient magnitude
        let n = self.coeffs.len();
        let k = self.max_terms.min(n);
        let mut indices: Vec<usize> = (0..n).collect();
        if k < n {
            indices.select_nth_unstable_by(k, |&i, &j| {
                self.coeffs[i]
                    .abs()
                    .partial_cmp(&self.coeffs[j].abs())
                    .unwrap()
                    .reverse()
            });
        }

        // Prepare buffers for re-use
        self.paulis_buffer.clear();
        self.coeffs_buffer.clear();
        self.paulis_buffer
            .reserve(self.ints_per_pauli * self.max_terms);
        self.coeffs_buffer.reserve(self.max_terms);

        // Stream the largest-magnitude terms into the new array.
        // We sort the new indices, so they remain sorted in new array.
        indices[..k].sort_unstable();

        // Compute the L1-norm of the truncated coefficients as we wish to return this later.
        let mut trunc_onenorm = indices[k..].iter().map(|&i| self.coeffs[i].abs()).sum();

        let ipp = self.ints_per_pauli;
        for &i in &indices[..k] {
            let c = self.coeffs[i].abs();
            if c > self.atol {
                self.paulis_buffer
                    .extend_from_slice(&self.paulis[i * ipp..(i + 1) * ipp]);
                self.coeffs_buffer.push(self.coeffs[i]);
            } else {
                trunc_onenorm += c.abs();
            }
        }

        // Update internal data
        std::mem::swap(&mut self.paulis, &mut self.paulis_buffer);
        std::mem::swap(&mut self.coeffs, &mut self.coeffs_buffer);

        trunc_onenorm
    }

    /// Evolve self (`O`) through a Pauli rotation, `U(θ)`.
    ///
    /// `frame` can be:
    ///     `s` for Schrodinger evolution: `U(θ) O U(θ)†`
    ///     `h` for Heisenberg evolution: `U(θ)† O U(θ)`
    ///
    /// This technique is described in the Methods section of this paper:
    /// https://www.science.org/doi/full/10.1126/sciadv.adk4321#
    fn evolve_by_pauli_rotation(
        &mut self,
        other: &[u64],
        mut theta: f64,
        qargs: &[usize],
        frame: char,
    ) -> f64 {
        let ipp = self.ints_per_pauli;
        let anticomm_ids = get_anticommuting(&self.paulis, &other, qargs, self.num_qubits, ipp);

        if frame == 's' {
            theta *= -1.0;
        }

        let mut trunc_onenorm = 0.0;

        // Get all the new terms to apply to the operator
        let mut new_terms: Vec<(Vec<u64>, f64)> = anticomm_ids
            .iter()
            .filter_map(|&idx| {
                let mut new_coeff = self.coeffs[idx] * theta.sin();
                if new_coeff.abs() > self.atol {
                    let pauli_slice = &self.paulis[ipp * idx..(idx + 1) * ipp];
                    let (new_pauli, phase_shift) =
                        multiply_paulis(pauli_slice, other, qargs, self.num_qubits, false);
                    let new_phase = (phase_shift + 3) & (PHASE_MOD - 1);
                    assert!(
                        new_phase % 2 == 0,
                        "CPT sin branch term should be real valued."
                    );
                    if new_phase == 2 {
                        new_coeff *= -1.0;
                    }
                    return Some((new_pauli, new_coeff));
                } else {
                    trunc_onenorm += new_coeff.abs();
                    return None;
                }
            })
            .collect();

        // Scale anticommuting terms by cos factor
        for pauli_id in &anticomm_ids {
            self.coeffs[*pauli_id] *= theta.cos();
        }

        // Apply the new terms to the operator. This method inherently de-duplicates.
        // The operator can outgrow `max_terms` here before being truncated in the
        // next step.
        trunc_onenorm += self.insert_or_combine(&mut new_terms);

        // Drop small terms and ensure the operator has fewer than max_terms terms
        trunc_onenorm += self.truncate();

        trunc_onenorm
    }

    /// Insert new terms into the operator.
    ///
    /// If a term already exists, the coeffs are summed. The lexicographical ordering
    /// of the Pauli terms will be maintained as terms are added. Terms with coeffs below
    /// `self.atol` will not be added.
    ///
    /// Returns the sum of coefficients that are dropped because their magnitude is below `atol`.
    fn insert_or_combine(&mut self, new_terms: &mut [(Vec<u64>, f64)]) -> f64 {
        // Sort the new terms so we can stream them into the new operator
        let ipp = self.ints_per_pauli;
        let num_paulis = self.paulis.len() / ipp;
        new_terms.sort_unstable_by(|a, b| cmp_paulis(&a.0, &b.0));

        // Prepare buffers for reuse
        self.paulis_buffer.clear();
        self.coeffs_buffer.clear();
        self.paulis_buffer
            .reserve((self.coeffs.len() + new_terms.len()) * ipp);
        self.coeffs_buffer
            .reserve(self.coeffs.len() + new_terms.len());

        let mut trunc_onenorm = 0.0;

        // Stream the current operator (sorted), along w the sorted new terms into a new array
        let mut i = 0;
        let mut j = 0;
        while i < num_paulis && j < new_terms.len() {
            let existing = &self.paulis[i * ipp..(i + 1) * ipp];
            let new = &new_terms[j].0;
            match cmp_paulis(existing, new) {
                Ordering::Less => {
                    self.paulis_buffer.extend_from_slice(existing);
                    self.coeffs_buffer.push(self.coeffs[i]);
                    i += 1;
                }
                Ordering::Greater => {
                    self.paulis_buffer.extend_from_slice(new);
                    self.coeffs_buffer.push(new_terms[j].1);
                    j += 1;
                }
                Ordering::Equal => {
                    let new_coeff = self.coeffs[i] + new_terms[j].1;
                    if new_coeff.abs() > self.atol {
                        self.paulis_buffer.extend_from_slice(existing);
                        self.coeffs_buffer.push(new_coeff);
                    } else {
                        trunc_onenorm += new_coeff.abs();
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        // Stream any leftover Paulis into the operator
        while i < num_paulis {
            self.paulis_buffer
                .extend_from_slice(&self.paulis[i * ipp..(i + 1) * ipp]);
            self.coeffs_buffer.push(self.coeffs[i]);
            i += 1;
        }
        while j < new_terms.len() {
            self.paulis_buffer.extend_from_slice(&new_terms[j].0);
            self.coeffs_buffer.push(new_terms[j].1);
            j += 1;
        }

        // Update internal data
        std::mem::swap(&mut self.paulis, &mut self.paulis_buffer);
        std::mem::swap(&mut self.coeffs, &mut self.coeffs_buffer);

        trunc_onenorm
    }
}

/// Compare two length-N Pauli chunks starting with least significant word.
#[inline(always)]
fn cmp_paulis(a: &[u64], b: &[u64]) -> Ordering {
    for i in 0..a.len() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            not_eq => return not_eq,
        }
    }
    Ordering::Equal
}

/// Find the indices of terms in `paulis` which anticommute with `other`.
fn get_anticommuting(
    paulis: &[u64],
    other: &[u64],
    qargs: &[usize],
    num_qubits: usize,
    ints_per_pauli: usize,
) -> Vec<usize> {
    paulis
        .chunks_exact(ints_per_pauli)
        .enumerate()
        .filter_map(|(pauli_id, p)| {
            let mut anticomm_flag = false;
            for &q in qargs {
                let int_id_x = q / 64;
                let int_id_z = (q + num_qubits) / 64;
                let bit_id_x = q % 64;
                let bit_id_z = (q + num_qubits) % 64;

                let x1 = (p[int_id_x] >> bit_id_x) & 1 != 0;
                let z1 = (p[int_id_z] >> bit_id_z) & 1 != 0;
                let x2 = (other[int_id_x] >> bit_id_x) & 1 != 0;
                let z2 = (other[int_id_z] >> bit_id_z) & 1 != 0;

                if (x1 && z2) ^ (z1 && x2) {
                    anticomm_flag = !anticomm_flag;
                }
            }
            if anticomm_flag {
                Some(pauli_id)
            } else {
                None
            }
        })
        .collect()
}

/// Multiply two Paulis, `a @ b` if `front=true`, and `b @ a` if `front=false`.
/// Return the resulting Pauli, `p`, and the phase, `q`, such that `a @ b = (-i^q)p`.
fn multiply_paulis(
    a: &[u64],
    b: &[u64],
    qargs: &[usize],
    num_qubits: usize,
    front: bool,
) -> (Vec<u64>, u8) {
    // I: 0, Z: 1 X: 2, Y: 3
    const PAULI_MULT_LOOKUP: [[(u8, u8); 4]; 4] = [
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(1, 0), (0, 0), (3, 3), (2, 1)],
        [(2, 0), (3, 1), (0, 0), (1, 3)],
        [(3, 0), (2, 3), (1, 1), (0, 0)],
    ];

    // CREATE OUTPUT DATA STRUCTURES
    let mut out = a.to_vec();
    let mut phase_out: u8 = 0;

    // Multiply the Paulis and collect total phase
    for &q in qargs {
        let int_id_x = q / 64;
        let int_id_z = (q + num_qubits) / 64;
        let bit_id_x = q % 64;
        let bit_id_z = (q + num_qubits) % 64;
        let mask_x = 1u64 << bit_id_x;
        let mask_z = 1u64 << bit_id_z;

        let mut a_lookup =
            (((a[int_id_x] & mask_x) != 0) as u8) << 1 | ((a[int_id_z] & mask_z) != 0) as u8;
        let mut b_lookup =
            (((b[int_id_x] & mask_x) != 0) as u8) << 1 | ((b[int_id_z] & mask_z) != 0) as u8;

        if !front {
            std::mem::swap(&mut a_lookup, &mut b_lookup);
        }

        let (new_p, phase) = PAULI_MULT_LOOKUP[a_lookup as usize][b_lookup as usize];
        phase_out = (phase_out + phase) % PHASE_MOD;

        // Convert Pauli term to X/Z parts and apply it to output Pauli
        let new_x = (new_p >> 1) & 1;
        let new_z = new_p & 1;
        out[int_id_x] &= !mask_x;
        out[int_id_z] &= !mask_z;
        if new_x != 0 {
            out[int_id_x] |= mask_x;
        }
        if new_z != 0 {
            out[int_id_z] |= mask_z;
        }
    }
    (out, phase_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    #[test]
    fn test_evolve_by_pauli_rotation_s() {
        let paulis: Vec<u64> = vec![12];
        let coeffs: Vec<f64> = vec![1.0];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 100,
            num_qubits: 2,
            ints_per_pauli: 1,
            atol: 1e-12,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        // Create ZZ operator
        // Create circuit: ry->rx->rzz
        let theta1q = 3.0 * PI / 8.0;
        let theta2q = PI / 2.0;
        let rzzgate = vec![12];
        let rxgate = vec![1];
        let rygate = vec![5];

        // Evolve operator through the 3 gates (Schrodinger evolution)
        cpt_op.evolve_by_pauli_rotation(&rygate, theta1q, &vec![0], 's');
        cpt_op.evolve_by_pauli_rotation(&rxgate, theta1q, &vec![0], 's');
        cpt_op.evolve_by_pauli_rotation(&rzzgate, theta2q, &vec![0, 1], 's');
        // Created expected evolved operator
        let expected_evolved_op: Vec<u64> = vec![1, 5, 12];
        let expected_coeffs: Vec<f64> =
            vec![0.35355339059327384, 0.9238795325112867, 0.1464466094067263];

        assert_eq!(expected_evolved_op, cpt_op.paulis);
        assert_eq!(expected_coeffs, cpt_op.coeffs);
    }
    #[test]
    fn stress_test_evolve_by_pauli_rotation() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use rand_distr::{StandardNormal, Uniform};
        use std::f64::consts::PI;

        let num_qubits = 10;
        let ints_per_pauli = (2 * num_qubits + 63) / 64;
        let num_terms = 700;
        let max_terms = 700;

        let mut rng = StdRng::seed_from_u64(42); // Optional: make deterministic
        let dist = Uniform::new(3.0 * PI / 16.0, 3.0 * PI / 8.0).unwrap();

        let mut paulis = Vec::with_capacity(num_terms * ints_per_pauli);
        let mut coeffs = Vec::with_capacity(num_terms);
        let mut gates = Vec::with_capacity(num_terms * ints_per_pauli);
        let mut thetas = Vec::with_capacity(num_terms);
        let mut qargs: Vec<Vec<usize>> = Vec::with_capacity(num_terms);

        for _ in 0..num_terms {
            let p: Vec<u64> = (0..ints_per_pauli).map(|_| rng.random()).collect();
            let g: Vec<u64> = (0..ints_per_pauli).map(|_| rng.random()).collect();
            paulis.extend_from_slice(&p);
            gates.extend_from_slice(&g);
            coeffs.push(rng.sample(StandardNormal));
            thetas.push(rng.sample(dist));
            let num_qargs = rng.random_range(1..=3);
            let args: Vec<usize> = (0..num_qargs)
                .map(|_| rng.random_range(0..num_qubits))
                .collect();
            qargs.push(args);
        }

        let mut cpt_op = CPTOperatorRust {
            paulis,
            coeffs,
            max_terms,
            num_qubits,
            ints_per_pauli,
            atol: 1e-12,
            paulis_buffer: Vec::with_capacity(ints_per_pauli * max_terms),
            coeffs_buffer: Vec::with_capacity(max_terms),
        };

        for i in 0..num_terms {
            let theta = thetas[i];
            let gate = &gates[i * ints_per_pauli..(i + 1) * ints_per_pauli];
            let qarg = &qargs[i];
            cpt_op.evolve_by_pauli_rotation(gate, theta, qarg, 's');
        }
    }
    #[test]
    fn test_evolve_by_pauli_rotation_h() {
        // Create ZZ operator
        let paulis: Vec<u64> = vec![12];
        let coeffs: Vec<f64> = vec![1.0];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 100,
            num_qubits: 2,
            ints_per_pauli: 1,
            atol: 1e-12,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        // Create circuit: ry->rx->rzz
        let theta1q = 3.0 * PI / 8.0;
        let theta2q = PI / 2.0;
        let rzzgate = vec![12];
        let rxgate = vec![1];
        let rygate = vec![5];

        // Evolve operator through the 3 gates (Heisenberg evolution)
        cpt_op.evolve_by_pauli_rotation(&rzzgate, theta2q, &vec![0, 1], 'h');
        cpt_op.evolve_by_pauli_rotation(&rxgate, theta1q, &vec![0], 'h');
        cpt_op.evolve_by_pauli_rotation(&rygate, theta1q, &vec![0], 'h');
        // Created expected evolved operator
        let expected_evolved_op: Vec<u64> = vec![9, 12, 13];
        let expected_coeffs: Vec<f64> =
            vec![-0.35355339059327384, 0.1464466094067263, 0.9238795325112867];

        assert_eq!(expected_evolved_op, cpt_op.paulis);
        assert_eq!(expected_coeffs, cpt_op.coeffs);
    }
    #[test]
    fn test_evolve_by_pauli_rotation_multi_int() {
        // 38-qubit Z operator, Z on qubit id 26
        let paulis: Vec<u64> = vec![0, 1];
        let coeffs: Vec<f64> = vec![1.0];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 100,
            num_qubits: 38,
            ints_per_pauli: 2,
            atol: 1e-14,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        // Create mirror circuit: ry(theta)->ry(-theta),
        let theta1q = PI / 6.0;
        let rygate = vec![1 << 26, 1];
        // Evolve operator through the gates
        cpt_op.evolve_by_pauli_rotation(&rygate, theta1q, &vec![26], 'h');
        cpt_op.evolve_by_pauli_rotation(&rygate, -theta1q, &vec![26], 'h');
        // Created expected evolved operator
        let expected_evolved_op: Vec<u64> = vec![0, 1];
        let expected_coeffs: Vec<f64> = vec![1.0];

        assert_eq!(expected_evolved_op, cpt_op.paulis);
        assert_eq!(expected_coeffs, cpt_op.coeffs);
    }
    #[test]
    fn test_truncate() {
        // Test max_terms
        let paulis: Vec<u64> = vec![1, 2, 3, 4, 5, 6];
        let coeffs: Vec<f64> = vec![-0.9, 0.001, -0.002, 0.1, 0.55, -0.55];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 3,
            num_qubits: 2,
            ints_per_pauli: 1,
            atol: 1e-12,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        let trunc_onenorm_max_terms: f64 = cpt_op.truncate();
        assert_eq!(vec![1, 5, 6], cpt_op.paulis);
        assert_eq!(vec![-0.9, 0.55, -0.55], cpt_op.coeffs);
        assert!((trunc_onenorm_max_terms - 0.103).abs() < 1e-10);

        // Test atol
        let paulis: Vec<u64> = vec![1, 2, 3, 4, 5, 6];
        let coeffs: Vec<f64> = vec![-0.9, 0.001, -0.002, 0.1, 0.55, -0.55];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 100,
            num_qubits: 2,
            ints_per_pauli: 1,
            atol: 0.1,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        let trunc_onenorm_atol: f64 = cpt_op.truncate();
        assert_eq!(vec![1, 5, 6], cpt_op.paulis);
        assert_eq!(vec![-0.9, 0.55, -0.55], cpt_op.coeffs);
        assert!((trunc_onenorm_atol - 0.103).abs() < 1e-10);
    }
    #[test]
    fn test_insert_or_combine() {
        let paulis: Vec<u64> = vec![12];
        let coeffs: Vec<f64> = vec![1.0];
        let mut cpt_op = CPTOperatorRust {
            paulis: paulis,
            coeffs: coeffs,
            max_terms: 100,
            num_qubits: 2,
            ints_per_pauli: 1,
            atol: 1e-12,
            paulis_buffer: vec![],
            coeffs_buffer: vec![],
        };
        cpt_op.insert_or_combine(&mut vec![(vec![4u64], 2.0)]);
        assert_eq!(cpt_op.paulis, vec![4, 12]);
        assert_eq!(cpt_op.coeffs, vec![2.0, 1.0]);
        cpt_op.insert_or_combine(&mut vec![(vec![4u64], 2.0)]);
        assert_eq!(cpt_op.paulis, vec![4, 12]);
        assert_eq!(cpt_op.coeffs, vec![4.0, 1.0]);
        let trunc_onenorm = cpt_op.insert_or_combine(&mut vec![(vec![12u64], -1.0 + 1e-13)]);
        assert_eq!(cpt_op.paulis, vec![4]);
        assert_eq!(cpt_op.coeffs, vec![4.0]);
        assert!((trunc_onenorm - 1e-13).abs() < f64::EPSILON);
    }
    #[test]
    fn test_cmp_paulis() {
        assert_eq!(Ordering::Equal, cmp_paulis(&vec![1, 10], &vec![1, 10]));
        assert_eq!(Ordering::Less, cmp_paulis(&vec![1, 10], &vec![10, 0]));
        assert_eq!(Ordering::Greater, cmp_paulis(&vec![10, 10], &vec![10, 9]));
    }
    #[test]
    fn test_multiply_paulis() {
        let xyz: Vec<u64> = vec![30];
        let xzy: Vec<u64> = vec![29];
        let yxz: Vec<u64> = vec![46];
        let yzx: Vec<u64> = vec![53];
        let zxy: Vec<u64> = vec![43];
        let zyx: Vec<u64> = vec![51];
        let xxz: Vec<u64> = vec![14];
        let xyi: Vec<u64> = vec![22];
        assert_eq!(
            (vec![0], 0),
            multiply_paulis(&xyz, &xyz, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![3], 0),
            multiply_paulis(&xyz, &xzy, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![48], 0),
            multiply_paulis(&xyz, &yxz, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![43], 1),
            multiply_paulis(&xyz, &yzx, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![53], 3),
            multiply_paulis(&xyz, &zxy, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![45], 0),
            multiply_paulis(&xyz, &zyx, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![45], 0),
            multiply_paulis(&xyz, &zyx, &vec![0, 1, 2], 3, false)
        );
        assert_eq!(
            (vec![16], 3),
            multiply_paulis(&xxz, &xyz, &vec![0, 1, 2], 3, true)
        );
        assert_eq!(
            (vec![16], 1),
            multiply_paulis(&xxz, &xyz, &vec![0, 1, 2], 3, false)
        );
        assert_eq!(
            (vec![24], 3),
            multiply_paulis(&xxz, &xyi, &vec![1, 2], 3, true)
        );
        assert_eq!(
            (vec![24], 1),
            multiply_paulis(&xxz, &xyi, &vec![1, 2], 3, false)
        );
        // I * 30 + ZZ + I
        let izzi: Vec<u64> = vec![1 << 63, 1];
        let ixxi: Vec<u64> = vec![3 << 30, 0];
        assert_eq!(
            (vec![(1 << 63) + (3 << 30), 1], 2),
            multiply_paulis(&izzi, &ixxi, &vec![30, 31], 33, true)
        );
    }
    #[test]
    fn test_get_anticommuting() {
        let mut paulis: Vec<u64> = Vec::with_capacity(16);
        paulis.push(0); // II
        paulis.push(2); // XI
        paulis.push(10); // YI
        paulis.push(8); // ZI
        paulis.push(1); // IX
        paulis.push(3); // XX
        paulis.push(11); // YX
        paulis.push(9); // ZX
        paulis.push(5); // IY
        paulis.push(7); // XY
        paulis.push(15); // YY
        paulis.push(13); // ZY
        paulis.push(4); // IZ
        paulis.push(6); // XZ
        paulis.push(14); // YZ
        paulis.push(12); // ZZ
        assert_eq!(
            Vec::<usize>::new(),
            get_anticommuting(&paulis, &vec![0], &vec![0, 1], 2, 1)
        );
        // XY
        let expected: Vec<usize> = vec![2, 3, 4, 5, 10, 11, 12, 13];
        let actual: Vec<usize> = get_anticommuting(&paulis, &vec![7], &vec![0, 1], 2, 1);
        assert_eq!(expected, actual);
        // Z + I * 32 + X + I * 32
        let mut bigpauli = vec![0; 3];
        bigpauli[0] = 1 << 32;
        bigpauli[2] = 1 << 3;
        let expected: Vec<usize> = vec![0];
        let actual: Vec<usize> = get_anticommuting(
            &bigpauli,
            &vec![0, 2, 0], // X + I * 65
            &vec![65],
            66,
            3,
        );
        assert_eq!(expected, actual);
        let expected: Vec<usize> = vec![0];
        let other = &vec![0, 1 << 34, 0]; // I * 33 + Z + I * 32
        let actual: Vec<usize> = get_anticommuting(&bigpauli, &other, &vec![32], 66, 3);
        assert_eq!(expected, actual);
    }
}

#[pymodule]
fn _accelerate(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evolve_by_circuit, m)?)?;
    m.add_function(wrap_pyfunction!(k_largest_products, m)?)?;
    Ok(())
}
