/*
 * CUDA-Accelerated Pauli Propagation Library
 * 
 * This header defines GPU-accelerated quantum circuit simulation using CUDA.
 * It provides massive parallelization for Pauli operator evolution through
 * quantum circuits, leveraging NVIDIA GPUs for 10-100x speedup over CPU.
 *
 * Key CUDA Libraries Used:
 * - cuQuantum: NVIDIA's quantum computing library for state vector operations
 * - Thrust: CUDA C++ template library for parallel algorithms
 * - cuBLAS: GPU-accelerated linear algebra
 * - cuSPARSE: Sparse matrix operations (for sparse Pauli representations)
 *
 * Architecture:
 * - Device (GPU) memory management for Pauli operators
 * - Parallel kernels for anticommutation checking
 * - Parallel Pauli multiplication using warp-level primitives
 * - Thrust-based sorting and reduction operations
 * - Pinned host memory for fast CPU-GPU transfers
 *
 * Reference: Original Rust implementation in lib.rs
 */

#ifndef PAULI_PROP_CUDA_H
#define PAULI_PROP_CUDA_H

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <vector>
#include <cstdint>
#include <memory>

// ============================================================================
// CUDA ERROR CHECKING MACROS
// ============================================================================

/// Macro for checking CUDA errors with file/line information
/// Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/// Macro for checking kernel launch errors
#define CUDA_CHECK_LAST_ERROR() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CONSTANTS AND TYPE DEFINITIONS
// ============================================================================

/// Phase modulo for Pauli multiplication (phases are powers of i: 0,1,2,3)
constexpr uint8_t PHASE_MOD = 4;

/// Warp size for CUDA (32 threads per warp on all current NVIDIA GPUs)
constexpr int WARP_SIZE = 32;

/// Block size for most kernels (tuned for occupancy)
constexpr int BLOCK_SIZE = 256;

/// Maximum number of qubits per u64 word (64 bits / 2 bits per qubit)
constexpr int QUBITS_PER_WORD = 32;

/// Type alias for Pauli bit-packed representation
/// Each u64 stores X and Z bits for up to 32 qubits
using PauliWord = uint64_t;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * @brief GPU-resident Pauli operator representation
 * 
 * This structure holds Pauli operators in GPU memory using bit-packed format.
 * Each Pauli term is represented by multiple u64 words (ints_per_pauli).
 * 
 * Memory Layout:
 * - paulis: Flat array [term0_word0, term0_word1, ..., term1_word0, ...]
 * - coeffs: Real coefficients for each term
 * 
 * Invariants:
 * - paulis.size() == num_terms * ints_per_pauli
 * - coeffs.size() == num_terms
 * - All terms are sorted lexicographically for efficient merging
 */
struct DevicePauliOperator {
    thrust::device_vector<PauliWord> paulis;  ///< Bit-packed Pauli terms
    thrust::device_vector<double> coeffs;      ///< Real coefficients
    int num_terms;                             ///< Number of Pauli terms
    int num_qubits;                            ///< Number of qubits
    int ints_per_pauli;                        ///< u64 words per Pauli term
    int max_terms;                             ///< Maximum allowed terms
    double atol;                               ///< Absolute tolerance for truncation
    
    /**
     * @brief Constructor allocates GPU memory
     * @param n_terms Initial number of terms
     * @param n_qubits Number of qubits in the system
     * @param max_t Maximum terms allowed
     * @param tolerance Absolute tolerance for coefficient truncation
     */
    DevicePauliOperator(int n_terms, int n_qubits, int max_t, double tolerance);
    
    /**
     * @brief Copy data from host to device
     * @param host_paulis Host array of bit-packed Paulis
     * @param host_coeffs Host array of coefficients
     */
    void copyFromHost(const std::vector<PauliWord>& host_paulis,
                      const std::vector<double>& host_coeffs);
    
    /**
     * @brief Copy data from device to host
     * @param host_paulis Output host array for Paulis
     * @param host_coeffs Output host array for coefficients
     */
    void copyToHost(std::vector<PauliWord>& host_paulis,
                    std::vector<double>& host_coeffs) const;
};

/**
 * @brief Configuration for k-largest products search
 * 
 * This structure holds parameters for the GPU-accelerated k-largest
 * products algorithm using parallel priority queue.
 */
struct KLargestConfig {
    int k;                      ///< Number of largest products to find
    bool assume_hermitian;      ///< Exploit Hermitian symmetry
    int heap_capacity;          ///< Initial heap capacity (typically 3*k)
    
    KLargestConfig(int k_val, bool hermitian = false)
        : k(k_val), assume_hermitian(hermitian), heap_capacity(3 * k_val) {}
};

// ============================================================================
// CUDA KERNEL DECLARATIONS
// ============================================================================

/**
 * @brief GPU kernel to find anticommuting Pauli terms
 * 
 * This kernel checks which terms in the operator anticommute with a gate.
 * Each thread processes one Pauli term, checking anticommutation on qargs.
 * 
 * Parallelization Strategy:
 * - One thread per Pauli term
 * - Warp-level reduction for anticommutation flag
 * - Coalesced memory access for bit-packed Paulis
 * 
 * @param paulis Device array of bit-packed Pauli terms
 * @param gate Device array for gate Pauli operator
 * @param qargs Device array of qubit indices
 * @param num_qargs Number of qubits the gate acts on
 * @param num_terms Total number of Pauli terms
 * @param num_qubits Total number of qubits
 * @param ints_per_pauli Number of u64 words per Pauli
 * @param anticomm_flags Output: 1 if term anticommutes, 0 otherwise
 */
__global__ void findAnticommutingKernel(
    const PauliWord* paulis,
    const PauliWord* gate,
    const int* qargs,
    int num_qargs,
    int num_terms,
    int num_qubits,
    int ints_per_pauli,
    int* anticomm_flags
);

/**
 * @brief GPU kernel for Pauli multiplication
 * 
 * Multiplies Pauli terms by a gate Pauli, computing both result and phase.
 * Uses lookup table for single-qubit Pauli products.
 * 
 * Parallelization Strategy:
 * - One thread per Pauli term
 * - Warp shuffle for phase accumulation
 * - Shared memory for lookup table
 * 
 * @param paulis_in Input Pauli terms
 * @param gate Gate Pauli operator
 * @param qargs Qubit indices
 * @param num_qargs Number of qubits
 * @param num_terms Number of terms
 * @param num_qubits Total qubits
 * @param ints_per_pauli Words per Pauli
 * @param paulis_out Output Pauli terms
 * @param phases_out Output phases (mod 4)
 */
__global__ void multiplyPaulisKernel(
    const PauliWord* paulis_in,
    const PauliWord* gate,
    const int* qargs,
    int num_qargs,
    int num_terms,
    int num_qubits,
    int ints_per_pauli,
    PauliWord* paulis_out,
    uint8_t* phases_out
);

/**
 * @brief GPU kernel for coefficient scaling (cos/sin factors)
 * 
 * Applies trigonometric scaling to coefficients based on anticommutation.
 * 
 * @param coeffs_inout Coefficients to scale (in-place)
 * @param anticomm_flags Anticommutation flags
 * @param cos_theta Cosine of rotation angle
 * @param sin_theta Sine of rotation angle
 * @param num_terms Number of terms
 */
__global__ void scaleCoefficientsKernel(
    double* coeffs_inout,
    const int* anticomm_flags,
    double cos_theta,
    double sin_theta,
    int num_terms
);

/**
 * @brief GPU kernel for bit-packing boolean arrays to u64
 * 
 * Converts NumPy-style boolean arrays to bit-packed representation.
 * Each thread processes one u64 word, packing 64 bits.
 * 
 * @param bool_array Input boolean array
 * @param packed_output Output bit-packed array
 * @param num_terms Number of Pauli terms
 * @param num_qubits Number of qubits
 * @param ints_per_pauli Words per Pauli
 */
__global__ void packBoolArrayKernel(
    const bool* bool_array,
    PauliWord* packed_output,
    int num_terms,
    int num_qubits,
    int ints_per_pauli
);

/**
 * @brief GPU kernel for unpacking u64 to boolean arrays
 * 
 * Reverse of packBoolArrayKernel for returning results to Python.
 * 
 * @param packed_input Input bit-packed array
 * @param bool_output Output boolean array
 * @param num_terms Number of Pauli terms
 * @param num_qubits Number of qubits
 * @param ints_per_pauli Words per Pauli
 */
__global__ void unpackToBoolArrayKernel(
    const PauliWord* packed_input,
    bool* bool_output,
    int num_terms,
    int num_qubits,
    int ints_per_pauli
);

// ============================================================================
// HOST FUNCTIONS (C++ API)
// ============================================================================

/**
 * @brief Initialize CUDA device and check capabilities
 * 
 * Selects best GPU, checks compute capability, and prints device info.
 * Should be called once at program startup.
 * 
 * @return Device ID of selected GPU
 */
int initializeCUDA();

/**
 * @brief GPU-accelerated k-largest products search
 * 
 * Finds k largest |other[l] * to_evolve[m] * other[n]| products using
 * parallel priority queue on GPU.
 * 
 * Algorithm:
 * 1. Initialize heap on GPU with (0,0,0)
 * 2. Parallel heap operations using atomic operations
 * 3. Thrust-based sorting for final output
 * 
 * @param to_evolve_real Real parts of to_evolve coefficients
 * @param to_evolve_imag Imaginary parts
 * @param other_real Real parts of other coefficients
 * @param other_imag Imaginary parts
 * @param config Configuration (k, hermitian flag)
 * @param output_triplets Output array of (l,m,n) triplets
 * @return Number of triplets found
 */
int kLargestProductsCUDA(
    const std::vector<double>& to_evolve_real,
    const std::vector<double>& to_evolve_imag,
    const std::vector<double>& other_real,
    const std::vector<double>& other_imag,
    const KLargestConfig& config,
    std::vector<std::array<int64_t, 3>>& output_triplets
);

/**
 * @brief GPU-accelerated circuit evolution
 * 
 * Evolves a Pauli operator through a quantum circuit using CPT algorithm.
 * All operations performed on GPU for maximum performance.
 * 
 * Workflow:
 * 1. Transfer operator and gates to GPU
 * 2. For each gate:
 *    a. Find anticommuting terms (parallel)
 *    b. Scale coefficients (cos/sin)
 *    c. Generate new terms (parallel multiplication)
 *    d. Merge and deduplicate (Thrust sort/reduce)
 *    e. Truncate small terms
 * 3. Transfer result back to host
 * 
 * @param operator_paulis Input Pauli terms (bit-packed)
 * @param operator_coeffs Input coefficients
 * @param gate_paulis Gate Pauli operators
 * @param qargs Qubit indices for each gate
 * @param thetas Rotation angles
 * @param max_terms Maximum terms allowed
 * @param atol Absolute tolerance
 * @param frame 's' for Schrödinger, 'h' for Heisenberg
 * @param output_paulis Output Pauli terms
 * @param output_coeffs Output coefficients
 * @return Total truncation error (L1-norm)
 */
double evolveByCircuitCUDA(
    const std::vector<PauliWord>& operator_paulis,
    const std::vector<double>& operator_coeffs,
    const std::vector<PauliWord>& gate_paulis,
    const std::vector<std::vector<int>>& qargs,
    const std::vector<double>& thetas,
    int num_qubits,
    int max_terms,
    double atol,
    char frame,
    std::vector<PauliWord>& output_paulis,
    std::vector<double>& output_coeffs
);

/**
 * @brief Cleanup CUDA resources
 * 
 * Frees all GPU memory and resets device.
 * Should be called at program exit.
 */
void cleanupCUDA();

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Compare two Pauli operators lexicographically (device function)
 * 
 * Used for sorting Pauli terms on GPU.
 * 
 * @param a First Pauli (pointer to ints_per_pauli words)
 * @param b Second Pauli
 * @param ints_per_pauli Number of words to compare
 * @return -1 if a<b, 0 if a==b, 1 if a>b
 */
__device__ int comparePaulis(
    const PauliWord* a,
    const PauliWord* b,
    int ints_per_pauli
);

/**
 * @brief Extract X and Z bits for a qubit (device function)
 * 
 * Helper function for Pauli operations.
 * 
 * @param pauli Pointer to Pauli term
 * @param qubit Qubit index
 * @param num_qubits Total number of qubits
 * @param ints_per_pauli Words per Pauli
 * @param x_out Output X bit
 * @param z_out Output Z bit
 */
__device__ void extractPauliBits(
    const PauliWord* pauli,
    int qubit,
    int num_qubits,
    int ints_per_pauli,
    bool& x_out,
    bool& z_out
);

/**
 * @brief Set X and Z bits for a qubit (device function)
 * 
 * @param pauli Pointer to Pauli term (modified in-place)
 * @param qubit Qubit index
 * @param num_qubits Total number of qubits
 * @param x_bit X bit value
 * @param z_bit Z bit value
 */
__device__ void setPauliBits(
    PauliWord* pauli,
    int qubit,
    int num_qubits,
    bool x_bit,
    bool z_bit
);

#endif // PAULI_PROP_CUDA_H

// Made with Bob
