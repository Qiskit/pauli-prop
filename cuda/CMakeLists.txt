# CMakeLists.txt for CUDA-Accelerated Pauli Propagation
#
# This CMake configuration builds the CUDA implementation of Pauli propagation
# with Python bindings. It handles:
# - CUDA compilation with appropriate flags
# - pybind11 integration for Python bindings
# - Thrust library (included with CUDA)
# - Optional cuQuantum integration
# - Proper linking and installation
#
# Build Instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86"
#   make -j
#   make install
#
# Requirements:
#   - CUDA Toolkit 11.0+ (for Thrust and modern CUDA features)
#   - CMake 3.18+ (for CUDA language support)
#   - Python 3.7+
#   - pybind11
#   - Optional: cuQuantum SDK for advanced features

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Project definition
project(pauli_prop_cuda 
        VERSION 0.1.0
        LANGUAGES CXX CUDA
        DESCRIPTION "CUDA-Accelerated Pauli Propagation for Quantum Circuits")

# ============================================================================
# CUDA CONFIGURATION
# ============================================================================

# Enable CUDA language
enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architectures (modify based on target GPUs)
# Common architectures:
#   70 = Volta (V100)
#   75 = Turing (RTX 2080, T4)
#   80 = Ampere (A100)
#   86 = Ampere (RTX 3090)
#   89 = Ada Lovelace (RTX 4090)
#   90 = Hopper (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
    message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# CUDA compiler flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

# Debug vs Release flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# ============================================================================
# DEPENDENCIES
# ============================================================================

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# Thrust is included with CUDA Toolkit (no separate find needed)
message(STATUS "Thrust will be used from CUDA Toolkit")

# Optional: Find cuQuantum (comment out if not available)
# find_package(cuQuantum)
# if(cuQuantum_FOUND)
#     message(STATUS "cuQuantum found: ${cuQuantum_VERSION}")
#     add_definitions(-DUSE_CUQUANTUM)
# else()
#     message(STATUS "cuQuantum not found - building without cuQuantum support")
# endif()

# ============================================================================
# SOURCE FILES
# ============================================================================

set(CUDA_SOURCES
    pauli_prop_cuda.cu
)

set(CPP_SOURCES
    python_bindings.cpp
)

set(HEADERS
    pauli_prop_cuda.h
)

# ============================================================================
# BUILD TARGETS
# ============================================================================

# Create CUDA library (static for linking with Python module)
add_library(pauli_prop_cuda_lib STATIC ${CUDA_SOURCES})

# Set properties for CUDA library
set_target_properties(pauli_prop_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Link CUDA libraries
target_link_libraries(pauli_prop_cuda_lib
    PUBLIC
        CUDA::cudart
        CUDA::cuda_driver
)

# Include directories for CUDA library
target_include_directories(pauli_prop_cuda_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Optional: Link cuQuantum if available
# if(cuQuantum_FOUND)
#     target_link_libraries(pauli_prop_cuda_lib PUBLIC cuQuantum::custatevec)
# endif()

# Create Python module using pybind11
pybind11_add_module(_pauli_prop_cuda ${CPP_SOURCES})

# Link Python module with CUDA library
target_link_libraries(_pauli_prop_cuda
    PRIVATE
        pauli_prop_cuda_lib
        pybind11::module
)

# Set properties for Python module
set_target_properties(_pauli_prop_cuda PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_pauli_prop_cuda"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

# Include directories for Python module
target_include_directories(_pauli_prop_cuda
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${Python_INCLUDE_DIRS}
        ${pybind11_INCLUDE_DIRS}
)

# ============================================================================
# INSTALLATION
# ============================================================================

# Determine installation directory
if(NOT DEFINED PYTHON_INSTALL_DIR)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(PYTHON_INSTALL_DIR "${PYTHON_SITE_PACKAGES}/pauli_prop_cuda")
endif()

message(STATUS "Python module will be installed to: ${PYTHON_INSTALL_DIR}")

# Install Python module
install(TARGETS _pauli_prop_cuda
        LIBRARY DESTINATION ${PYTHON_INSTALL_DIR}
        COMPONENT python)

# Install Python wrapper (if exists)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")
    install(FILES __init__.py
            DESTINATION ${PYTHON_INSTALL_DIR}
            COMPONENT python)
endif()

# Install headers (for C++ users)
install(FILES ${HEADERS}
        DESTINATION include/pauli_prop_cuda
        COMPONENT headers)

# ============================================================================
# TESTING (Optional)
# ============================================================================

option(BUILD_TESTS "Build test suite" OFF)

if(BUILD_TESTS)
    enable_testing()
    
    # Add test executable
    add_executable(test_cuda_kernels test_cuda_kernels.cu)
    target_link_libraries(test_cuda_kernels pauli_prop_cuda_lib)
    
    add_test(NAME CUDAKernelTests COMMAND test_cuda_kernels)
endif()

# ============================================================================
# DOCUMENTATION
# ============================================================================

# Print configuration summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "Install directory: ${PYTHON_INSTALL_DIR}")
message(STATUS "============================")
message(STATUS "")

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

# Print build instructions
message(STATUS "Build instructions:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  make install")
message(STATUS "")
message(STATUS "To specify CUDA architectures:")
message(STATUS "  cmake .. -DCMAKE_CUDA_ARCHITECTURES=\"70;75;80;86\"")
message(STATUS "")
message(STATUS "To build tests:")
message(STATUS "  cmake .. -DBUILD_TESTS=ON")
message(STATUS "")