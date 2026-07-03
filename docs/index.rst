#################
Pauli propagation
#################

The ``pauli-prop`` package provides a Rust-accelerated Python interface for performing Pauli propagation.

Pauli propagation is a framework for approximating the evolution of operators in the Pauli basis under the action of other operators, such as quantum circuit gates and noise channels. This approach can be effective when the operators involved are expected to remain sparse in the Pauli basis.

The subroutines in this package may be used to implement error mitigation techniques such as `lightcone shading <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-slc>`_ `[6] <ref6_>`_ and `propagated noise absorption <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-pna>`_ `[7] <ref7_>`_, `operator backpropagation <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-obp>`_ `[8] <ref8_>`_ for circuit depth reduction, and classical simulation of expectation values `[1-5] <references_>`_.

Getting started
---------------

A simple guide to help you get started quickly with this package is available in the :doc:`quickstart guide <guides/quickstart>`.

Use case examples
-----------------

Pauli propagation can be used as a lower-level engine to implement a variety of techniques. Some examples of where this has been used are:

- Lightcone shading to reduce the sampling overhead of probabilistic error cancellation (PEC) for mitigating expectation values in a 1- and 2D transverse field Ising model `[6] <ref6_>`_ [``qiskit-addon-slc`` `docs <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-slc>`__]
- Absorbing noise model information into a target observable to mitigate expectation values in a 2D transverse field Ising model `[7] <ref7_>`_ [``qiskit-addon-pna`` `docs <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-pna>`__]
- Trimming trailing gates to produce lower-depth Trotter circuits for the time-evolution of a 2D spin model `[8] <ref8_>`_ [``qiskit-addon-obp`` `docs <https://quantum.cloud.ibm.com/docs/addons/qiskit-addon-obp>`__]

Technical discussion
--------------------

Software details
""""""""""""""""
- Rust-accelerated Python interface
- Support for noisy simulations [`guide <https://quantum.cloud.ibm.com/docs/addons/pauli-prop/guides/simulate_noisy_expectation_values>`_]
- Ability to truncate terms during evolution based on an absolute coefficient tolerance, a fixed number of terms in the evolving operator, or a combination of both
- Ability to perform Pauli propagation in both the Schrödinger and Heisenberg frameworks
- Novel technique for approximating the conjugation of a Pauli-sum operator by another Pauli operator. This heuristic implementation greedily generates contributions to the product expected to be most significant. See Appendix B of `[7] <ref7_>`_ for more information.
- Single-threaded

Computational requirements
""""""""""""""""""""""""""

Both the memory and time cost for Pauli propagation routines generally scale with the size to which the evolved operator is allowed to grow.

:func:`~pauli_prop.propagation.propagate_through_rotation_gates`: As the Pauli operator is propagated in the Pauli basis under the action of a sequence of :math:`N` Pauli rotation gates of an :math:`M`-qubit circuit, the number of terms will grow as :math:`\mathcal{O}(2^N)` towards a maximum of :math:`4^M` unique Pauli components. To control memory usage, the operator is truncated after application of each gate, which introduces some error proportional to the magnitudes of the truncated terms' coefficients. The memory requirements are linear in the size of the evolved operator, and runtime scales linearly in both the operator size and the number of gates.

:func:`~pauli_prop.propagation.propagate_through_operator`: Conjugates one operator in the Pauli basis by another by greedily accumulating terms in the sum, :math:`\sum_{i,j,k}G^{\dagger}_iO_jG_k`, where :math:`i,j,k` are sparse indices over the Pauli basis. This implementation sorts the coefficients in each operator by descending magnitude then searches the 3D index space for the terms with the largest coefficients, starting with the origin :math:`(0,0,0)`, and accumulating :math:`(i,j,k)` triplets up to a specified cutoff. The time spent searching can often be made negligible by increasing the search step size in :math:`(i,j,k)` space, which provides a cubic speedup for this subroutine. In our profiling, significant time can be spent sorting the operators and performing Pauli multiplication to generate the terms in the new operator.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/pauli-prop>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/pauli-prop/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/pauli-prop/issues/new/choose>`_ for tracking requests and bugs.

Citing this package
-------------------

If you use this package in your research, use the `CITATION.bib <https://github.com/Qiskit/pauli-prop/blob/main/CITATION.bib>`_ file in this project's repository to cite the appropriate reference(s).

License
-------

`Apache License 2.0 <https://github.com/Qiskit/pauli-prop/blob/main/LICENSE.txt>`_

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_. We may occasionally make breaking changes in order to
improve the user experience. When possible, we will keep old interfaces and mark them as deprecated, as long
as they can co-exist with the new ones. Each substantial improvement, breaking change, or deprecation will be
documented in the `release notes <https://quantum.cloud.ibm.com/docs/api/qiskit-addon-sqd/release-notes>`_.

.. _references:

References
----------

.. _ref1:

1. [Tomislav Begušić, Johnnie Gray, Garnet Kin-Lic Chan, `Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance <https://arxiv.org/abs/2308.05077>`_, arXiv:2308.05077 [quant-ph].

.. _ref2:

2. Nicolas Loizeau, et al., `Quantum many-body simulations with PauliStrings.jl <https://arxiv.org/abs/2410.09654>`_, arXiv:2410.09654 [quant-ph].

.. _ref3:

3. Manuel S. Rudolph, et al., `Pauli Propagation: A Computational Framework for Simulating Quantum Systems <https://arxiv.org/abs/2505.21606>`_, arXiv:2505.21606 [quant-ph].

.. _ref4:

4. Hrant Gharibyan, et al., `A Practical Guide to using Pauli Path Simulators for Utility-Scale Quantum Experiments <https://arxiv.org/abs/2507.10771>`_, arXiv:2507.10771 [quant-ph].

.. _ref5:

5. Lukas Broers, et al., `Scalable Simulation of Quantum Many-Body Dynamics with Or-Represented Quantum Algebra <https://arxiv.org/abs/2506.13241>`_, arXiv:2506.13241 [quant-ph].

.. _ref6:

6. Andrew Eddins, Minh C. Tran, Patrick Rall, `Lightcone shading for classically accelerated quantum error mitigation <https://arxiv.org/abs/2409.04401>`_, arXiv:2409.04401 [quant-ph].

.. _ref7:

7. Andrew Eddins, et al., `Computing noise-canceling observables via Pauli propagation <https://arxiv.org/abs/2606.20441>`_, arXiv:2606.20441 [quant-ph].

.. _ref8:

8 Bryce Fuller, et al., `Improved Quantum Computation using Operator Backpropagation <https://arxiv.org/abs/2502.01897>`_, arXiv:2502.01897 [quant-ph].

.. toctree::
   :hidden:

   Documentation home <self>
   Installation instructions <install>
   Guides <guides/index>
   GitHub <https://github.com/Qiskit/pauli-prop>

.. toctree::
   :hidden:
   :caption: API reference

   Python API reference <https://quantum.cloud.ibm.com/docs/api/pauli-prop>
   Release notes <release-notes>
