##########
Pauli Prop
##########

About
-----

Pauli propagation, also known as sparse Pauli dynamics (SPD), is a framework for approximating the evolution of operators in the Pauli basis under the action of other operators, such as quantum circuit gates and noise channels [1] - [4]. This approach can be effective when the operators involved are expected to remain sparse in the Pauli basis. The technique has been used to classically estimate expectation values of quantum systems and also to reduce the depths of quantum circuits to be run on a quantum processor [5]. To learn how to use this package to simulate expectation values of quantum systems, check out the tutorial.

This package provides a Rust-accelerated Python interface for performing the most common Pauli propagation routines.

Primary functions
^^^^^^^^^^^^^^^^^

- ``propagate_through_rotation_gates``: Evolve an operator specified in the Pauli basis, :math:`O`, through a sequence of Pauli rotation gates, :math:`P`, creating a transformed operator, :math:`\tilde{O}`. This evolution can be done in either the Heisenberg (:math:`\tilde{O}=P^{\dagger}OP`) or the Schrödinger (:math:`\tilde{O}=POP^{\dagger}`) framework.
- ``propagate_through_operator``: Evolve an operator specified in the Pauli basis, :math:`O`, through another such operator, :math:`G`, creating a transformed operator, :math:`\tilde{O}`. This evolution can be done in either the Heisenberg (:math:`\tilde{O}=G^{\dagger}OG`) or Schrödinger (:math:`GOG^{\dagger}`) framework.
- ``evolve_through_cliffords``: Separate a quantum circuit, :math:`U`, into Clifford and non-Clifford parts, :math:`C` and :math:`P` respectively, such that :math:`U = PC`.

Technical details
^^^^^^^^^^^^^^^^^

- Rust-accelerated Python interface
- Ability to truncate terms from :math:`\tilde{O}` during evolution based on an absolute coefficient tolerance, a fixed number of terms in the evolving operator, or a combination of both.
- Ability to perform Pauli propagation in both the Schrödinger and Heisenberg frameworks.
- Novel technique for approximating the conjugation of a Pauli-sum operator, :math:`O`, by another such operator, :math:`G` (:math:`GOG^\dagger` or :math:`G^{\dagger}OG`). This heuristic implementation greedily generates contributions to the product expected to be most significant.

Computational requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^

Both the memory and time cost for Pauli propagation routines generally scale with the size to which the evolved operator is allowed to grow.

``propagate_through_rotation_gates``: As the Pauli operator, :math:`\tilde{O}`, is propagated in the Pauli basis under the action of a sequence of :math:`N` Pauli rotation gates of an :math:`M`-qubit circuit, the number of terms will grow as :math:`\mathcal{O}(2^N)` towards a maximum of :math:`4^M` unique Pauli components. To control memory usage, the operator is truncated after application of each gate, which introduces some error proportional to the magnitudes of the truncated terms' coefficients. The memory requirements are generally linear in the size of the evolved operator and runtime scales linearly in both the operator size and the number of gates.

``propagate_through_operator``: Conjugates one operator in the Pauli basis by another, (:math:`G^{\dagger}OG`), by greedily accumulating terms in the sum, :math:`\sum_{i,j,k}G^{\dagger}_iO_jG_k`, where :math:`i,j,k` are sparse indices over the Pauli basis. This implementation sorts the coefficients in each operator by descending magnitude then searches the 3D index space for the terms with the largest coefficients, starting with the origin :math:`(0,0,0)`, and accumulating :math:`(i,j,k)` triplets up to a specified cutoff. The time spent searching can often be made negligible by increasing the search step size in :math:`(i,j,k)` space, which provides a cubic speedup for this subroutine. In our profiling, significant time can be spent sorting the operators and performing Pauli multiplication to generate the terms in the new operator.

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/pauli-prop/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'pauli-prop'
This package is still in early development and has not yet had an initial release. Users who want to
try this out early may install from source. Installing from source requires the user have the Rust
toolchain installed. Visit the `Rust docs<https://rust-lang.org/tools/install/>`_ for more info.

.. code-block:: bash

   cd path_to_repository/
   pip install -e .


Citing this project
-------------------

If you use this package in your research, please cite it according to ``CITATON.bib`` file included in this repository:

.. literalinclude:: ../CITATION.bib
   :language: bibtex

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
release notes.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/pauli-prop>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/pauli-prop/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/pauli-prop/issues/new/choose>`_ for tracking requests and bugs.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/pauli-prop/blob/main/LICENSE.txt>`_

.. _references:

References
----------

[1] Tomislav Begušić, Johnnie Gray, Garnet Kin-Lic Chan, [Fast and converged classical simulations of evidence for the utility of quantum computing before fault tolerance](https://arxiv.org/abs/2308.05077), arXiv:2308.05077 [quant-ph].

[2] Manuel S. Rudolph, et al., [Pauli Propagation: A Computational Framework for Simulating Quantum Systems](https://arxiv.org/abs/2505.21606), arXiv:2505.21606 [quant-ph].

[3] Hrant Gharibyan, et al., [A Practical Guide to using Pauli Path Simulators for Utility-Scale Quantum Experiments](https://arxiv.org/abs/2507.10771), arXiv:2507.10771 [quant-ph].

[4] Lukas Broers, et al., [Scalable Simulation of Quantum Many-Body Dynamics with Or-Represented Quantum Algebra](https://arxiv.org/abs/2506.13241), arXiv:2506.13241 [quant-ph].

[5] Bryce Fuller, et al., [Improved Quantum Computation using Operator Backpropagation](https://arxiv.org/abs/2502.01897), arXiv:2502.01897 [quant-ph].

.. toctree::
  :hidden:
   
   Documentation Home <self>
   Installation Instructions <install>
   Tutorials <tutorials/index>
   How-To Guides <how_tos/index>
   API Reference <apidocs/index>
   GitHub <https://github.com/qiskit/pauli-prop>