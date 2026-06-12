Installation instructions
=========================

Prerequisites
^^^^^^^^^^^^^^^^

First, create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

There are two primary ways to install this package -- from PyPI or source. The preferred method is to install from PyPI:

Install from PyPI
^^^^^^^^^^^^^^^^^

.. code:: sh

    pip install pauli-prop


Install from source
^^^^^^^^^^^^^^^^^^^

You can install from source if you want to develop in the repository or run the notebooks locally.

First, clone the ``pauli-prop`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/pauli-prop.git

Next, install the Rust toolchain, upgrade pip, and enter the repository. Refer to the `Rust documentation <https://www.rust-lang.org/tools/install>`__
for instructions on installing the toolchain.

.. code:: sh
    
    ### <INSTALL RUST HERE> ###
    pip install --upgrade pip
    cd pauli-prop

The next step is to install ``pauli-prop`` to the virtual environment. Install the
notebook dependencies if you want to run all the visualizations in the notebooks. If you plan on developing in the repository, you
can install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab'
