# Solver
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>

<h1>Parallel Multi-freedom Constraint Application with PETSc</h1>

<p>This project demonstrates a parallel implementation of constraint application using MPI under PETSc library for inter-process communication.  
  At this stage, application supports matrices given in Matrix Market <code>(.mtx)</code> or binary file format.</p>

<h2>Requirements</h2>
<ul>
  <li><strong>CMake</strong>: Used for easily building the software.</li>
    <li><strong>MPI Library</strong>: An implementation of the Message Passing Interface (MPI), such as OpenMPI or MPICH.</li>
    <li><strong>PETSc 3.23.2 (latest)</strong>: A high-performance computing toolkit for robust sparse matrix representation, as
well as a higher-level wrapper for MPI operations.</li>
</ul>
<h2>Important Files</h2>
<ul>
    <li><code>main.cpp</code>: The main program file containing the parallel constraint application implementation.</li>
    <li><code>term.hpp</code>: Header file for term object used to represent constraint equation terms.</li>
    <li><code>constraint.h</code>: Header file for the constraint equation objects.</li>
    <li><code>constraint.cpp</code>: Implementation of the constraint equation objects.</li>
  <li><code>master_stiffness_equation.hpp</code>: Header file for objects holding the linear equation system and virtual functions.</li>
  <li><code>petsc_master_stiffness_equation_adaptee</code>: Implementation of the system solution functions with attributes inherited from master_stiffness_equation.</li>
  <li><code>run_all.sh</code>: Shell script for automated benchmarking.</li>
  <li><code>plot_timing.py</code>: Python code for post-processing/visualizing the benchmark results.</li>
</ul>

<h2>Compilation</h2>
<p>To compile the test programs, using CMake is suggested. Development environments such as CLion are helpful:</p>

<h3>Using CMake in the project directory:</h3>
<pre><code>cmake --build <dir> [<options>] [-- <build-tool-options>...]</code></pre>

<h2>Running the Program</h2>
<p>To run the parallel program, use the <code>mpirun</code> command with the desired number of processes:</p>
<pre><code>mpirun -np &lt;number_of_processes&gt; ./test_parallel &lt;matrix_folder&gt; &lt;number_of_constraints&gt;</code></pre>
<p>Replace <code>&lt;number_of_processes&gt;</code> with the number of MPI processes you want to launch.</p>

<h3>Example</h3>
<pre><code>mpirun -np 4 ./cmake-build-debug/main bcsstk21.mtx 100</code></pre>
<p>This example runs the parallel program with 4 MPI processes using bcsstk21.mtx matrix file and applies 100 randomized constraints.</p>


