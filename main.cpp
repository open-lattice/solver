//
// Created by Nitel Muhtaroglu on 2023-11-06.
//
#include <boost/assign.hpp>
#include <boost/container/vector.hpp>
#include <boost/array.hpp>

#include <gtest/gtest.h>

#include "constraint.h"
#include "petsc_master_stiffness_equation_adaptee.h"
#include "term.hpp"

#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscerror.h>

static char help[] = "Writes an array to a file, then reads an array from a "
                     "file, then forms a vector.\n\n";

/*
    This uses the low level PetscBinaryWrite() and PetscBinaryRead() to access a
   binary file. It will not work in parallel!

    We HIGHLY recommend using instead VecView() and VecLoad() to read and write
   Vectors in binary format (which also work in parallel). Then you can use
    share/petsc/matlab/PetscBinaryRead() and
   share/petsc/matlab/PetscBinaryWrite() to read (or write) the vector into
   MATLAB.

    Note this also works for matrices with MatView() and MatLoad().
*/

bool TestNonHomogeniousMfcs();

int main(int argc, char **args) {
  // Initialize the MPI environment
  MPI_Init(nullptr, nullptr);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
         processor_name, world_rank, world_size);

  // Finalize the MPI environment.
  Mat Kdense;
  Mat K;
  PetscViewer fd;                        /* viewer */
  char file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode ierr;
  PetscInt m;
  PetscInt n;
  PetscInt rstart;
  PetscInt rend;
  PetscBool flg;
  PetscInt ncols;
  PetscInt nrows;
  PetscInt nnzA = 0;
  PetscInt nnzAsp = 0;
  const PetscInt *cols;
  const PetscScalar *vals;
  PetscReal norm, percent, val, dtol = 1.e-16;
  PetscMPIInt rank;
  MatInfo matinfo;
  PetscInt Dnnz, Onnz;
  ierr = PetscInitialize(&argc, &args, (char *) 0, help);
  if (ierr) { return ierr; }
  int world_size_{0};
  MPI_Comm_size(PETSC_COMM_WORLD, &world_size_);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  printf("Hello world from: rank %d out of %d processes.\n", rank, world_size);
  CHKERRQ(ierr);

  /* Determine files from which we read the linear systems. */
  ierr = PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg);
  CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate binary file with the -f option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd);
  CHKERRQ(ierr);

  /* Load the matrix; then destroy the viewer. */
  ierr = MatCreate(PETSC_COMM_WORLD, &Kdense);
  CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(Kdense, "a_");
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(Kdense);
  CHKERRQ(ierr);
  ierr = MatLoad(Kdense, fd);
  CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);
  CHKERRQ(ierr);
  ierr = MatGetSize(Kdense, &m, &n);
  CHKERRQ(ierr);
  ierr = MatGetInfo(Kdense, MAT_LOCAL, &matinfo);
  CHKERRQ(ierr);

  /* Get a sparse matrix K by dumping zero entries of Kdense */
  ierr = MatCreate(PETSC_COMM_WORLD, &K);
  CHKERRQ(ierr);
  ierr = MatSetSizes(K, m, n, PETSC_DECIDE, PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(K, "asp_");
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(K);
  CHKERRQ(ierr);
  ierr = MatSetType(K, MATMPIAIJ);
  CHKERRQ(ierr);
  Dnnz = (PetscInt) matinfo.nz_used / m + 1;
  Onnz = Dnnz / 2;
  printf("Dnnz %d %d\n", Dnnz, Onnz);
  CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(K, 1, Dnnz, NULL, Onnz, NULL);
  CHKERRQ(ierr);
  /* The allocation above is approximate, so we must set this option to be permissive.
   * Real code should preallocate exactly. */
  ierr = MatSetOption(K, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
  CHKERRQ(ierr);

  /* Check zero rows */
  ierr = MatGetOwnershipRange(Kdense, &rstart, &rend);
  CHKERRQ(ierr);
  nrows = 0;
  for (PetscInt row{rstart}; row < rend; row++) {
    ierr = MatGetRow(Kdense, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
    nnzA += ncols;
    norm = 0.0;
    for (int j{0}; j < ncols; ++j) {
      val = PetscAbsScalar(vals[j]);
      if (norm < val) { norm = norm; }
      if (val > dtol) {
        ierr = MatSetValues(K, 1, &row, 1, &cols[j], &vals[j], INSERT_VALUES);
        if (row != cols[j]) {
          ierr = MatSetValues(K, 1, &cols[j], 1, &row, &vals[j], INSERT_VALUES);
        }
        CHKERRQ(ierr);
        nnzAsp++;
      }
    }
    if (!norm) { ++nrows; }
    ierr = MatRestoreRow(Kdense, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  percent = (PetscReal) nnzA * 100 / (m * n);
  PetscPrintf(PETSC_COMM_SELF,
              " [%d] Matrix Kdense local size %d,%d; nnzA %d, %g percent; No. of zero rows: %d\n",
              rank,
              m,
              n,
              nnzA,
              percent,
              nrows);
  percent = (PetscReal) nnzAsp * 100 / (m * n);
  PetscPrintf(PETSC_COMM_SELF, " [%d] Matrix K nnzAsp %d, %g percent\n", rank, nnzAsp, percent);

  /* investigate matcoloring for K */
  PetscBool Asp_coloring = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-Asp_color", &Asp_coloring);
  CHKERRQ(ierr);
  if (Asp_coloring) {
    MatColoring mc;
    ISColoring iscoloring;
    MatFDColoring matfdcoloring;
    PetscPrintf(PETSC_COMM_WORLD, " Create coloring of K...\n");
    ierr = MatColoringCreate(K, &mc);
    CHKERRQ(ierr);
    ierr = MatColoringSetType(mc, MATCOLORINGSL);
    CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(mc);
    CHKERRQ(ierr);
    ierr = MatColoringApply(mc, &iscoloring);
    CHKERRQ(ierr);
    ierr = MatColoringDestroy(&mc);
    CHKERRQ(ierr);
    ierr = MatFDColoringCreate(K, iscoloring, &matfdcoloring);
    CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(matfdcoloring);
    CHKERRQ(ierr);
    ierr = MatFDColoringSetUp(K, iscoloring, matfdcoloring);
    CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);
    CHKERRQ(ierr);
    ierr = MatFDColoringDestroy(&matfdcoloring);
    CHKERRQ(ierr);
  }

  /* Write K in binary for study - see ~petsc/src/mat/examples/tests/ex124.c */
  PetscBool Asp_write = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-Asp_write", &Asp_write);
  CHKERRQ(ierr);
  if (Asp_write) {
    PetscViewer viewer;
    ierr = PetscPrintf(PETSC_COMM_SELF, "Write K into file K.dat ...\n");
    CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "K.dat", FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = MatView(K, viewer);
    CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
    CHKERRQ(ierr);
  }
  PetscMasterStiffnessEquationAdaptee master_stiffness_equation_;
  master_stiffness_equation_.SetStiffnessMatrix(K);

  Vec forces;
  VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &forces);
  VecSetFromOptions(forces);
  VecSet(forces, 0.0F);
  VecSetValue(forces, 0, -90.0F, INSERT_VALUES);
  VecSetValue(forces, 2, 80.0F, INSERT_VALUES);
  VecAssemblyBegin(forces);
  VecAssemblyEnd(forces);
  master_stiffness_equation_.SetForces(forces);

  boost::container::vector<Term> master_terms;
  //master_terms.push_back(Term(5, -1.0F));
  for (int i{1}; i < nrows; ++i) {
    master_terms.push_back(Term(i, 1.0F));
  }

  boost::container::vector constraints{Constraint(Term(0, 1.0F), master_terms)};
  master_stiffness_equation_.SetConstraints(
      constraints);

  master_stiffness_equation_.ApplyConstraints();
  MPI_Finalize();
  return 0;
  master_stiffness_equation_.Solve();

  //TestNonHomogeniousMfcs();

  ierr = MatDestroy(&Kdense);
  CHKERRQ(ierr);
  ierr = MatDestroy(&K);
  CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

bool TestNonHomogeniousMfcs() {
  PetscMPIInt rank;
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscMasterStiffnessEquationAdaptee master_stiffness_equation_;
  static constexpr int kGlobalProblemSize{6};
  static constexpr int kNumberOfNonZeroEntries{16};
  Mat K;
  boost::array<PetscInt, kGlobalProblemSize + 1> beginning_of_each_row{0, 2, 5, 8,
                                                                       11, 14, 16};
  boost::array<PetscInt, kNumberOfNonZeroEntries>
      column_numbers{0, 1, 0, 1, 2, 1, 2, 3,
                     2, 3, 4, 3, 4, 5, 4, 5}; // j vec size nnz
  boost::array<PetscScalar, kNumberOfNonZeroEntries>
      non_zero_values{100, -100, -100, 200, -100, -100,
                      200, -100, -100, 200, -100, -100,
                      200, -100, -100, 200}; // v vec size nnz
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, kGlobalProblemSize, kGlobalProblemSize, PETSC_DETERMINE,
                            PETSC_DETERMINE, beginning_of_each_row.data(),
                            column_numbers.data(), non_zero_values.data(), &K);
  master_stiffness_equation_.SetStiffnessMatrix(K);

  Vec f;
  VecCreate(PETSC_COMM_WORLD, &f);
  VecSetSizes(f, kGlobalProblemSize, PETSC_DECIDE);
  VecSetFromOptions(f);
  VecSet(f, 0.0F);
  VecSetValue(f, 0, -20.0F, INSERT_VALUES);

  Vec g;
  VecCreate(PETSC_COMM_WORLD, &g);
  VecSetSizes(g, kGlobalProblemSize, PETSC_DECIDE);
  VecSetFromOptions(g);
  VecSet(g, 0.0F);
  master_stiffness_equation_.SetGaps(g);
  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };

  master_stiffness_equation_.SetConstraints(
      constraints);
  master_stiffness_equation_.ApplyConstraints();
  master_stiffness_equation_.Solve();
  auto a = master_stiffness_equation_.GetTransformationMatrix();
  MatView(a, PETSC_VIEWER_STDOUT_WORLD);
  auto b = master_stiffness_equation_.GetModifiedForces();
  VecView(b, PETSC_VIEWER_STDOUT_WORLD);
  auto c = master_stiffness_equation_.GetModifiedStiffnessMatrix();
  MatView(c, PETSC_VIEWER_STDOUT_WORLD);
  auto d = master_stiffness_equation_.GetModifiedDisplacements();
  VecView(d, PETSC_VIEWER_STDOUT_WORLD);
  auto e = master_stiffness_equation_.GetDisplacements();
  VecView(e, PETSC_VIEWER_STDOUT_WORLD);
  return false;
}
