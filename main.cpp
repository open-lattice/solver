//
// Created by Nitel Muhtaroglu on 2023-11-06.
//
#include <boost/assign.hpp>
#include <boost/container/vector.hpp>
#include <boost/array.hpp>
#include <boost/assert.hpp>

#include <gtest/gtest.h>

#include "constraint.h"
#include "petsc_master_stiffness_equation_adaptee.h"
#include "term.hpp"

#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscksp.h>
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
  Mat Kdense;
  Mat K;
  PetscViewer fd;                        /* viewer */
  char file[PETSC_MAX_PATH_LEN];  /* input file name */
  PetscErrorCode ierr;
  PetscInt m, n, rstart, rend;
  PetscBool flg;
  PetscInt row, ncols, j, nrows, nnzA = 0, nnzAsp = 0;
  const PetscInt *cols;
  const PetscScalar *vals;
  PetscReal norm, percent, val, dtol = 1.e-16;
  PetscMPIInt rank;
  MatInfo matinfo;
  PetscInt Dnnz, Onnz;

  ierr = PetscInitialize(&argc, &args, (char *) 0, help);
  if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
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
  /*printf("matinfo.nz_used %g\n",matinfo.nz_used);*/

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
  //ierr = MatSeqAIJSetPreallocation(K, Dnnz, NULL);
  //ierr = MatSeqSBAIJSetPreallocation(K, 1, Dnnz, NULL);
  CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(K, 1, Dnnz, NULL, Onnz, NULL);
  //ierr = MatMPIAIJSetPreallocation(K, Dnnz, NULL, Onnz, NULL);
  CHKERRQ(ierr);
  /* The allocation above is approximate so we must set this option to be permissive.
   * Real code should preallocate exactly. */
  ierr = MatSetOption(K, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);
  CHKERRQ(ierr);

  /* Check zero rows */
  ierr = MatGetOwnershipRange(Kdense, &rstart, &rend);
  CHKERRQ(ierr);
  nrows = 0;
  for (row = rstart; row < rend; row++) {
    ierr = MatGetRow(Kdense, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
    nnzA += ncols;
    norm = 0.0;
    for (j = 0; j < ncols; ++j) {
      val = PetscAbsScalar(vals[j]);
      if (norm < val) norm = norm;
      if (val > dtol) {
        ierr = MatSetValues(K, 1, &row, 1, &cols[j], &vals[j], INSERT_VALUES);
        if (row != cols[j]) {
          ierr = MatSetValues(K, 1, &cols[j], 1, &row, &vals[j], INSERT_VALUES);
        }
        CHKERRQ(ierr);
        nnzAsp++;
      }
    }
    if (!norm) { ++nrows; };
    ierr = MatRestoreRow(Kdense, row, &ncols, &cols, &vals);
    CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);

  percent = (PetscReal) nnzA * 100 / (m * n);
  ierr = PetscPrintf(PETSC_COMM_SELF,
                     " [%d] Matrix Kdense local size %d,%d; nnzA %d, %g percent; No. of zero rows: %d\n",
                     rank,
                     m,
                     n,
                     nnzA,
                     percent,
                     nrows);
  percent = (PetscReal) nnzAsp * 100 / (m * n);
  ierr = PetscPrintf(PETSC_COMM_SELF, " [%d] Matrix K nnzAsp %d, %g percent\n", rank, nnzAsp, percent);

  /* investigate matcoloring for K */
  PetscBool Asp_coloring = PETSC_FALSE;
  ierr = PetscOptionsHasName(NULL, NULL, "-Asp_color", &Asp_coloring);
  CHKERRQ(ierr);
  if (Asp_coloring) {
    MatColoring mc;
    ISColoring iscoloring;
    MatFDColoring matfdcoloring;
    ierr = PetscPrintf(PETSC_COMM_WORLD, " Create coloring of K...\n");
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
    /*ierr = MatFDColoringView(matfdcoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);*/
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
  master_terms.push_back(Term(5, -1.0F));
  //for (int i{1}; i < nrows; ++i) {
  //  master_terms.push_back(Term(i, 1.0F));
  //}

  boost::container::vector constraints{Constraint(Term(0, 1.0F), master_terms)};
  master_stiffness_equation_.SetConstraints(
      constraints);

  master_stiffness_equation_.ApplyConstraints();
  master_stiffness_equation_.Solve();



















  //Mat expected_transformation_matrix;
  //boost::array<PetscInt, 7> _beginning_of_each_row{0, 1, 2, 7, 8, 9, 10};
  //boost::array<PetscInt, 10>
  //    _column_numbers{0, 1, 0, 1, 3, 4, 5, 3, 4, 5}; // j vec size nnz
  //boost::array<PetscScalar, 10>
  //    _non_zero_values{1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  //MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, 6, 6, PETSC_DETERMINE,
  //                          PETSC_DETERMINE, _beginning_of_each_row.data(),
  //                          _column_numbers.data(), _non_zero_values.data(), &expected_transformation_matrix);
  //auto actual_transformation_matrix = master_stiffness_equation_.GetTransformationMatrix();
  //PetscBool equal;
  //MatEqual(expected_transformation_matrix, actual_transformation_matrix, &equal);
  //BOOST_ASSERT(equal);

  //Mat expected_modified_stiffness_matrix;
  //boost::array<PetscInt, 7> _beginning_of_each_row_{0, 4, 7, 7, 10, 14, 19};
  //boost::array<PetscInt, 19>
  //    _column_numbers_{0, 3, 4, 5, 1, 4, 5, 0, 3, 5, 0, 1, 4, 5, 0, 1, 3, 4, 5}; // j vec size nnz
  //boost::array<PetscScalar, 19>
  //    _non_zero_values_
  //    {300.0F, 100.0F, 200.0F, 200.0F, 200.0F, 100.0F, 100.0F, 100.0F, 200.0F, 100.0F, 200.0F, 100.0F, 400.0F,
  //     100.0F, 200.0F, 100.0F, 100.0F, 100.0F, 400.F};
  //MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, 6, 6, PETSC_DETERMINE,
  //                          PETSC_DETERMINE, _beginning_of_each_row_.data(),
  //                          _column_numbers_.data(), _non_zero_values_.data(), &expected_modified_stiffness_matrix);
  //auto actual_modified_stiffness_matrix = master_stiffness_equation_.GetModifiedStiffnessMatrix();
  //MatEqual(actual_modified_stiffness_matrix, expected_modified_stiffness_matrix, &equal);
  //MatView(expected_modified_stiffness_matrix, PETSC_VIEWER_STDOUT_WORLD);
  //MatView(actual_modified_stiffness_matrix, PETSC_VIEWER_STDOUT_WORLD);
  //BOOST_ASSERT(equal);

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
