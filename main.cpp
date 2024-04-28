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
  PetscInitialize(&argc, &args, (char *) 0, help);
  int world_size_{0};
  MPI_Comm_size(PETSC_COMM_WORLD, &world_size_);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  /* Determine files from which we read the linear systems. */
  PetscOptionsGetString(NULL, NULL, "-f", file, PETSC_MAX_PATH_LEN, &flg);
  if (!flg) SETERRQ(PETSC_COMM_WORLD, 1, "Must indicate binary file with the -f option");

  /* Open binary file.  Note that we use FILE_MODE_READ to indicate
     reading from this file. */
  PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd);

  /* Load the matrix; then destroy the viewer. */
  MatCreate(PETSC_COMM_WORLD, &Kdense);
  MatSetOptionsPrefix(Kdense, "a_");
  MatSetFromOptions(Kdense);
  MatLoad(Kdense, fd);
  PetscViewerDestroy(&fd);
  MatGetSize(Kdense, &m, &n);
  MatGetInfo(Kdense, MAT_LOCAL, &matinfo);

  /* Get a sparse matrix K by dumping zero entries of Kdense */
  MatCreate(PETSC_COMM_WORLD, &K);
  MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, m, n);
  //MatSetOptionsPrefix(K, "asp_");
  MatSetOption(K, MAT_STRUCTURE_ONLY, PETSC_TRUE);
  MatSetType(K, MATMPISBAIJ);
  Dnnz = (PetscInt) matinfo.nz_used / m + 1;
  Onnz = Dnnz / 2;
  printf("Dnnz %d %d\n", Dnnz, Onnz);
  MatMPISBAIJSetPreallocation(K, 1, Dnnz, NULL, Onnz, NULL);
  /* The allocation above is approximate, so we must set this option to be permissive.
   * Real code should preallocate exactly. */
  MatSetOption(K, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE);

  /* Check zero rows */
  MatGetOwnershipRange(Kdense, &rstart, &rend);
  nrows = 0;
  for (PetscInt row{rstart}; row < rend; row++) {
    MatGetRow(Kdense, row, &ncols, &cols, &vals);
    nnzA += ncols;
    norm = 0.0;
    for (int j{0}; j < ncols; ++j) {
      val = PetscAbsScalar(vals[j]);
      if (norm < val) { norm = norm; }
      if (val > dtol) {
        MatSetValues(K, 1, &row, 1, &cols[j], &vals[j], INSERT_VALUES);
        if (row != cols[j]) {
          MatSetValues(K, 1, &cols[j], 1, &row, &vals[j], INSERT_VALUES);
        }
        nnzAsp++;
      }
    }
    if (!norm) { ++nrows; }
    MatRestoreRow(Kdense, row, &ncols, &cols, &vals);
  }
  MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

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

  PetscMasterStiffnessEquationAdaptee master_stiffness_equation_;
  master_stiffness_equation_.SetStiffnessMatrix(K);

  Vec            x,y;
  int total_ranks;
  PetscScalar one = 1.0;
  PetscScalar zero = 0.0;
  MatCreateVecs(K, &x, &y);
  //ierr = VecCreate(PETSC_ICOMM_WORLD,&x);CHKERRQ(ierr);
  //ierr = VecSetType(x,VECMPI);
  //ierr = VecSetSizes(x,m/total_ranks,m);CHKERRQ(ierr); //Force local size instead of PETSC_DECIDE
  //ierr = VecSetFromOptions(x);CHKERRQ(ierr);

 // ierr = VecSetType(x,VECMPI);
 // ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
 // ierr = VecSetSizes(y,m/total_ranks,m);CHKERRQ(ierr); //Force local size instead of PETSC_DECIDE
 // ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,zero); CHKERRQ(ierr);



/* SpMV*/
  ierr = MatMult(K,x,y);CHKERRQ(ierr);
  //MatView(K, PETSC_VIEWER_STDOUT_WORLD);
  //MatView(K, PETSC_VIEWER_DRAW_WORLD);
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);
  VecView(y, PETSC_VIEWER_STDOUT_WORLD);
  //VecView(y, PETSC_VIEWER_DRAW_WORLD);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = MatDestroy(&K);CHKERRQ(ierr);
  Vec forces;
  //VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n, &forces);
  //VecSetType(forces, VECMPI);
  //VecSetFromOptions(forces);
  //VecSet(forces, 0.0F);
  //VecSetValue(forces, 0, -90.0F, INSERT_VALUES);
  //VecSetValue(forces, 2, 80.0F, INSERT_VALUES);
  //VecAssemblyBegin(forces);
  //VecAssemblyEnd(forces);
  //master_stiffness_equation_.SetForces(forces);

  //boost::container::vector<Term> master_terms;
  ////master_terms.push_back(Term(5, -1.0F));
  //for (int i{1}; i < nrows; ++i) {
  //  master_terms.push_back(Term(i, 1.0F));
  //}

  //boost::container::vector constraints{Constraint(Term(0, 1.0F), master_terms)};
  //master_stiffness_equation_.SetConstraints(
  //    constraints);

  //master_stiffness_equation_.ApplyConstraints();
  PetscFinalize();
  return 0;
  master_stiffness_equation_.Solve();

  //TestNonHomogeniousMfcs();

  MatDestroy(&Kdense);
  MatDestroy(&K);
  PetscFinalize();
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
