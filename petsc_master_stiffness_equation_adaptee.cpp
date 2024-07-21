//
// Created by Nitel Muhtaroglu on 2023-12-23.
//

#include "petsc_master_stiffness_equation_adaptee.h"

PetscMasterStiffnessEquationAdaptee::PetscMasterStiffnessEquationAdaptee() = default;

void PetscMasterStiffnessEquationAdaptee::ApplyConstraints() {
  static unsigned long size{MasterStiffnessEquation::ReadActiveRowSize()};
  InitializeGlobalToMasterIndicesLookupTable(size);
  boost::container::vector<PetscScalar> values(size - MasterStiffnessEquation::GetConstraintCount(), 1.0F);
  boost::container::vector<PetscInt> column_index;
  for (int i{0}; i < size - MasterStiffnessEquation::GetConstraintCount(); ++i) {
    column_index.push_back(i);
  }
  boost::container::vector<PetscInt> row_index{0};
  for (unsigned long i{0}; i < size; ++i) {
    if (MasterStiffnessEquation::IsSlaveIndexForAConstraint(i)) {
      row_index.push_back(row_index.at(i));
    } else {
      row_index.push_back(row_index.at(i) + 1);
    }
  }

  int world_size_{0};
  MPI_Comm_size(PETSC_COMM_WORLD, &world_size_);
  PetscMPIInt rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  printf("Rank: %d \n", rank);

  PetscMPIInt chunk_size{static_cast<PetscMPIInt>(size / world_size_)};
  PetscMPIInt remainder{static_cast<PetscMPIInt>(size - world_size_ * chunk_size)};
  printf("Chunk size: %d\n", chunk_size);
  printf("remainder size: %d\n", remainder);
  for (int i{0}; i < remainder; ++i) {
    if (i == rank) {
      ++chunk_size;
    }
  }

  for (int i{rank * chunk_size}; i < (rank + 1) * chunk_size; ++i) {
    printf("%d ", i);
  }

  printf("\n");
  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,
                            size / world_size_,
                            PETSC_DECIDE,
                            PETSC_DETERMINE,
                            size - MasterStiffnessEquation::GetConstraintCount(),
                            row_index.data(),
                            column_index.data(),
                            values.data(),
                            &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  PetscInt m;
  PetscInt n;
  MatGetSize(transformation_matrix_, &m, &n);
  printf("Transformation Matrix sizes: %d %d\n", m, n);
  MatView(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, PETSC_VIEWER_STDOUT_WORLD);
  MatTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
               MAT_INPLACE_MATRIX,
               &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  MatGetSize(transformation_matrix_, &m, &n);
  MatSetOption(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);
  printf("Transposed Transformation Matrix sizes: %d %d\n", m, n);
  boost::container::vector<PetscInt> rows;
  for (const auto &constraint : MasterStiffnessEquation::GetConstraints()) {
    //VecSetValue(PetscMasterStiffnessEquationAdaptee::gaps_,
    //            constraint.GetSlaveTermIndex(),
    //            constraint.GetGap() / constraint.GetSlaveTermCoefficient(),
    //            INSERT_VALUES);
    rows.push_back(constraint.GetSlaveTermIndex());

    for (const auto &master_term : constraint.GetMasterTerms()) {
      MatSetValue(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
                  PetscMasterStiffnessEquationAdaptee::global_to_master_indices_lookup_.left.find(master_term.GetIndex())->second,
                  constraint.GetSlaveTermIndex(),
                  -1.0F * master_term.GetCoefficient() / constraint.GetSlaveTermCoefficient(),
                  INSERT_VALUES);
    }
  }

  MatAssemblyBegin(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_FINAL_ASSEMBLY);
  /* We now have T^T in stored as transformation_matrix_ */


  PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::modified_forces_),
                                                        size - MasterStiffnessEquation::GetConstraintCount());

  /* application of the formula:  _f = T^T.f */
  MatView(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, PETSC_VIEWER_STDOUT_WORLD);
  MatCreateVecs(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, &forces_, &modified_forces_);
  MatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
          PetscMasterStiffnessEquationAdaptee::forces_,
          PetscMasterStiffnessEquationAdaptee::modified_forces_);

  std::cout << "f: " << std::endl;
  VecView(PetscMasterStiffnessEquationAdaptee::forces_, PETSC_VIEWER_STDOUT_WORLD);
  std::cout << "_f: " << std::endl;
  VecView(PetscMasterStiffnessEquationAdaptee::modified_forces_, PETSC_VIEWER_STDOUT_WORLD);

  return;
  /* application of the formula(divided into three steps):  _K = T^T.K.T => _K = (T^T.K).T */
  /* get T^T.K */
  /* create _K matrix */
  /* Get a sparse matrix K by dumping zero entries of Kdense */
  MatCreate(PETSC_COMM_WORLD, &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));
  MatSetSizes(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
              size,
              size - MasterStiffnessEquation::GetConstraintCount(),
              PETSC_DECIDE,
              PETSC_DECIDE);
  MatSetType(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_, MATMPIAIJ);

  /* The allocation above is approximate so we must set this option to be permissive.
   * Real code should preallocate exactly. */
  MatSetOption(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
               MAT_NEW_NONZERO_LOCATION_ERR,
               PETSC_FALSE);

  MatAssemblyBegin(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_, MAT_FINAL_ASSEMBLY);

  MatMatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             PetscMasterStiffnessEquationAdaptee::stiffness_matrix_,
             MAT_INITIAL_MATRIX,
             PETSC_DEFAULT,
             &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));
  /* get T */
  MatTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
               MAT_INPLACE_MATRIX,
               &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  /* (T^T.K).T */
  MatMatMult(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
             PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             MAT_INITIAL_MATRIX,
             PETSC_DEFAULT,
             &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));

  //std::cout << "_K: " << std::endl;
  //MatView(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_, PETSC_VIEWER_STDOUT_WORLD);
}

void PetscMasterStiffnessEquationAdaptee::Solve() {
  KSP krylov_method;
  PC preconditioner;
  KSPCreate(PETSC_COMM_WORLD, &krylov_method);
  KSPSetOperators(krylov_method,
                  PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
                  PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_);
  KSPGetPC(krylov_method, &preconditioner);
  PCSetType(preconditioner, PCJACOBI);
  KSPSetTolerances(krylov_method, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetFromOptions(krylov_method);
  PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::modified_displacements_),
                                                        MasterStiffnessEquation::ReadActiveRowSize()
                                                            - MasterStiffnessEquation::GetConstraintCount());
  KSPSolve(krylov_method,
           PetscMasterStiffnessEquationAdaptee::modified_forces_,
           PetscMasterStiffnessEquationAdaptee::modified_displacements_);
  KSPDestroy(&krylov_method);

  //std::cout << "_u: " << std::endl;
  //VecView(PetscMasterStiffnessEquationAdaptee::modified_displacements_, PETSC_VIEWER_STDOUT_WORLD);
  PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::displacements_),
                                                        MasterStiffnessEquation::ReadActiveRowSize());
  PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::gaps_),
                                                        MasterStiffnessEquation::ReadActiveRowSize());
  MatMultAdd(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             PetscMasterStiffnessEquationAdaptee::modified_displacements_,
             PetscMasterStiffnessEquationAdaptee::gaps_,
             PetscMasterStiffnessEquationAdaptee::displacements_);
  std::cout << "u: " << std::endl;
  VecView(PetscMasterStiffnessEquationAdaptee::displacements_, PETSC_VIEWER_STDOUT_WORLD);
}

void PetscMasterStiffnessEquationAdaptee::SetStiffnessMatrix(const Mat &stiffness_matrix) {
  PetscMasterStiffnessEquationAdaptee::stiffness_matrix_ = stiffness_matrix;
  MatSetOption(PetscMasterStiffnessEquationAdaptee::stiffness_matrix_, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);
  static PetscInt number_of_rows;
  static PetscInt number_of_columns;
  MatGetSize(PetscMasterStiffnessEquationAdaptee::stiffness_matrix_, &number_of_rows, &number_of_columns);
  MasterStiffnessEquation::InitializeReductionVectors(number_of_rows);
}

[[nodiscard]] const Mat &PetscMasterStiffnessEquationAdaptee::GetStiffnessMatrix() const {
  return PetscMasterStiffnessEquationAdaptee::stiffness_matrix_;
}

void PetscMasterStiffnessEquationAdaptee::SetForces(const Vec &forces) {
  PetscMasterStiffnessEquationAdaptee::forces_ = forces;
}

[[nodiscard]] const Vec &PetscMasterStiffnessEquationAdaptee::GetForces() const {
  return PetscMasterStiffnessEquationAdaptee::forces_;
}

void PetscMasterStiffnessEquationAdaptee::SetGaps(const Vec &gaps) {
  PetscMasterStiffnessEquationAdaptee::gaps_ = gaps;
}

[[nodiscard]] const Vec &PetscMasterStiffnessEquationAdaptee::GetGaps() const {
  return PetscMasterStiffnessEquationAdaptee::gaps_;
}

[[nodiscard]] const Mat &PetscMasterStiffnessEquationAdaptee::GetTransformationMatrix() const {
  return PetscMasterStiffnessEquationAdaptee::transformation_matrix_;
}

[[nodiscard]] const Vec &PetscMasterStiffnessEquationAdaptee::GetModifiedForces() const {
  return PetscMasterStiffnessEquationAdaptee::modified_forces_;
}

[[nodiscard]] const Mat &PetscMasterStiffnessEquationAdaptee::GetModifiedStiffnessMatrix() const {
  return PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_;
}

[[nodiscard]] const Vec &PetscMasterStiffnessEquationAdaptee::GetModifiedDisplacements() const {
  return PetscMasterStiffnessEquationAdaptee::modified_displacements_;
}

[[nodiscard]] const Vec &PetscMasterStiffnessEquationAdaptee::GetDisplacements() const {
  return PetscMasterStiffnessEquationAdaptee::displacements_;
}
void PetscMasterStiffnessEquationAdaptee::InitializeVector(Vec *vector, PetscInt size) {
  VecCreateMPI(PETSC_COMM_WORLD,
               PETSC_DECIDE,
               size,
               vector);
  VecSetFromOptions(*vector);
  VecSet(*vector, 0.0F);
  VecAssemblyBegin(*vector);
  VecAssemblyEnd(*vector);
}

unsigned long PetscMasterStiffnessEquationAdaptee::InitializeGlobalToMasterIndicesLookupTable(unsigned long problem_size) {
  PetscMasterStiffnessEquationAdaptee::global_to_master_indices_lookup_.clear();
  std::unordered_set<unsigned long> slave_indices_for_constraints;

  for (auto i{0}; i < MasterStiffnessEquation::GetConstraintCount(); ++i) {
    slave_indices_for_constraints.insert(MasterStiffnessEquation::GetConstraint(i).GetSlaveTermIndex());
  }

  unsigned long master_index_for_constraint{0};
  for (int i{0}; i < problem_size; ++i) {
    if (slave_indices_for_constraints.find(i) == slave_indices_for_constraints.end()) {
      PetscMasterStiffnessEquationAdaptee::global_to_master_indices_lookup_.insert(boost::bimap<unsigned long,
                                                                                                unsigned long>::value_type(
          i,
          master_index_for_constraint++));
    }
  }
  return PetscMasterStiffnessEquationAdaptee::global_to_master_indices_lookup_.size();
}
