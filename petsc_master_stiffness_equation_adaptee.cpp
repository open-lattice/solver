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

  MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,
                            size,
                            PETSC_DECIDE,
                            PETSC_DETERMINE,
                            size - MasterStiffnessEquation::GetConstraintCount(),
                            row_index.data(),
                            column_index.data(),
                            values.data(),
                            &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  MatTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
               MAT_INPLACE_MATRIX,
               &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  MatSetOption(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);
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
  MatView(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, PETSC_VIEWER_STDOUT_WORLD);
  VecView(PetscMasterStiffnessEquationAdaptee::forces_, PETSC_VIEWER_STDOUT_WORLD);
  VecView(PetscMasterStiffnessEquationAdaptee::modified_forces_, PETSC_VIEWER_STDOUT_WORLD);
  /* application of the formula:  _f = T^T.f */
  MatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
          PetscMasterStiffnessEquationAdaptee::forces_,
          PetscMasterStiffnessEquationAdaptee::modified_forces_);

  /* application of the formula(divided into three steps):  _K = T^T.K.T => _K = (T^T.K).T */
  /* get T^T.K */
  MatMatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             PetscMasterStiffnessEquationAdaptee::stiffness_matrix_,
             MAT_INITIAL_MATRIX,
             PETSC_DEFAULT,
             &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));
  MatView(PetscMasterStiffnessEquationAdaptee::stiffness_matrix_, PETSC_VIEWER_STDOUT_WORLD);
  //MatView(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_, PETSC_VIEWER_STDOUT_WORLD);
  /* get T */
  MatTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
               MAT_INPLACE_MATRIX,
               &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
  /* (T^T.K).T */
  MatMatMult(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
             PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             MAT_REUSE_MATRIX,
             PETSC_DEFAULT,
             &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));

  //VecScale(PetscMasterStiffnessEquationAdaptee::gaps_, -1.0F);
  //Vec temporary_vector;
  //PetscMasterStiffnessEquationAdaptee::InitializeVector(&temporary_vector);
  //MatMultAdd(PetscMasterStiffnessEquationAdaptee::stiffness_matrix_,
  //           PetscMasterStiffnessEquationAdaptee::gaps_,
  //           PetscMasterStiffnessEquationAdaptee::forces_,
  //           temporary_vector);
  //VecScale(PetscMasterStiffnessEquationAdaptee::gaps_, -1.0F);
  //PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::modified_forces_));
  //MatMultTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
  //                 temporary_vector,
  //                 PetscMasterStiffnessEquationAdaptee::modified_forces_);
  //VecDestroy(&temporary_vector);
  //MatProductSetFromOptions(PetscMasterStiffnessEquationAdaptee::transformation_matrix_);
  //MatTransposeMatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
  //                    PetscMasterStiffnessEquationAdaptee::stiffness_matrix_,
  //                    MAT_INITIAL_MATRIX,
  //                    PETSC_DEFAULT,
  //                    &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));
  //MatMatMult(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
  //           PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
  //           MAT_INITIAL_MATRIX,
  //           PETSC_DECIDE,
  //           &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));

}

void PetscMasterStiffnessEquationAdaptee::Solve() {
  KSP ksp;
  PC pc;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp,
                  PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
                  PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_);
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCJACOBI);
  KSPSetTolerances(ksp, 1.e-5, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetFromOptions(ksp);
  InitializeVector(&(PetscMasterStiffnessEquationAdaptee::modified_displacements_));
  KSPSolve(ksp,
           PetscMasterStiffnessEquationAdaptee::modified_forces_,
           PetscMasterStiffnessEquationAdaptee::modified_displacements_);
  KSPDestroy(&ksp);

  PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::displacements_));
  MatMultAdd(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
             PetscMasterStiffnessEquationAdaptee::modified_displacements_,
             PetscMasterStiffnessEquationAdaptee::gaps_,
             PetscMasterStiffnessEquationAdaptee::displacements_);
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
void PetscMasterStiffnessEquationAdaptee::InitializeVector(Vec *vec) {
  VecCreate(PETSC_COMM_WORLD, vec);
  VecSetSizes(*vec, MasterStiffnessEquation::ReadActiveRowSize(), PETSC_DECIDE);
  VecSetFromOptions(*vec);
  VecSet(*vec, 0.0F);
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
