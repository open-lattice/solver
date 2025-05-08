//
// Created by Nitel Muhtaroglu on 2023-12-23.
//

#include "petsc_master_stiffness_equation_adaptee.h"

PetscMasterStiffnessEquationAdaptee::PetscMasterStiffnessEquationAdaptee() = default;

void PetscMasterStiffnessEquationAdaptee::ApplyConstraints() {
    static unsigned long global_size{MasterStiffnessEquation::ReadActiveRowSize()};
    PetscMPIInt rank, size_mpi;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size_mpi);

    MatCreate(PETSC_COMM_WORLD, &transformation_matrix_);
    MatSetSizes(transformation_matrix_, PETSC_DECIDE, PETSC_DECIDE, global_size, global_size);
    MatSetFromOptions(transformation_matrix_);
    MatSetUp(transformation_matrix_);

    MatSetOption(transformation_matrix_, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);

    PetscInt rstart, rend;
    MatGetOwnershipRange(transformation_matrix_, &rstart, &rend);

    for (PetscInt i = rstart; i < rend; ++i) {
        if (!MasterStiffnessEquation::IsSlaveIndexForAConstraint(i)) {
            MatSetValue(transformation_matrix_, i, i, 1.0, INSERT_VALUES);
        }
    }

    const auto &constraints = MasterStiffnessEquation::GetConstraints();
    std::size_t total_constraints = constraints.size();

    std::size_t constraints_per_proc = total_constraints / size_mpi;
    std::size_t remainder = total_constraints % size_mpi;

    std::size_t start_idx, end_idx;
    if (rank < remainder) {
        start_idx = rank * (constraints_per_proc + 1);
        end_idx = start_idx + (constraints_per_proc + 1);
    } else {
        start_idx = rank * constraints_per_proc + remainder;
        end_idx = start_idx + constraints_per_proc;
    }

    for (std::size_t idx = start_idx; idx < end_idx; ++idx) {
        const auto &constraint = constraints[idx];
        int slave_idx = constraint.GetSlaveTermIndex();

        for (const auto &master_term : constraint.GetMasterTerms()) {
            MatSetValue(transformation_matrix_,
                        master_term.GetIndex(),  // (adjust indexing if needed)
                        slave_idx,
                        -master_term.GetCoefficient() / constraint.GetSlaveTermCoefficient(),
                        INSERT_VALUES);
        }
        VecSetValue(gaps_, slave_idx,
                    constraint.GetGap() / constraint.GetSlaveTermCoefficient(),
                    INSERT_VALUES);
    }

    MatAssemblyBegin(transformation_matrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transformation_matrix_, MAT_FINAL_ASSEMBLY);

    MatTranspose(transformation_matrix_, MAT_INPLACE_MATRIX, &transformation_matrix_);

    //MatView(transformation_matrix_, PETSC_VIEWER_STDOUT_WORLD);
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

