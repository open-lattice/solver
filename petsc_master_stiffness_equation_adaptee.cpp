//
// Created by Nitel Muhtaroglu on 2023-12-23.
//

#include "petsc_master_stiffness_equation_adaptee.h"

PetscMasterStiffnessEquationAdaptee::PetscMasterStiffnessEquationAdaptee() = default;

void PetscMasterStiffnessEquationAdaptee::ApplyConstraints() {
    static unsigned long size{MasterStiffnessEquation::ReadActiveRowSize()};
    InitializeGlobalToMasterIndicesLookupTable(size);

    int world_size;
    PetscMPIInt rank;

    PetscInt rstart = 0, rend = 0;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &world_size);

    std::vector<PetscInt> counts(world_size), displs(world_size + 1, 0);
    for (int i = 0; i < world_size; ++i) {
        counts[i] = size / world_size + (i < size % world_size ? 1 : 0);
        displs[i + 1] = displs[i] + counts[i];
    }
    rstart = displs[rank];
    rend = displs[rank + 1];

    // 2. Build local CSR arrays (ia, ja, a)
    std::vector ia_local{0};
    std::vector<PetscInt> ja_local;
    std::vector<PetscScalar> a_local;

    for (PetscInt i = rstart; i < rend; ++i) {
        if (MasterStiffnessEquation::IsSlaveIndexForAConstraint(i)) {
            // Slave DOF → empty row
            ia_local.push_back(ia_local.back());
        } else {
            // Retained DOF → identity row
            PetscInt master_col = global_to_master_indices_lookup_.left.at(i);
            ja_local.push_back(master_col);
            a_local.push_back(1.0);
            ia_local.push_back(ia_local.back() + 1);
        }
    }

    // printf("\n");
    // printf("Content of %d: \n", rank);
    // for (int i{rstart}; i < rend; ++i) {
    //     printf("%d ", i);
    // }
    // printf("\n");

    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD,
                              rend - rstart,
                              PETSC_DECIDE,
                              size,
                              size - MasterStiffnessEquation::GetConstraintCount(),
                              ia_local.data(),
                              ja_local.data(),
                              a_local.data(),
                              &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_));
    PetscInt m;
    PetscInt n;
    MatGetSize(transformation_matrix_, &m, &n);

    if (rank == 0)
        printf("Transformation Matrix sizes: %d %d\n\n", m, n);

    MatSetOption(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE);

    const auto &constraints = MasterStiffnessEquation::GetConstraints();
    std::size_t total_constraints = constraints.size();

    std::size_t constraints_per_proc = total_constraints / world_size;
    std::size_t remainder = total_constraints % world_size;

    std::size_t start_idx, end_idx;
    if (rank < remainder) {
        start_idx = rank * (constraints_per_proc + 1);
        end_idx = start_idx + (constraints_per_proc + 1);
    } else {
        start_idx = rank * constraints_per_proc + remainder;
        end_idx = start_idx + constraints_per_proc;
    }

    for (std::size_t idx{start_idx}; idx < end_idx; ++idx) {
        const auto &constraint = constraints[idx];
        VecSetValue(PetscMasterStiffnessEquationAdaptee::gaps_,
                    constraint.GetSlaveTermIndex(),
                    constraint.GetGap() / constraint.GetSlaveTermCoefficient(),
                    INSERT_VALUES);

        for (const auto &master_term: constraint.GetMasterTerms()) {
            MatSetValue(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
                        constraint.GetSlaveTermIndex(),
                        PetscMasterStiffnessEquationAdaptee::global_to_master_indices_lookup_.left.find(
                            master_term.GetIndex())->second,
                        -1.0F * master_term.GetCoefficient() / constraint.GetSlaveTermCoefficient(),
                        INSERT_VALUES);
        }
    }

    VecAssemblyBegin(PetscMasterStiffnessEquationAdaptee::gaps_);
    VecAssemblyEnd(PetscMasterStiffnessEquationAdaptee::gaps_);

    MatAssemblyBegin(transformation_matrix_, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(transformation_matrix_, MAT_FINAL_ASSEMBLY);

    PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::modified_forces_),
                                                          size - MasterStiffnessEquation::GetConstraintCount());

    Vec temp;
    PetscMasterStiffnessEquationAdaptee::InitializeVector(&temp, size);

    MatMult(PetscMasterStiffnessEquationAdaptee::stiffness_matrix_,
            PetscMasterStiffnessEquationAdaptee::gaps_,
            temp);

    VecScale(temp, -1.0F);
    VecAXPY(temp, 1.0F, PetscMasterStiffnessEquationAdaptee::forces_);

    MatTranspose(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, MAT_INPLACE_MATRIX,
                &(PetscMasterStiffnessEquationAdaptee::transformation_matrix_)); //T.T
    MatMult(PetscMasterStiffnessEquationAdaptee::transformation_matrix_, temp,
            PetscMasterStiffnessEquationAdaptee::modified_forces_);

    MatCreate(PETSC_COMM_WORLD, &(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_));
    MatSetSizes(PetscMasterStiffnessEquationAdaptee::modified_stiffness_matrix_,
                size - MasterStiffnessEquation::GetConstraintCount(),
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

    // if (rank == 0) std::cout << "T: " << std::endl;
    // MatView(transformation_matrix_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
    //
    // if (rank == 0) std::cout << "K: " << std::endl;
    // MatView(stiffness_matrix_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
    //
    // if (rank == 0) std::cout << "_K: " << std::endl;
    // MatView(modified_stiffness_matrix_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
    //
    // if (rank == 0) std::cout << "g: " << std::endl;
    // VecView(gaps_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
    //
    // if (rank == 0) std::cout << "f: " << std::endl;
    // VecView(PetscMasterStiffnessEquationAdaptee::forces_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
    //
    // if (rank == 0) std::cout << "_f: " << std::endl;
    // VecView(PetscMasterStiffnessEquationAdaptee::modified_forces_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank == 0) std::cout << "\n" << std::endl;
}


void PetscMasterStiffnessEquationAdaptee::Solve() {
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

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
    PetscMasterStiffnessEquationAdaptee::InitializeVector(
        &(PetscMasterStiffnessEquationAdaptee::modified_displacements_),
        MasterStiffnessEquation::ReadActiveRowSize()
        - MasterStiffnessEquation::GetConstraintCount());
    KSPSolve(krylov_method,
             PetscMasterStiffnessEquationAdaptee::modified_forces_,
             PetscMasterStiffnessEquationAdaptee::modified_displacements_);
    KSPDestroy(&krylov_method);

    // if (rank==0) std::cout << "_u: " << std::endl;
    // VecView(PetscMasterStiffnessEquationAdaptee::modified_displacements_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank==0) std::cout << "\n" << std::endl;

    PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::displacements_),
                                                          MasterStiffnessEquation::ReadActiveRowSize());
    PetscMasterStiffnessEquationAdaptee::InitializeVector(&(PetscMasterStiffnessEquationAdaptee::gaps_),
                                                          MasterStiffnessEquation::ReadActiveRowSize());
    MatMultAdd(PetscMasterStiffnessEquationAdaptee::transformation_matrix_,
               PetscMasterStiffnessEquationAdaptee::modified_displacements_,
               PetscMasterStiffnessEquationAdaptee::gaps_,
               PetscMasterStiffnessEquationAdaptee::displacements_);

    // if (rank==0) std::cout << "u: " << std::endl;
    // VecView(PetscMasterStiffnessEquationAdaptee::displacements_, PETSC_VIEWER_STDOUT_WORLD);
    // if (rank==0) std::cout << "\n" << std::endl;
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

unsigned long PetscMasterStiffnessEquationAdaptee::InitializeGlobalToMasterIndicesLookupTable(
    unsigned long problem_size) {
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
