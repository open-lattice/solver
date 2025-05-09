#include <iostream>
#include <petscvec.h>
#include <petscerror.h>

#include <boost/assign.hpp>
#include <boost/container/vector.hpp>
#include <boost/array.hpp>
#include <boost/assert.hpp>
#include <unordered_set>

#include <sstream>
#include <fstream>
#include <string>

#include <ctime>
#include <random>

#include "constraint.h"
#include "petsc_master_stiffness_equation_adaptee.h"
#include "term.hpp"

std::tuple<int, Mat> read_matrix_from_mtx(const char* mtx_file);
boost::container::vector<Constraint> generate_constraints(PetscInt size, PetscInt nC, std::mt19937 gen,
    PetscInt nm_max = 4, float max_coeff = 5.0);
bool homogenous_manual_constraints_trial(int argc, char **args);

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    PetscErrorCode err;
    PetscInt nrows, ncols;
    bool print_constraints = false;
    bool print_mat = false;
    double sparsity;

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Usage: %s input.mtx number_of_constraints\n", argv[0]);
        }
        PetscFinalize();
        return -1;
    }

    const char* mtx_file = argv[1];
    const PetscInt nC = atoi(argv[2]);

    double total_start = MPI_Wtime();

    auto [nnz, K] = read_matrix_from_mtx(mtx_file);

    MatGetSize(K, &nrows, &ncols);
    const int kGlobalProblemSize = nrows;

    if (print_mat)
        MatView(K, PETSC_VIEWER_STDOUT_SELF);

    // --- Matrix Info Output ---
    if (rank == 0) {
        sparsity = 1.0 - static_cast<double>(nnz * 2 - nrows) / (nrows * ncols);
        PetscPrintf(PETSC_COMM_WORLD,
            "Matrix Info:\n"
            " - Global Rows: %d\n"
            " - Global Cols: %d\n"
            " - Non-zeros: %d\n"
            " - Sparsity: %.6f\n\n",
            nrows, ncols, nnz, sparsity);
    }

    PetscMasterStiffnessEquationAdaptee master_stiffness_equation_;
    master_stiffness_equation_.SetStiffnessMatrix(K);

    Vec f;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kGlobalProblemSize, &f);
    VecSetFromOptions(f);
    VecSet(f, 0.0F);
    VecSetValue(f, 0, -20.0F, INSERT_VALUES);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    Vec g;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, kGlobalProblemSize, &g);
    VecSetFromOptions(g);
    VecSet(g, 0.0F);
    VecAssemblyBegin(g);
    VecAssemblyEnd(g);

    master_stiffness_equation_.SetGaps(g);
    master_stiffness_equation_.SetForces(f);

    unsigned int seed;
    std::random_device rd;  // Non-deterministic generator
    if (rank == 0) {
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPIU_INT,0, MPI_COMM_WORLD);

    std::mt19937 gen(seed); // Mersenne Twister generator seeded with rd()

    const auto constraints = generate_constraints(kGlobalProblemSize, nC, gen);

    if (print_constraints) {
        for (const auto& constraint : constraints) {
            PetscPrintf(PETSC_COMM_SELF, "rank %d -> s: %d, m: [", rank, constraint.GetSlaveTermIndex());
            for (auto m: constraint.GetMasterTerms()){
                PetscPrintf(PETSC_COMM_SELF, "%d ", m.GetIndex());
            }
            PetscPrintf(PETSC_COMM_SELF, "]\n");
        }
    }

    master_stiffness_equation_.SetConstraints(constraints);

    double constraint_start = MPI_Wtime(); // Start constraint timing
    master_stiffness_equation_.ApplyConstraints();
    double constraint_end = MPI_Wtime();   // End constraint timing

    double total_end = MPI_Wtime();
    double total_elapsed = total_end - total_start;
    double constraint_elapsed = constraint_end - constraint_start;

    double total_max, constraint_max;
    MPI_Reduce(&total_elapsed, &total_max, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&constraint_elapsed, &constraint_max, 1, MPI_DOUBLE, MPI_MAX, 0, PETSC_COMM_WORLD);

    if (rank == 0) {
        // Final time report
        PetscPrintf(PETSC_COMM_WORLD,
            "\nTiming Summary:\n"
            " - Total Time Elapsed: %.6f seconds\n"
            " - Constraint Application Time: %.6f seconds\n",
            total_max, constraint_max);

        // Log results
        std::ofstream log("timing_log.csv", std::ios::app);
        if (log.tellp() == 0) {
            log << "matrix,rows,cols,nnz,sparsity,constraints,total_time_s,constraint_time_s\n";
        }

        log << mtx_file << ","
            << nrows << "," << ncols << "," << nnz << ","
            << sparsity << "," << nC << ","
            << total_max << "," << constraint_max << "\n";
        log.close();
    }

    MatDestroy(&K);
    VecDestroy(&f);
    VecDestroy(&g);
    PetscFinalize();

    //homogenous_manual_constraints_trial(argc, argv);

    return 0;
}

std::tuple<int, Mat> read_matrix_from_mtx(const char* mtx_file) {
    PetscMPIInt rank;
    Mat K;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    // 1. Parse .mtx file
    std::vector<std::tuple<int, int, PetscScalar>> entries;
    PetscInt rows, cols, nnz;

    if (rank == 0) {
        std::ifstream fin(mtx_file);
        if (!fin.is_open()) {
            PetscPrintf(PETSC_COMM_WORLD, "Cannot open input file %s\n", mtx_file);
            PetscFinalize();
            return {};
        }

        std::string line;
        while (std::getline(fin, line)) {
            if (line[0] != '%') break;
        }

        std::istringstream header(line);
        header >> rows >> cols >> nnz;

        int r, c;
        PetscScalar v;
        while (fin >> r >> c >> v) {
            entries.emplace_back(r - 1, c - 1, v); // 1-based -> 0-based
        }
    }

    // 2. Broadcast matrix size
    MPI_Bcast(&rows, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // Step: Broadcast the entries
    if (rank != 0) {
        entries.resize(nnz);
    }
    MPI_Bcast(entries.data(), nnz * sizeof(std::tuple<int, int, PetscScalar>), MPI_BYTE, 0, PETSC_COMM_WORLD);

    // 3. Create matrix
    MatCreate(PETSC_COMM_WORLD, &K);
    MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, rows, cols);
    MatSetType(K, MATMPIAIJ);
    MatSetFromOptions(K);
    MatSetUp(K);

    // 4. Insert values
    PetscInt rstart, rend;
    MatGetOwnershipRange(K, &rstart, &rend);

    for (const auto& [row, col, val] : entries) {
        if (row >= rstart && row < rend) {
            MatSetValue(K, row, col, val, INSERT_VALUES);
        }
    }

    // 5. Assemble matrix
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    return {nnz, K};
}

boost::container::vector<Constraint> generate_constraints(const PetscInt size, const PetscInt nC, std::mt19937 gen,
                            const PetscInt nm_max, const float max_coeff) {
    boost::container::vector<Constraint> constraints;

    // Define the distribution range
    std::uniform_int_distribution<int> distr_node(0, size-1);
    std::uniform_int_distribution<int> distr_m(1, nm_max);
    std::uniform_real_distribution<float> distr_coeff(0.01, max_coeff);

    std::unordered_set<int> usedNodes;
    int s_idx, m_idx;
    float s_coeff, m_coeff, gap;

    for (size_t i{0}; i < nC; ++i) {
        // Create random slave node
        do {
            s_idx = distr_node(gen);
        } while (usedNodes.contains(s_idx));

        usedNodes.insert(s_idx);

        s_coeff = distr_coeff(gen);
        auto s = Term{s_idx, s_coeff, 1.0};

        // Create random master nodes
        boost::container::vector<Term> m(distr_m(gen));
        boost::container::vector<int> m_used(m.size());

        for(size_t j{0}; j < m.size(); ++j) {
            do {
                m_idx = distr_node(gen);
            } while (usedNodes.contains(m_idx) || std::find(m_used.begin(), m_used.end(), m_idx) != m_used.end());

            m_coeff = distr_coeff(gen);
            m[j] = Term{m_idx, m_coeff, 1.0};
            m_used[j] = m_idx;
        }

        gap = distr_coeff(gen);

        constraints.push_back(Constraint{s, m, gap});
    }

    return constraints;
}

bool homogenous_manual_constraints_trial(int argc, char **args) {
    PetscErrorCode ierr = PetscInitialize(&argc, &args, nullptr, "--help");
    PetscMPIInt rank, size;

    static constexpr int N{6};
    static constexpr int NNZ{16};

    PetscFunctionBeginUser;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    if (rank == 0)
        std::cout << "Initiating trial" << std::endl;

    if (ierr) {
        std::cout << "PetscInitialize failed with error code " << ierr << std::endl;
        return ierr;
    }

    boost::array<PetscInt, N + 1> ia_global{
        0, 2, 5, 8,
        11, 14, 16
    };
    boost::array<PetscInt, NNZ>
            ja_global{
                0, 1, 0, 1, 2, 1, 2, 3,
                2, 3, 4, 3, 4, 5, 4, 5
            };
    boost::array<PetscScalar, NNZ>
            a_global{
                100, -100, -100, 200, -100, -100,
                200, -100, -100, 200, -100, -100,
                200, -100, -100, 200
            };

    // Partition rows
    PetscInt rstart = 0, rend = 0;
    PetscInt local_rows = N / size + (rank < N % size ? 1 : 0);
    std::vector<PetscInt> counts(size), offset(size + 1, 0);
    for (int i = 0; i < size; ++i) {
        counts[i] = N / size + (i < N % size ? 1 : 0);
        offset[i + 1] = offset[i] + counts[i];
    }
    rstart = offset[rank];
    rend = offset[rank + 1];

    // Build local CSR
    std::vector<PetscInt> ia_local(local_rows + 1, 0);
    std::vector<PetscInt> ja_local;
    std::vector<PetscScalar> a_local;

    PetscInt row_nz = 0;
    for (PetscInt i = rstart; i < rend; ++i) {
        PetscInt row_start = ia_global[i];
        PetscInt row_end = ia_global[i + 1];
        ia_local[i - rstart + 1] = ia_local[i - rstart] + (row_end - row_start);
        for (PetscInt k = row_start; k < row_end; ++k) {
            ja_local.push_back(ja_global[k]);
            a_local.push_back(a_global[k]);
        }
    }

    Mat K;

    MatCreateMPIAIJWithArrays(PETSC_COMM_WORLD, local_rows, PETSC_DECIDE, PETSC_DETERMINE,
                            N, ia_local.data(),
                            ja_local.data(), a_local.data(), &K);

    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    PetscMasterStiffnessEquationAdaptee master_stiffness_equation_;
    master_stiffness_equation_.SetStiffnessMatrix(K);

    Vec f;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &f);
    VecSetFromOptions(f);
    VecSet(f, 0.0F);
    VecSetValue(f, 0, -20.0F, INSERT_VALUES);
    VecAssemblyBegin(f);
    VecAssemblyEnd(f);

    Vec g;
    VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, N, &g);
    VecSetFromOptions(g);
    VecSet(g, 0.0F);
    VecAssemblyBegin(g);
    VecAssemblyEnd(g);

    master_stiffness_equation_.SetGaps(g);
    master_stiffness_equation_.SetForces(f);

    boost::container::vector constraints{
        Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)}, 0.),
        Constraint(Term(1, 0.954F), boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)}, 0.),
    };

    master_stiffness_equation_.SetConstraints(constraints);

    master_stiffness_equation_.ApplyConstraints();
    master_stiffness_equation_.Solve();

    if (rank == 0)
        std::cout << "Trial complete" << std::endl;

    MatDestroy(&K);
    PetscFinalize();
    return true;
}
