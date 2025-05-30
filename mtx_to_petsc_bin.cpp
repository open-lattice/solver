#include <petscmat.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <filesystem>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, nullptr, nullptr);

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Usage: %s input_matrix.mtx output_folder/\n", argv[0]);
        }
        PetscFinalize();
        return -1;
    }

    const std::string input_path = argv[1];
    const std::string output_folder = argv[2];
    const std::string base_filename = std::filesystem::path(input_path).stem().string();
    const std::string output_path = output_folder + "/" + base_filename + ".bin";
    const std::string info_path = output_folder + "/" + base_filename + ".info";

    PetscInt m = 0, n = 0, nnz = 0;
    std::vector<std::tuple<PetscInt, PetscInt, PetscScalar>> entries;

    if (rank == 0) {
        std::ifstream fin(input_path);
        if (!fin.is_open()) {
            PetscPrintf(PETSC_COMM_WORLD, "Error opening file: %s\n", input_path.c_str());
            PetscFinalize();
            return -1;
        }

        std::string line;
        while (std::getline(fin, line)) {
            if (line[0] == '%') continue; // skip comments
            std::istringstream header(line);
            header >> m >> n >> nnz;
            break;
        }

        PetscInt i, j;
        PetscScalar val;
        while (fin >> i >> j >> val) {
            entries.emplace_back(i - 1, j - 1, val);
        }
    }

    // 2. Broadcast matrix size
    MPI_Bcast(&m, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&n, 1, MPIU_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // Step: Broadcast the entries
    if (rank != 0) {
        entries.resize(nnz);
    }
    MPI_Bcast(entries.data(), nnz * sizeof(std::tuple<int, int, PetscScalar>), MPI_BYTE, 0, PETSC_COMM_WORLD);

    // 3. Create matrix
    Mat K;
    MatCreate(PETSC_COMM_WORLD, &K);
    MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, m, n);
    MatSetType(K, MATMPIAIJ);
    MatSetFromOptions(K);
    MatSetUp(K);

    // 4. Insert values
    PetscInt rstart, rend;
    MatGetOwnershipRange(K, &rstart, &rend);

    for (const auto &[row, col, val]: entries) {
        if (row >= rstart && row < rend) {
            MatSetValue(K, row, col, val, INSERT_VALUES);
        }
    }

    // 5. Assemble matrix
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    // Save matrix in binary
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, output_path.c_str(), FILE_MODE_WRITE, &viewer);
    MatView(K, viewer);
    PetscViewerDestroy(&viewer);

    std::ofstream info(info_path.c_str(), std::ios::app);
    if (info.is_open() && rank == 0) {
        info << m << " " << n << " " << nnz;
        info.close();
    }

    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Saved PETSc binary matrix to: %s\n", output_path.c_str());
    }

    MatDestroy(&K);
    PetscFinalize();
    return 0;
}

