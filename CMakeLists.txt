cmake_minimum_required(VERSION 3.15)
project(multifreedom_constraints)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/modules)

set(ENV{PETSC_DIR} /home/nitelm/workspace/petsc)
set(ENV{PETSC_ARCH} arch-linux-c-debug)
set(PETSC_INCLUDE /home/nitelm/workspace/petsc/include)

find_package(PETSc)
add_library(multifreedom_constraints SHARED constraint.cpp constraint.h
        eigen_master_stiffness_equation_adaptee.cpp eigen_master_stiffness_equation_adaptee.h
        petsc_master_stiffness_equation_adaptee.cpp petsc_master_stiffness_equation_adaptee.h
        term.hpp master_stiffness_equation.hpp petsc_master_stiffness_equation_adaptee.cpp)
target_include_directories(multifreedom_constraints PRIVATE /usr/include)
target_include_directories(multifreedom_constraints PRIVATE /usr/include/eigen3)
target_include_directories(multifreedom_constraints PRIVATE ${PETSC_INCLUDE})
target_include_directories(multifreedom_constraints PRIVATE ${PETSC_DIR}/${PETSC_ARCH}/include)
target_link_libraries(multifreedom_constraints PRIVATE ${PETSC_LIBRARIES})

if (${PETSC_FOUND})
    add_executable(main main.cpp)
    target_include_directories(main PRIVATE ${PETSC_INCLUDE} ${PETSC_DIR}/${PETSC_ARCH}/include)
    target_link_libraries(main PRIVATE ${PETSC_LIBRARIES} multifreedom_constraints)
endif (${PETSC_FOUND})

find_package(GTest REQUIRED)
enable_testing()
add_executable(
        run_tests
        run_tests.cpp
        test_constraint.cpp
        test_eigen_master_stiffness_equation_adaptee.cpp
        test_petsc_master_stiffness_equation_adaptee.cpp)
target_link_libraries(
        run_tests
        ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} multifreedom_constraints
)
target_include_directories(run_tests PRIVATE ${GTEST_INCLUDE_DIRS} /usr/include/eigen3)
target_include_directories(run_tests PRIVATE ${PETSC_INCLUDE} ${PETSC_DIR}/${PETSC_ARCH}/include)

include(GoogleTest)
gtest_discover_tests(run_tests)
