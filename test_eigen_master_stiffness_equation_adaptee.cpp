//
// Created by Nitel Muhtaroglu on 2023-11-17.
//

#include <boost/assign.hpp>
#include <boost/container/vector.hpp>

#include <gtest/gtest.h>

#include "constraint.h"
#include "eigen_master_stiffness_equation_adaptee.h"
#include "term.hpp"

class EigenMasterStiffnessEquationAdapteeTest : public ::testing::Test {
 protected:
  EigenMasterStiffnessEquationAdaptee master_stiffness_equation_;
  static constexpr int kGlobalProblemSize{6};
  void SetUp() override {
    Eigen::Matrix<float, kGlobalProblemSize, kGlobalProblemSize> K;
    Eigen::Vector<float, kGlobalProblemSize> _f;
    Eigen::Vector<float, kGlobalProblemSize> _g;
    float k11{100.F};
    float k12{-100.F};
    float k22{200.F};
    float k23{-100.F};
    float k33{200.F};
    float k34{-100.F};
    float k44{200.F};
    float k45{-100.F};
    float k55{200.F};
    float k56{-100.F};
    float k66{200.F};
    K.setZero();
    K(0, 0) = k11;
    K(0, 1) = k12;
    K(1, 0) = K(0, 1);
    K(1, 1) = k22;
    K(1, 2) = k23;
    K(2, 1) = K(1, 2);
    K(2, 2) = k33;
    K(2, 3) = k34;
    K(3, 2) = K(2, 3);
    K(3, 3) = k44;
    K(3, 4) = k45;
    K(4, 3) = K(3, 4);
    K(4, 4) = k55;
    K(4, 5) = k56;
    K(5, 4) = K(4, 5);
    K(5, 5) = k66;

    _f.setZero();
    _f(0) = -20.0F;

    _g.setZero();
    master_stiffness_equation_.SetStiffnessMatrix(K);
    master_stiffness_equation_.SetForces(_f);
    master_stiffness_equation_.SetGaps(_g);
  }

  void TearDown() override {}
};

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestForces) {
  Eigen::Vector<float, EigenMasterStiffnessEquationAdapteeTest::kGlobalProblemSize>
      expected_forces;
  expected_forces.setZero();
  expected_forces(0) = -20.0F;
  auto actual_forces =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.GetForces();
  ASSERT_EQ(actual_forces, expected_forces);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestStiffnessMatrix) {
  Eigen::Matrix<float, EigenMasterStiffnessEquationAdapteeTest::kGlobalProblemSize,
                EigenMasterStiffnessEquationAdapteeTest::kGlobalProblemSize>
      expected_stiffness_matrix;
  float k11{100.F};
  float k12{-100.F};
  float k22{200.F};
  float k23{-100.F};
  float k33{200.F};
  float k34{-100.F};
  float k44{200.F};
  float k45{-100.F};
  float k55{200.F};
  float k56{-100.F};
  float k66{200.F};
  expected_stiffness_matrix.setZero();
  expected_stiffness_matrix(0, 0) = k11;
  expected_stiffness_matrix(0, 1) = k12;
  expected_stiffness_matrix(1, 0) = expected_stiffness_matrix(0, 1);
  expected_stiffness_matrix(1, 1) = k22;
  expected_stiffness_matrix(1, 2) = k23;
  expected_stiffness_matrix(2, 1) = expected_stiffness_matrix(1, 2);
  expected_stiffness_matrix(2, 2) = k33;
  expected_stiffness_matrix(2, 3) = k34;
  expected_stiffness_matrix(3, 2) = expected_stiffness_matrix(2, 3);
  expected_stiffness_matrix(3, 3) = k44;
  expected_stiffness_matrix(3, 4) = k45;
  expected_stiffness_matrix(4, 3) = expected_stiffness_matrix(3, 4);
  expected_stiffness_matrix(4, 4) = k55;
  expected_stiffness_matrix(4, 5) = k56;
  expected_stiffness_matrix(5, 4) = expected_stiffness_matrix(4, 5);
  expected_stiffness_matrix(5, 5) = k66;

  auto actual_stiffness_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetStiffnessMatrix();
  ASSERT_EQ(actual_stiffness_matrix, expected_stiffness_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformationMatrixWithSingleHomogeneousConstraint) {

  boost::container::vector constraints{
      Constraint(Term(5, -1.0F), boost::container::vector{Term(1, 1.0F)})};
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();

  static constexpr int kRowSize{6};
  static constexpr int kColumnSize{5};

  Eigen::Matrix<float, kRowSize, kColumnSize> expected_transformation_matrix;
  expected_transformation_matrix.setZero();

  expected_transformation_matrix(0, 0) = 1.0F;
  expected_transformation_matrix(1, 1) = 1.0F;
  expected_transformation_matrix(2, 2) = 1.0F;
  expected_transformation_matrix(3, 3) = 1.0F;
  expected_transformation_matrix(4, 4) = 1.0F;
  expected_transformation_matrix(5, 1) = 1.0F;

  auto actual_transformation_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetTransformationMatrix();

  ASSERT_EQ(actual_transformation_matrix, expected_transformation_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformationMatrixWithMultipleNonHomogeneousConstraints) {
  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();

  static constexpr int kRowSize{6};
  static constexpr int kColumnSize{4};

  Eigen::Matrix<float, kRowSize, kColumnSize> expected_transformation_matrix;
  expected_transformation_matrix.setZero();

  expected_transformation_matrix(0, 0) = 1.0F;
  expected_transformation_matrix(1, 1) = -0.234801F;
  expected_transformation_matrix(1, 2) = 0.620545F;
  expected_transformation_matrix(2, 1) = 1.0F;
  expected_transformation_matrix(3, 2) = 1.0F;
  expected_transformation_matrix(4, 3) = 1.0F;
  expected_transformation_matrix(5, 3) = 5.597315F;

  auto actual_transformation_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetTransformationMatrix();

  ASSERT_EQ(actual_transformation_matrix, expected_transformation_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformedStiffnessMatrixWithMultipleNonHomogeneousConstraints) {

  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();

  static constexpr int kRowSize{4};
  static constexpr int kColumnSize{4};

  Eigen::Matrix<float, kRowSize, kColumnSize>
      expected_transformed_stiffness_matrix;
  expected_transformed_stiffness_matrix.setZero();

  expected_transformed_stiffness_matrix(0, 0) = 100.0F;
  expected_transformed_stiffness_matrix(0, 1) = 23.4801F;
  expected_transformed_stiffness_matrix(0, 2) = -62.0545F;
  expected_transformed_stiffness_matrix(1, 0) = 23.4801F;
  expected_transformed_stiffness_matrix(1, 1) = 257.986502F;
  expected_transformed_stiffness_matrix(1, 2) = -191.195417F;
  expected_transformed_stiffness_matrix(2, 0) = -62.0545F;
  expected_transformed_stiffness_matrix(2, 1) = -191.195417F;
  expected_transformed_stiffness_matrix(2, 2) = 277.015219F;
  expected_transformed_stiffness_matrix(2, 3) = -100.0F;
  expected_transformed_stiffness_matrix(3, 2) = -100.0F;
  expected_transformed_stiffness_matrix(3, 3) = 5346.524042F;

  auto actual_transformed_stiffness_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedStiffnessMatrix();
  ASSERT_EQ(actual_transformed_stiffness_matrix,
            expected_transformed_stiffness_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformedForceVectorWithMultipleNonHomogeneousConstraints) {

  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();

  static constexpr int kSize{4};
  Eigen::Vector<float, kSize> expected_forces;
  expected_forces.setZero();
  expected_forces(0) = -5.3249F;
  expected_forces(1) = 21.566556F;
  expected_forces(2) = -18.21312F;
  expected_forces(3) = -2189.450731F;
  auto actual_forces = EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
      .GetModifiedForces();
  ASSERT_EQ(actual_forces, expected_forces);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformedDisplacementsWithMultipleNonHomogeneousConstraints) {

  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();

  static constexpr int kSize{4};
  Eigen::Vector<float, kSize> expected_transformed_displacements;
  expected_transformed_displacements.setZero();
  expected_transformed_displacements(0) = -0.262133F;
  expected_transformed_displacements(1) = -0.197356F;
  expected_transformed_displacements(2) = -0.411289F;
  expected_transformed_displacements(3) = -0.417202F;
  auto actual_transformed_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedDisplacements();
  ASSERT_EQ(actual_transformed_displacements,
            expected_transformed_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestFinalDisplacementsWithMultipleNonHomogeneousConstraints) {

  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();

  static constexpr int kSize{6};
  Eigen::Vector<float, kSize> expected_displacements;
  expected_displacements.setZero();
  expected_displacements(0) = -0.262133F;
  expected_displacements(1) = -0.062133F;
  expected_displacements(2) = -0.197356F;
  expected_displacements(3) = -0.411289F;
  expected_displacements(4) = -0.417202F;
  expected_displacements(5) = -0.187559F;
  auto actual_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetDisplacements();
  ASSERT_EQ(actual_displacements, expected_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestForceAfterApplyConstraints) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};

  static constexpr int kSize{5};
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  Eigen::Vector<float, kSize> expected_forces;
  expected_forces.setZero();
  expected_forces(0) = -20.0F;
  auto actual_forces = EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
      .GetModifiedForces();
  ASSERT_EQ(actual_forces, expected_forces);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestStiffnessMatrixAfterApplyConstraints) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();

  static constexpr int kSize{5};
  Eigen::Matrix<float, kSize, kSize> expected_stiffness_matrix;
  expected_stiffness_matrix.setZero();

  expected_stiffness_matrix(0, 0) = 100;
  expected_stiffness_matrix(0, 1) = -100;

  expected_stiffness_matrix(1, 0) = -100;
  expected_stiffness_matrix(1, 1) = 400;
  expected_stiffness_matrix(1, 2) = -100;
  expected_stiffness_matrix(1, 4) = -100;

  expected_stiffness_matrix(2, 1) = -100;
  expected_stiffness_matrix(2, 2) = 200;
  expected_stiffness_matrix(2, 3) = -100;

  expected_stiffness_matrix(3, 2) = -100;
  expected_stiffness_matrix(3, 3) = 200;
  expected_stiffness_matrix(3, 4) = -100;

  expected_stiffness_matrix(4, 1) = -100;
  expected_stiffness_matrix(4, 3) = -100;
  expected_stiffness_matrix(4, 4) = 200;

  auto actual_stiffness_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedStiffnessMatrix();
  ASSERT_EQ(actual_stiffness_matrix, expected_stiffness_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestModifiedDisplacementsWithoutGaps) {

  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kSize{5};
  Eigen::Vector<float, kSize> expected_modified_displacements;
  expected_modified_displacements.setZero();
  expected_modified_displacements(0) = -0.4F;
  expected_modified_displacements(1) = -0.2F;
  expected_modified_displacements(2) = -0.2F;
  expected_modified_displacements(3) = -0.2F;
  expected_modified_displacements(4) = -0.2F;
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();
  auto actual_modified_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedDisplacements();

  ASSERT_EQ(actual_modified_displacements, expected_modified_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest,
       TestTransformationMatrixWithMultipleHomogeneousMfcs) {

  boost::container::vector constraints{
      Constraint(Term(5, -1.0F), boost::container::vector{Term(1, 1.0F)}),
      Constraint(Term(3, 4.0F), boost::container::vector{Term(0, 1.0F)}),
      Constraint(Term(2, 1.0F),
                 boost::container::vector{Term(0, -0.125F), Term(4, 0.5F)})};
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kRowSize{6};
  static constexpr int kColumnSize{3};

  Eigen::Matrix<float, kRowSize, kColumnSize> expected_transformation_matrix;
  expected_transformation_matrix.setZero();

  expected_transformation_matrix(0, 0) = 1.0F;
  expected_transformation_matrix(1, 1) = 1.0F;
  expected_transformation_matrix(2, 0) = 0.125F;
  expected_transformation_matrix(2, 2) = -0.5F;
  expected_transformation_matrix(3, 0) = -0.25F;
  expected_transformation_matrix(4, 2) = 1.0F;
  expected_transformation_matrix(5, 1) = 1.0F;

  auto actual_transformation_matrix =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetTransformationMatrix();

  ASSERT_EQ(actual_transformation_matrix, expected_transformation_matrix);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestDisplacementsWithoutGaps) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kSize{6};
  Eigen::Vector<float, kSize> expected_displacements;
  expected_displacements.setZero();
  expected_displacements(0) = -0.4f;
  expected_displacements(1) = -0.2f;
  expected_displacements(2) = -0.2f;
  expected_displacements(3) = -0.2f;
  expected_displacements(4) = -0.2f;
  expected_displacements(5) = -0.2f;
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();
  auto actual_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetDisplacements();

  ASSERT_EQ(actual_displacements, expected_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestDisplacementsWithGaps) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.2F};

  boost::container::vector<Constraint> constraints{
      Constraint(coefficients.at(0), gap)};

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  Eigen::Vector<float, EigenMasterStiffnessEquationAdapteeTest::kGlobalProblemSize>
      expected_displacements;
  expected_displacements.setZero();
  expected_displacements(0) = -0.2F;
  expected_displacements(1) = 0.0F;
  expected_displacements(2) = -0.05F;
  expected_displacements(3) = -0.1F;
  expected_displacements(4) = -0.15F;
  expected_displacements(5) = -0.2F;

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();
  auto actual_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetDisplacements();

  ASSERT_EQ(actual_displacements, expected_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestModifiedDisplacementsWithGaps) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0f)(5, -1.0f)};
  float gap{0.2f};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kSize{5};
  Eigen::Vector<float, kSize> expected_modified_displacements;
  expected_modified_displacements.setZero();
  expected_modified_displacements(0) = -0.2;
  expected_modified_displacements(1) = 0.0;
  expected_modified_displacements(2) = -0.05;
  expected_modified_displacements(3) = -0.1;
  expected_modified_displacements(4) = -0.15;
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.Solve();
  auto actual_modified_displacements =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedDisplacements();

  ASSERT_EQ(actual_modified_displacements, expected_modified_displacements);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestGaps) {

  boost::container::vector constraints{
      Constraint(Term(5, 0.149F), boost::container::vector{Term(4, -0.834F)},
                 0.32F),
      Constraint(Term(1, 0.954F),
                 boost::container::vector{Term(2, 0.224F), Term(3, -0.592F)},
                 0.14F),
  };

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kSize{6};
  Eigen::Vector<float, kSize> expected_gaps;
  expected_gaps.setZero();
  expected_gaps(1) = 0.146751F;
  expected_gaps(5) = 2.147651F;

  auto actual_gaps =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.GetGaps();
  ASSERT_EQ(actual_gaps, expected_gaps);
}

TEST_F(EigenMasterStiffnessEquationAdapteeTest, TestModifiedForcesWithGaps) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.2F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};

  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.SetConstraints(
      constraints);
  EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_.ApplyConstraints();
  static constexpr int kSize{5};
  Eigen::Vector<float, kSize> expected_modified_forces;
  expected_modified_forces.setZero();
  expected_modified_forces(0) = -20.f;
  expected_modified_forces(1) = 40.f;
  expected_modified_forces(4) = -20.f;
  auto actual_modified_forces =
      EigenMasterStiffnessEquationAdapteeTest::master_stiffness_equation_
          .GetModifiedForces();
  ASSERT_EQ(actual_modified_forces, expected_modified_forces);
}
