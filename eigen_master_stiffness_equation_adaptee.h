//
// Created by Nitel Muhtaroglu on 2023-11-07.
//

#ifndef MULTI_FREEDOM_CONSTRAINTS_EIGEN_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_
#define MULTI_FREEDOM_CONSTRAINTS_EIGEN_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_

#include <map>
#include <vector>

#include <boost/container/vector.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

#include "constraint.h"
#include "master_stiffness_equation.hpp"

class EigenMasterStiffnessEquationAdaptee : public MasterStiffnessEquation {
 public:
  EigenMasterStiffnessEquationAdaptee();
  [[nodiscard]] const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
  GetStiffnessMatrix() const;
  [[nodiscard]] const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
  GetModifiedStiffnessMatrix() const;
  [[nodiscard]] const Eigen::Vector<float, Eigen::Dynamic> &GetForces() const;
  [[nodiscard]] const Eigen::Vector<float, Eigen::Dynamic> &GetGaps() const;
  [[nodiscard]] const Eigen::Vector<float, Eigen::Dynamic> &
  GetModifiedForces() const;
  [[nodiscard]] const Eigen::Vector<float, Eigen::Dynamic> &
  GetDisplacements() const;
  [[nodiscard]] const Eigen::Vector<float, Eigen::Dynamic> &
  GetModifiedDisplacements() const;
  [[nodiscard]] const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
  GetTransformationMatrix() const;

  void SetStiffnessMatrix(
      const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &);
  void SetForces(const Eigen::Vector<float, Eigen::Dynamic> &);
  void SetGaps(const Eigen::Vector<float, Eigen::Dynamic> &);
  void Solve() override;
  void ApplyConstraints() override;
 private:
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> stiffness_matrix_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
      modified_stiffness_matrix_;
  Eigen::Vector<float, Eigen::Dynamic> displacements_;
  Eigen::Vector<float, Eigen::Dynamic> modified_displacements_;
  Eigen::Vector<float, Eigen::Dynamic> forces_;
  Eigen::Vector<float, Eigen::Dynamic> gaps_;
  Eigen::Vector<float, Eigen::Dynamic> modified_forces_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> transformation_matrix_;
};

#endif // MULTI_FREEDOM_CONSTRAINTS_EIGEN_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_
