//
// Created by Nitel Muhtaroglu on 2023-11-07.
//

#include "eigen_master_stiffness_equation_adaptee.h"

EigenMasterStiffnessEquationAdaptee::EigenMasterStiffnessEquationAdaptee() = default;

const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetStiffnessMatrix() const {
  return EigenMasterStiffnessEquationAdaptee::stiffness_matrix_;
}

const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetModifiedStiffnessMatrix() const {
  return EigenMasterStiffnessEquationAdaptee::modified_stiffness_matrix_;
}

const Eigen::Vector<float, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetForces() const {
  return EigenMasterStiffnessEquationAdaptee::forces_;
}

const Eigen::Vector<float, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetGaps() const {
  return EigenMasterStiffnessEquationAdaptee::gaps_;
}

const Eigen::Vector<float, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetModifiedForces() const {
  return EigenMasterStiffnessEquationAdaptee::modified_forces_;
}

const Eigen::Vector<float, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetDisplacements() const {
  return EigenMasterStiffnessEquationAdaptee::displacements_;
}

const Eigen::Vector<float, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetModifiedDisplacements() const {
  return EigenMasterStiffnessEquationAdaptee::modified_displacements_;
}

void EigenMasterStiffnessEquationAdaptee::ApplyConstraints() {
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> _transformation_matrix_;
  _transformation_matrix_.resize(MasterStiffnessEquation::ReadActiveRowSize(),
                                 MasterStiffnessEquation::ReadActiveColumnSize());
  _transformation_matrix_.setZero();
  _transformation_matrix_.diagonal().array() += 1.0F;

  for (const auto &constraint : MasterStiffnessEquation::GetConstraints()) {
    EigenMasterStiffnessEquationAdaptee::gaps_(constraint.GetSlaveTermIndex()) =
        constraint.GetGap() / constraint.GetSlaveTermCoefficient();

    MasterStiffnessEquation::EraseFromActiveRows(constraint.GetSlaveTermIndex());
    _transformation_matrix_.col(constraint.GetSlaveTermIndex()).setZero();

    for (const auto &master_term : constraint.GetMasterTerms()) {
      _transformation_matrix_.row(constraint.GetSlaveTermIndex()) +=
          _transformation_matrix_.row(master_term.GetIndex()) *
              master_term.GetCoefficient() / -constraint.GetSlaveTermCoefficient();
    }
  }

  EigenMasterStiffnessEquationAdaptee::transformation_matrix_.resize(
      MasterStiffnessEquation::ReadActiveRowSize(), MasterStiffnessEquation::ReadActiveColumnSize());
  EigenMasterStiffnessEquationAdaptee::transformation_matrix_.setZero();

  EigenMasterStiffnessEquationAdaptee::transformation_matrix_ =
      _transformation_matrix_(MasterStiffnessEquation::GetActiveRows(),
                              MasterStiffnessEquation::GetActiveColumns());

  EigenMasterStiffnessEquationAdaptee::modified_forces_ =
      EigenMasterStiffnessEquationAdaptee::transformation_matrix_.transpose() *
          (EigenMasterStiffnessEquationAdaptee::forces_ -
              EigenMasterStiffnessEquationAdaptee::stiffness_matrix_ *
                  EigenMasterStiffnessEquationAdaptee::gaps_);

  EigenMasterStiffnessEquationAdaptee::modified_stiffness_matrix_ =
      (EigenMasterStiffnessEquationAdaptee::transformation_matrix_.transpose() *
          EigenMasterStiffnessEquationAdaptee::stiffness_matrix_) *
          EigenMasterStiffnessEquationAdaptee::transformation_matrix_;
}

void EigenMasterStiffnessEquationAdaptee::Solve() {
  EigenMasterStiffnessEquationAdaptee::modified_displacements_ =
      EigenMasterStiffnessEquationAdaptee::modified_stiffness_matrix_.fullPivLu().solve(
          EigenMasterStiffnessEquationAdaptee::modified_forces_);

  EigenMasterStiffnessEquationAdaptee::displacements_ =
      EigenMasterStiffnessEquationAdaptee::transformation_matrix_ *
          EigenMasterStiffnessEquationAdaptee::modified_displacements_ +
          EigenMasterStiffnessEquationAdaptee::gaps_;
}

void EigenMasterStiffnessEquationAdaptee::SetStiffnessMatrix(
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>
    &stiffness_matrix) {
  EigenMasterStiffnessEquationAdaptee::stiffness_matrix_ = stiffness_matrix;
  MasterStiffnessEquation::InitializeReductionVectors(
      EigenMasterStiffnessEquationAdaptee::stiffness_matrix_.rows());
}

void EigenMasterStiffnessEquationAdaptee::SetForces(
    const Eigen::Vector<float, Eigen::Dynamic> &forces) {
  EigenMasterStiffnessEquationAdaptee::forces_ = forces;
}

void EigenMasterStiffnessEquationAdaptee::SetGaps(
    const Eigen::Vector<float, Eigen::Dynamic> &gaps) {
  EigenMasterStiffnessEquationAdaptee::gaps_ = gaps;
}

const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &
EigenMasterStiffnessEquationAdaptee::GetTransformationMatrix() const {
  return EigenMasterStiffnessEquationAdaptee::transformation_matrix_;
}


