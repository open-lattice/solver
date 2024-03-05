//
// Created by Nitel Muhtaroglu on 2023-12-22.
//

#ifndef MULTI_FREEDOM_CONSTRAINTS_MATRIX_HPP_
#define MULTI_FREEDOM_CONSTRAINTS_MATRIX_HPP_
#include <cstdio>
#include <boost/unordered_set.hpp>

class MasterStiffnessEquation {
 public:
  virtual ~MasterStiffnessEquation() = default;

  void SetConstraints(
      const boost::container::vector<Constraint> &constraints) {
    MasterStiffnessEquation::constraints_ = constraints;
    for (const auto &constraint : MasterStiffnessEquation::constraints_) {
      MasterStiffnessEquation::slave_indices_.insert(constraint.GetSlaveTermIndex());
    }
  }

  const boost::container::vector<Constraint> &GetConstraints() {
    return MasterStiffnessEquation::constraints_;
  }
  virtual void ApplyConstraints() = 0;
  virtual void Solve() = 0;

  const std::vector<long> &GetActiveRows() {
    return MasterStiffnessEquation::active_rows_;
  }

  const std::vector<long> &GetActiveColumns() {
    return MasterStiffnessEquation::active_columns_;
  }

  size_t ReadActiveRowSize() {
    return MasterStiffnessEquation::active_rows_.size();
  }

  size_t ReadActiveColumnSize() {
    return MasterStiffnessEquation::active_columns_.size();
  }

  void EraseFromActiveRows(int index) {
    MasterStiffnessEquation::active_rows_.erase(MasterStiffnessEquation::active_rows_.begin() + index);
  }

  void InitializeReductionVectors(long problem_size) {
    for (int i{0}; i < problem_size; ++i) {
      MasterStiffnessEquation::active_rows_.push_back(i);
      MasterStiffnessEquation::active_columns_.push_back(i);
    }
  }

  size_t GetConstraintCount() const {
    return MasterStiffnessEquation::constraints_.size();
  }

  bool IsSlaveIndexForAConstraint(int index) {
    return MasterStiffnessEquation::slave_indices_.find(index) != MasterStiffnessEquation::slave_indices_.end();
  }

 private:
  boost::container::vector<Constraint> constraints_;
  boost::unordered_set<int> slave_indices_;
  std::vector<long> active_rows_;
  std::vector<long> active_columns_;
};

#endif //MULTI_FREEDOM_CONSTRAINTS_MATRIX_HPP_
