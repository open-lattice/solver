//
// Created by Nitel Muhtaroglu on 2023-11-06.
//

#include <iostream>
#include <map>
#include <vector>
#include <utility>

#include <boost/container/vector.hpp>

#include "term.hpp"

#ifndef MULTIFREEDOM_CONSTRAINTS_CONSTRAINT_H_
#define MULTIFREEDOM_CONSTRAINTS_CONSTRAINT_H_

class Constraint {
 public:
  explicit Constraint(std::map<int, float>, float = 0.0F);
  Constraint(const Term &, boost::container::vector<Term>, float = 0.0F);
  [[nodiscard]] float GetGap() const;
  [[nodiscard]] int GetSlaveTermIndex() const;
  [[nodiscard]] float GetSlaveTermCoefficient() const;
  [[nodiscard]] const boost::container::vector<Term> &GetMasterTerms() const;

 private:
  Term slave_term_;
  boost::container::vector<Term> master_terms_;
  float gap_;
};

#endif // MULTIFREEDOM_CONSTRAINTS_CONSTRAINT_H_
