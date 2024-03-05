//
// Created by Nitel Muhtaroglu on 2023-11-06.
//

#include "constraint.h"

Constraint::Constraint(std::map<int, float> coefficients, float gap)
    : slave_term_(Term(std::prev(coefficients.end())->first,
                       std::prev(coefficients.end())->second)),
      master_terms_(
          {Term(coefficients.begin()->first, coefficients.begin()->second)}),
      gap_(gap) {}

Constraint::Constraint(const Term &slave,
                       boost::container::vector<Term> masters, float gap)
    : slave_term_(slave), master_terms_(std::move(masters)), gap_(gap) {}

float Constraint::GetGap() const { return Constraint::gap_; }

int Constraint::GetSlaveTermIndex() const {
  return Constraint::slave_term_.GetIndex();
}

float Constraint::GetSlaveTermCoefficient() const {
  return Constraint::slave_term_.GetCoefficient();
}

const boost::container::vector<Term> &Constraint::GetMasterTerms() const {
  return Constraint::master_terms_;
}
