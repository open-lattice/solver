//
// Created by Nitel Muhtaroglu on 2023-11-17.
//

#include <boost/assign.hpp>
#include <boost/container/vector.hpp>

#include <gtest/gtest.h>

#include "constraint.h"

TEST(TestConstraint, GetGap) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.2F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};
  ASSERT_FLOAT_EQ(0.2F, constraints.at(0).GetGap());
}

TEST(TestConstraint, GetSlaveTermIndex) {
  boost::container::vector<std::map<int, float>> coefficients{
      boost::assign::map_list_of(1, 1.0F)(5, -1.0F)};
  float gap{0.2F};

  boost::container::vector constraints{Constraint(coefficients.at(0), gap)};
  ASSERT_EQ(5, constraints.at(0).GetSlaveTermIndex());
}
