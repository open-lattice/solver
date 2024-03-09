//
// Created by Nitel Muhtaroglu on 2023-12-23.
//

#ifndef MULTI_FREEDOM_CONSTRAINTS_PETSC_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_
#define MULTI_FREEDOM_CONSTRAINTS_PETSC_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_

#include <map>
#include <vector>
#include <unordered_set>

#include <boost/container/vector.hpp>
#include <boost/shared_ptr.hpp>

#include <petscmat.h>
#include <petscvec.h>
#include <petscsys.h>
#include <petscksp.h>

#include "constraint.h"
#include "master_stiffness_equation.hpp"

class PetscMasterStiffnessEquationAdaptee : public MasterStiffnessEquation {
 public:
  PetscMasterStiffnessEquationAdaptee();
  [[nodiscard]] const Mat &GetStiffnessMatrix() const;
  [[nodiscard]] const Mat &GetModifiedStiffnessMatrix() const;
  [[nodiscard]] const Vec &GetForces() const;
  [[nodiscard]] const Vec &GetGaps() const;
  [[nodiscard]] const Vec &GetModifiedForces() const;
  [[nodiscard]] const Vec &GetDisplacements() const;
  [[nodiscard]] const Vec &GetModifiedDisplacements() const;
  [[nodiscard]] const Mat &GetTransformationMatrix() const;

  void SetStiffnessMatrix(const Mat &);
  void SetForces(const Vec &);
  void SetGaps(const Vec &);
  void Solve() override;
  void ApplyConstraints() override;
 private:
  Mat stiffness_matrix_;
  Vec forces_;
  Vec gaps_;
  Mat transformation_matrix_;
  Vec modified_forces_;
  Mat modified_stiffness_matrix_;
  Vec modified_displacements_;
  Vec displacements_;
  boost::bimap<unsigned long, unsigned long> global_to_master_indices_lookup_;
  void InitializeVector(Vec *);
  unsigned long InitializeGlobalToMasterIndicesLookupTable(unsigned long);
};

#endif // MULTI_FREEDOM_CONSTRAINTS_PETSC_MASTER_STIFFNESS_EQUATION_ADAPTEE_H_
