//
// Created by Nitel Muhtaroglu on 2023-12-02.
//

#ifndef MULTI_FREEDOM_CONSTRAINTS_TERM_HPP_
#define MULTI_FREEDOM_CONSTRAINTS_TERM_HPP_

class Term {
 public:
  Term() = default;
  Term(int index, float coefficient, float degree = 1.0F)
      : index_(index), coefficient_(coefficient), degree_(degree) {}

  [[nodiscard]] int GetIndex() const { return Term::index_; }
  [[nodiscard]] float GetCoefficient() const { return Term::coefficient_; }

 private:
  int index_;
  float coefficient_;
  [[maybe_unused]] float degree_;
};

#endif // MULTI_FREEDOM_CONSTRAINTS_TERM_HPP_
