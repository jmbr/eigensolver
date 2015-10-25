#pragma once

#include <vector>

#include <SuiteSparseQR.hpp>

#include "linear-algebra.h"

// Class for solving sparse square systems of linear equations.
class linear_solver_spqr {
 public:
  linear_solver_spqr();

  ~linear_solver_spqr();

  int setup(const std::vector<long>& i, const std::vector<long>& j,
            const std::vector<double>& values, size_t dim);

  int solve(const std::vector<double>& right_hand_side,
            std::vector<double>& soluton);

  int solve(const double* right_hand_side, double* solution);

 private:
  cholmod_common common;
  cholmod_sparse* mat;

  SuiteSparseQR_factorization<double>* QR;
};
