#pragma once

#include <sys/types.h>

#include "linear-algebra.h"

// Eigensolver class mean to obtain the stationary distribution of a
// matrix representing a Markov chain.
class eigen_solver {
 public:
  eigen_solver(size_t max_iterations = 10000, size_t max_basis_vectors = 32);

  int solve(const linear_algebra::matrix& A,
            linear_algebra::vector& x,
            double* ritz_value = nullptr) const noexcept;

 public:
  // Maximum number of iterations.
  const size_t max_iterations;

  // Largest number of basis vectors that will be used when doing
  // implicitly restarted Arnoldi iterations.
  const size_t max_basis_vectors;
};
