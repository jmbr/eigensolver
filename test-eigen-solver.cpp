#include <cmath>

#include <iostream>

#include <gtest/gtest.h>

#include "linear-algebra.h"
#include "eigen-solver.h"

const double tolerance = 1e-8;

namespace la = linear_algebra;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " matrix-file vector-file\n";
    return EXIT_FAILURE;
  }

  const std::string input_filename(argv[1]);
  const std::string output_filename(argv[2]);

  la::matrix A;

  int status;

  status = A.load(input_filename);
  if (status != 0) {
    std::cerr << "Unable to load \"" << input_filename << "\"\n";
    return EXIT_FAILURE;
  }

  la::vector v(A.size());
  v.ones();

  double ritz_value = 0.0;

  eigen_solver eigsolver;

  status = eigsolver.solve(A, v, &ritz_value);
  if (status != 0) {
    std::cerr << "Unable to compute dominant eigenpair.\n";
    return EXIT_FAILURE;
  }

  std::cout << "Ritz value: " << ritz_value << "\n";

  v.save(output_filename);

  if (fabs(ritz_value - 1.0) >= tolerance) {
    std::cerr << "Warning: Ritz value may not be close enough to one.\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
