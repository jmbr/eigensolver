#include <cstdlib>

#include <vector>
#include <iostream>

#include <gtest/gtest.h>

// #include "linear-solver-umfpack.h"
#include "linear-solver-spqr.h"
#include "misc.h"

namespace {

class LinearSolverTest : public ::testing::Test {
 public:
  virtual void SetUp() {
    n = 1000000;

    i.push_back(0);
    j.push_back(1);
    a.push_back(1.0);

    size_t k;
    for (k = 1; k < n-1; ++k) {
      i.push_back(k);
      j.push_back(k-1);
      a.push_back(0.25);

      i.push_back(k);
      j.push_back(k+1);
      a.push_back(0.75);
    }

    i.push_back(n-1);
    j.push_back(n-2);
    a.push_back(1.0);

    b = std::vector<double>(n, 1.0);
    u = std::vector<double>(n, 0.0);

    // MATLAB "one"-liner:
    // n = 10; A = zeros(n); A(1, 2) = 1; for k = 2:(n-1); A(k, k-1) =
    // 0.25; A(k, k+1) = 0.75; end; A(end, end-1) = 1; b =
    // ones(size(A, 1), 1); disp(A\b);
  }

  size_t n;
  std::vector<long> i, j;
  std::vector<double> a;

  std::vector<double> b, u;
};

#if 0
TEST_F(LinearSolverTest, UMFPACKTest) {
  EXPECT_EQ(i.size(), j.size());
  EXPECT_EQ(i.size(), a.size());
  EXPECT_EQ(i.size(), 2 + 2 * (n - 2));

  linear_solver_umfpack linsolver;

  int status;

  status = linsolver.setup(i, j, a, n);
  EXPECT_EQ(status, 0);

  for (size_t k = 0; k < n; ++k)
    EXPECT_NE(b[k], u[k]);

  status = linsolver.solve(b, u);
  EXPECT_EQ(status, 0);

  for (size_t k = 0; k < n; ++k)
    EXPECT_EQ(b[k], u[k]);
}
#endif

TEST_F(LinearSolverTest, SPQRTest) {
  EXPECT_EQ(i.size(), j.size());
  EXPECT_EQ(i.size(), a.size());
  EXPECT_EQ(i.size(), 2 + 2 * (n - 2));

  linear_solver_spqr linsolver;

  int status;

  status = linsolver.setup(i, j, a, n);
  EXPECT_EQ(status, 0);

  for (size_t k = 0; k < n; ++k)
    EXPECT_NE(b[k], u[k]);

  status = linsolver.solve(b, u);
  EXPECT_EQ(status, 0);

  for (size_t k = 0; k < n; ++k)
    EXPECT_EQ(b[k], u[k]);
}

}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  enable_floating_point_exceptions();

  return RUN_ALL_TESTS();
}
