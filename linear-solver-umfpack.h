#pragma once

#include <vector>

#include <umfpack.h>

class linear_solver_umfpack {
 public:
  linear_solver_umfpack();
  ~linear_solver_umfpack();

  int setup(const std::vector<long>& ii, const std::vector<long>& jj,
            const std::vector<double>& A, size_t dim);

  int solve(const std::vector<double>& b, std::vector<double>& u);

  int solve(const double* right_hand_side, double* solution);
  
 private:
  double info[UMFPACK_INFO];
  double control[UMFPACK_CONTROL];

  void* symbolic;
  void* numeric;

  std::vector<long> Ap;
  std::vector<long> Ai;
  std::vector<double> Ax;
};
