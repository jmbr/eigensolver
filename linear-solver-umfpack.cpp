#include <cassert>

#include <vector>
#include <iostream>

#include <umfpack.h>

#include "linear-solver-umfpack.h"

using std::vector;

linear_solver_umfpack::linear_solver_umfpack() : symbolic(nullptr), numeric(nullptr) {
  std::fill_n(info, UMFPACK_INFO, 0);
  std::fill_n(control, UMFPACK_CONTROL, 0);
}

linear_solver_umfpack::~linear_solver_umfpack() {
  if (symbolic != nullptr) umfpack_dl_free_symbolic(&symbolic);
  if (numeric != nullptr) umfpack_dl_free_numeric(&numeric);
}

int linear_solver_umfpack::setup(const vector<long>& ii, const vector<long>& jj,
                                 const vector<double>& A, size_t n) {
  const size_t nz = ii.size();
  assert(ii.size() == jj.size());
  assert(jj.size() == A.size());

  Ap = vector<long>(n + 1);
  Ai = vector<long>(nz);
  Ax = vector<double>(nz);

  int status;

  status = umfpack_dl_triplet_to_col(n, n, nz, &ii[0], &jj[0], &A[0], &Ap[0], &Ai[0], &Ax[0], nullptr);
  if (status < 0) {
    umfpack_dl_report_status(control, status) ;
    std::cerr << "umfpack_dl_triplet_to_col failed\n";
    return -1;
  }

  // clog << "Preparing to solve sparse linear system." << endl;

  status = umfpack_dl_symbolic(n, n, &Ap[0], &Ai[0], &Ax[0], &symbolic, control, info);
  if (status < 0) {
    umfpack_dl_report_info(control, info) ;
    umfpack_dl_report_status(control, status) ;
    std::cerr << "umfpack_dl_symbolic failed\n";
    return -1;
  }

  status = umfpack_dl_numeric(&Ap[0], &Ai[0], &Ax[0], symbolic, &numeric, control, info);
  if (status < 0) {
    umfpack_dl_report_info(control, info);
    umfpack_dl_report_status(control, status);
    std::cerr << "umfpack_dl_numeric failed\n";
    return -1;
  }

  return 0;
}

int linear_solver_umfpack::solve(const std::vector<double>& b, std::vector<double>& u) {
  return solve(&b[0], &u[0]);
}

int linear_solver_umfpack::solve(const double* b, double* u) {
  int status;

  status = umfpack_dl_solve (UMFPACK_A, &Ap[0], &Ai[0], &Ax[0], u, b, numeric, control, info) ;
  umfpack_dl_report_info(control, info) ;
  umfpack_dl_report_status(control, status) ;
  if (status < 0) {
    std::cerr << __func__ << ": umfpack_dl_solve failed\n";
    return -1;
  }

  return 0;
}
