#include <cassert>

#include <vector>
#include <iostream>

#include "linear-solver-spqr.h"
#include "abort_unless.h"
#include "misc.h"

using namespace std;

linear_solver_spqr::linear_solver_spqr() : mat(nullptr) {
  cholmod_l_start(&common);
}

linear_solver_spqr::~linear_solver_spqr() {
  if (mat != nullptr)
    cholmod_l_free_sparse(&mat, &common);

  cholmod_l_finish(&common);

  if (QR != nullptr)
    SuiteSparseQR_free<double>(&QR, &common);
}

int linear_solver_spqr::setup(const vector<long>& ii, const vector<long>& jj,
                              const vector<double>& A, size_t n) {
  abort_unless(ii.size() == jj.size() && jj.size() == A.size());

  size_t max_nonzeros = ii.size();

  cholmod_triplet *T = cholmod_l_allocate_triplet(n, n, max_nonzeros, 0,
                                                  CHOLMOD_REAL, &common);
  if (T == nullptr)
    return -1;

  auto i = reinterpret_cast<long*>(T->i);
  auto j = reinterpret_cast<long*>(T->j);
  auto v = reinterpret_cast<double*>(T->x);

  for (size_t k = 0; k < ii.size(); ++k) {
    i[k] = ii.at(k);
    j[k] = jj.at(k);
    v[k] =  A.at(k);
  }

  T->nnz = ii.size();

  mat = cholmod_l_triplet_to_sparse(T, T->nnz, &common);
  cholmod_l_free_triplet(&T, &common);
  if (mat == nullptr)
    return -1;

  QR = SuiteSparseQR_factorize<double>(SPQR_ORDERING_DEFAULT, 0.0, mat, &common);
  if (QR == nullptr)
    return -1;

  return 0;
}

int linear_solver_spqr::solve(const std::vector<double>& b, std::vector<double>& u) {
  assert(mat != nullptr);
  assert(b.size() == mat->ncol);
  assert(u.size() == mat->nrow);

  return solve(&b[0], &u[0]);
}

int linear_solver_spqr::solve(const double* b, double* u) {
  assert(b != nullptr);
  assert(u != nullptr);

  cholmod_dense* rhs;
  rhs = cholmod_l_zeros(mat->ncol, 1, CHOLMOD_REAL, &common);
  if (rhs == nullptr)
    return -1;

  std::copy(b, b + mat->ncol, reinterpret_cast<double*>(rhs->x));

  // cholmod_dense* sol = SuiteSparseQR<double>(mat, rhs, &common);
  cholmod_dense* aux;
  aux = SuiteSparseQR_qmult<double>(SPQR_QTX, QR, rhs, &common);
  if (aux == nullptr) {
    cholmod_l_free_dense(&rhs, &common);
    return -1;
  }

  cholmod_dense* sol = SuiteSparseQR_solve<double>(SPQR_RETX_EQUALS_B, QR, aux, &common);
  if (sol == nullptr) {
    cholmod_l_free_dense(&aux, &common);
    cholmod_l_free_dense(&rhs, &common);
    return -1;
  }

  const double* values = reinterpret_cast<const double*>(sol->x);
  std::copy(values, values + sol->nrow, u);

  cholmod_l_free_dense(&aux, &common);
  cholmod_l_free_dense(&rhs, &common);
  cholmod_l_free_dense(&sol, &common);
  return 0;
}
