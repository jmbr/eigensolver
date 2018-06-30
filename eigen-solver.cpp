#include <cmath>
#include <cstring>

#include <map>
#include <limits>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include <SuiteSparseQR.hpp>

#include "linear-algebra.h"
#include "linear-solver-umfpack.h"
#include "linear-solver-spqr.h"
#include "eigen-solver.h"
#include "arnoldi.h"

#include "abort_unless.h"
#include "timer.h"
#include "misc.h"

static double normalize(const double* v, size_t n, double* x);

const std::map<int, std::string> dnaupd_messages = {
  { 0, "Normal exit." },
  { 1, "Maximum number of iterations taken. "
    "All possible eigenvalues of OP has been found. "
    "IPARAM(5) returns the number of wanted converged Ritz values." },
  { 2, "No longer an informational error. "
    "Deprecated starting with release 2 of ARPACK." },
  { 3, "No shifts could be applied during a cycle of the Implicitly restarted "
    "Arnoldi iteration. "
    "One possibility is to increase the size of NCV relative to NEV. See "
    "remark 4 below." },
  { -1, "N must be positive." },
  { -2, "NEV must be positive." },
  { -3, "NCV-NEV >= 2 and less than or equal to N." },
  { -4, "The maximum number of Arnoldi update iteration must be greater than "
    "zero." },
  { -5, "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'" },
  { -6, "BMAT must be one of 'I' or 'G'." },
  { -7, "Length of private work array is not sufficient." },
  { -8, "Error return from LAPACK eigenvalue calculation;" },
  { -9, "Starting vector is zero." },
  { -10, "IPARAM(7) must be 1,2,3,4." },
  { -11, "IPARAM(7) = 1 and BMAT = 'G' are incompatible." },
  { -12, "IPARAM(1) must be equal to 0 or 1." },
  { -9999, "Could not build an Arnoldi factorization. "
   "IPARAM(5) returns the size of the current Arnoldi factorization." }
};

const std::map<int, std::string> dneupd_messages = {
  { 0, "Normal exit." },
  { 1, "The Schur form computed by LAPACK routine dlahqr could not be "
    "reordered by LAPACK routine dtrsen. Re-enter subroutine dneupd with "
    "IPARAM(5)=NCV and increase the size of the arrays DR and DI to have "
    "dimension at least dimension NCV and allocate at least NCV columns for Z. "
    "NOTE, \"Not necessary if Z and V share the same space. "
    "Please notify the authors if this error occurs.\"" },
  { -1, "N must be positive." },
  { -2, "NEV must be positive."},
  { -3, "NCV-NEV >= 2 and less than or equal to N." },
  { -5, "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'" },
  { -6, "BMAT must be one of 'I' or 'G'." },
  { -7, "Length of private work WORKL array is not sufficient." },
  { -8, "Error return from calculation of a real Schur form. "
    "Informational error from LAPACK routine dlahqr." },
  { -9, "Error return from calculation of eigenvectors. "
    "Informational error from LAPACK routine dtrevc." },
  { -10, "IPARAM(7) must be 1,2,3,4." },
  { -11, "IPARAM(7) = 1 and BMAT = 'G' are incompatible." },
  { -12, "HOWMNY = 'S' not yet implemented." },
  { -13, "HOWMNY must be one of 'A' or 'P' if RVEC = .true." },
  { -14, "DNAUPD did not find any eigenvalues to sufficient accuracy." },
  { -15, "DNEUPD got a different count of the number of converged Ritz values "
    "than DNAUPD got. This indicates the user probably made an error in "
    "passing data from DNAUPD to DNEUPD or that the data was modified before "
    "entering DNEUPD." }
};

eigen_solver::eigen_solver(size_t max_iterations_, size_t max_basis_vectors_)
    : max_iterations(max_iterations_), max_basis_vectors(max_basis_vectors_) {
}

int eigen_solver::solve(linear_algebra::matrix const& A,
                        linear_algebra::vector& x0,
                        double* ritz_value) const noexcept {
  int n = A.size();
  int maxn = n;  // Maximum problem size.
  int nev = 10;   // Number of eigenvalues requested.
  // int maxnev = nev; // Maximum NEV allowed.
  int ncv = max_basis_vectors;
  int maxncv = ncv;  // Maximum NCV allowed
  int ldv = maxn;

  double tol = epsilon;  // Threshold for relative error.

  std::vector<double> d(maxncv * 3, 0.0);
  double* resid = x0.memptr();

  std::vector<double> v(ldv * maxncv, 0.0);
  std::vector<double> workd(3 * maxn, 0.0);
  std::vector<double> workev(3 * maxncv, 0.0);
  std::vector<double> workl(3 * maxncv * maxncv + 6 * maxncv, 0.0);

  std::vector<int> iparam(11, 0);
  std::vector<int> ipntr(14, 0);
  std::vector<int> select(maxncv, 0);

  char bmat[] = "I";   // Standard (i.e., not generalized)
                       // eigenproblem.
  char which[] = "LM"; // Largest in magnitude.

  int ido = 0;  // Reverse communication parameter
  int lworkl = workl.size();
  int info = 1;  // We start the iteration using our own vector.

  const int ishfts = 1;     // Exact shifts.
  const int maxitr = max_iterations;
  const int mode = 3;       // Shift and invert.

  iparam[0] = ishfts;
  iparam[2] = maxitr;
  iparam[6] = mode;

  int ierr;
  int rvec = 1;
  char howmny = 'A';
  double sigmar = 1.0, sigmai = 0.0;

  linear_solver_spqr linsolver;
  // linear_solver_umfpack linsolver;
  {
    linear_algebra::matrix B = A;
    B.subtract_identity();
    auto triplets = B.as_triplets();
    linsolver.setup(std::get<0>(triplets),
                    std::get<1>(triplets),
                    std::get<2>(triplets),
                    B.size());
  }

  std::clog << "Running Arnoldi iteration with tolerance " << tol << "...\n";

  timer clock;
  clock.tic();

  int itr;
  for (itr = 0; itr < maxitr; ++itr) {
    dnaupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, &v[0], &ldv,
            &iparam[0], &ipntr[0], &workd[0], &workl[0], &lworkl, &info);

    if (info != 0) {
      const auto message = dnaupd_messages.at(info);
      std::cerr << "Error after call to dnaupd: \"" << message << "\"\n";
      return -1;
    }

    if (ido == 99) {
      break;
    } else if (abs(ido) != 1) {
      std::cerr << "Warning: dnaupd reported IDO = " << ido << "\n";
      break;
    }

    auto rhs = &workd[ipntr[0] - 1];
    auto sol = &workd[ipntr[1] - 1];
    if (linsolver.solve(rhs, sol) != 0) {
      std::cerr << "Warning: Failed to solve linear system.\n";
      break;  // XXX Throw exception.
    }
  }

  clock.toc();

  std::clog << "Done with Arnoldi iteration after " << itr
            << " iterations (" << clock.seconds() << " seconds).\n"
            << "Running postprocessing step...\n";

  clock.tic();

  dneupd_(&rvec, &howmny, &select[0], &d[0], &d[maxncv], &v[0], &ldv, &sigmar,
          &sigmai, &workev[0], bmat, &n, which, &nev, &tol, resid, &ncv, &v[0], &ldv,
          &iparam[0], &ipntr[0], &workd[0], &workl[0], &lworkl, &ierr);

  clock.toc();

  int nconv = iparam[4];

  std::clog << "Done with postprocessing step. Status: "
            << (ierr == 0 ? "OK" : "FAIL")
            << " (" << clock.seconds() << " seconds)." << "\n"
            << "Number of converged Ritz values: " << nconv << "\n"
            << "Number of implicit Arnoldi updates: " << iparam[2] << "\n"
            << "Number of linear solutions computed: " << iparam[8]
            << std::endl;

  if (ierr != 0) {
    const auto message = dneupd_messages.at(ierr);
    std::cerr << "Error after call to dneupd: \"" << message << "\""
              << std::endl;
    return -1;
  }

  normalize(&v[0], n, &x0[0]);

  // Save the largest Ritz value (= best approximation to the largest
  // eigenvalue).
  std::clog << "Largest Ritz value: " << d[0] << std::endl;
  if (ritz_value != nullptr)
    *ritz_value = d[0];

  return 0;
}

double normalize(const double* v, size_t n, double* x) {
  double sum = 0.0, nrm = 0.0;

  for (size_t i = 0; i < n; ++i) {
    x[i] = v[i];
    sum += x[i];
    nrm += std::fabs(x[i]);
  }

  const double sign = std::signbit(sum) == 1 ? -1.0 : 1.0;

  for (size_t i = 0; i < n; ++i)
    x[i] /= (nrm * sign);

  return nrm;
}
