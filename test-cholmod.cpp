#include <cstdlib>

#include <iostream>

#include <SuiteSparseQR.hpp>

#include <gtest/gtest.h>

const double tolerance = 1e-15;

TEST(Cholmod, CholmodGeneral) {
  cholmod_common common;
  cholmod_sparse *A;
  cholmod_dense *x, *b, *residual;
  double residual_norm;
  double one[] = { 1, 0 }, minusone[] = { -1, 0 };

  cholmod_l_start(&common);

  {
    const size_t n = 1000;
    const size_t nnz = 2 + 2 * (n - 2);
    cholmod_triplet *T = cholmod_l_allocate_triplet(n, n, nnz, 0, CHOLMOD_REAL, &common);
    ASSERT_FALSE(T == nullptr);
    // if (T == nullptr) {
    //   perror("cholmod_l_allocate_triplet");
    //   return EXIT_FAILURE;
    // }

    size_t k = 0;
    long* i = (long*) T->i;
    long* j = (long*) T->j;
    double* v = (double*) T->x;

    i[k] = 0; j[k] = 1; v[k] = 1.0; ++k;
    for (size_t row = 1; row < n-1; ++row) {
      i[k] = row; j[k] = row - 1; v[k] = 0.5; ++k;
      i[k] = row; j[k] = row + 1; v[k] = 0.5; ++k;
    }
    i[k] = n - 1; j[k] = n - 2; v[k] = 1.0; ++k;

    T->nnz = k;

    // cholmod_l_print_triplet(T, "triplet", &common);

    A = cholmod_l_triplet_to_sparse(T, T->nnz, &common);
    ASSERT_FALSE(A == nullptr);
    // if (A == nullptr) {
    //   perror("cholmod_l_triplet_to_sparse");
    //   return EXIT_FAILURE;
    // }

    // cholmod_l_print_sparse(A, "sparse", &common);
    cholmod_l_free_triplet(&T, &common);
  }

  b = cholmod_l_ones(A->nrow, 1, A->xtype, &common);

  x = SuiteSparseQR<double>(A, b, &common);

  residual = cholmod_l_copy_dense(b, &common);
  cholmod_l_sdmult(A, 0, minusone, one, x, residual, &common);
  residual_norm = cholmod_l_norm_dense(residual, 2, &common) ;

  // std::cout << "|| A x - b ||_2 = " << residual_norm << "\n";

  cholmod_l_free_dense(&residual, &common);
  cholmod_l_free_sparse(&A, &common);
  cholmod_l_free_dense(&x, &common);
  cholmod_l_free_dense(&b, &common);
  cholmod_l_finish(&common);

  ASSERT_NEAR(residual_norm, 0.0, tolerance);

  // if (residual_norm >= tolerance)
  //   return EXIT_FAILURE;
  //
  // return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
