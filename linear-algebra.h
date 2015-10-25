#pragma once

#include <sys/types.h>

#include <map>
#include <vector>
#include <string>

namespace linear_algebra {

/**
 * Dense vector class.
 */
class vector {
 public:
  vector();
  vector(size_t dim);

  void zeros();
  void ones();

  double& operator[](size_t k);
  double operator[](size_t k) const;

  int load(const std::string& filename, std::vector<size_t>& product_indices);
  int save(const std::string& filename) const;

  inline size_t size() const { return dim; }

  const double* memptr() const { return &values[0]; }
  double* memptr() { return &values[0]; }

  double operator*(const vector& other);

  friend double dot(const vector& u, const vector& v);

 private:
  size_t dim;
  std::vector<double> values;
};

double dot(const vector& u, const vector& v);

/**
 * Sparse square matrix class.
 */
class matrix {
 public:
  typedef std::vector<std::pair<size_t, double>> row;

  matrix();
  matrix(size_t dim);

  void zeros();

  int load(const std::string& filename, bool transpose = true);
  int save(const std::string& filename) const;

  inline size_t size() const { return dim; }

  void set(size_t i, size_t j, double value);
  double get(size_t i, size_t j) const;

  row& get_row(size_t i) { return entries.at(i); }

  // void normalize_rows();

  std::tuple<std::vector<long>, std::vector<long>, std::vector<double>> as_triplets() const;

  void matrix_vector_multiply(double const* input, double* output) const;
  vector operator*(const vector& vec) const;

  void subtract_identity();

 private:
  size_t dim;
  std::map<size_t, row> entries;
};

}
