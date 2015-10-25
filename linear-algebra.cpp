#include <cassert>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <cholmod.h>

#include "abort_unless.h"
#include "linear-algebra.h"

namespace linear_algebra {

const size_t default_precision = 10;

vector::vector()
    : dim(0),
      values(dim, 0.0) {
}

vector::vector(size_t dim_)
    : dim(dim_),
      values(dim, 0.0) {
}

void vector::zeros() {
  std::fill(values.begin(), values.end(), 0.0);
}

void vector::ones() {
  std::fill(values.begin(), values.end(), 1.0);
}

double& vector::operator[](size_t k) {
  assert(k < dim);
  return values[k];
}

double vector::operator[](size_t k) const {
  assert(k < dim);
  return values[k];
}

int vector::save(const std::string& filename) const {
  std::ofstream ofs(filename);
  if (!ofs.is_open())
    return -1;

  ofs << std::setprecision(default_precision);

  for (auto x : values) {
    ofs << x << "\n";
    if (!ofs.good())
      return -1;
  }

  return 0;
}

int vector::load(const std::string& filename, std::vector<size_t>& product_indices) {
  std::ifstream ifs(filename);
  if (!ifs.is_open())
    return -1;

  values = std::vector<double>();

  size_t i = 0;
  while (ifs.good()) {
    double v;

    ifs >> v;
    ++i;

    if (ifs.eof())
      break;

    if (v == 0.0) {
      product_indices.push_back(i - 1);
    }

    values.push_back(v);
  }

  dim = values.size();

  return 0;
}

double vector::operator*(const vector& other) {
  assert(other.dim == dim);

  double result = 0.0;

  for (size_t i = 0; i < dim; ++i)
    result += (*this)[i] * other[i];

  return result;
}

double dot(const vector& u, const vector& v) {
  assert(u.dim == v.dim);

  double val = 0.0;
  for (size_t k = 0; k < u.dim; ++k)
    val += u[k] * v[k];
  return val;
}

matrix::matrix() : dim(0) {}

matrix::matrix(size_t dim_) : dim(dim_) {}

void matrix::zeros() {
  entries.clear();
}

int matrix::load(const std::string& filename, bool transpose) {
  std::ifstream ifs(filename);
  if (!ifs.is_open())
    return -1;

  int N;
  ifs >> N;
  if (N <= 0)
    return -1;

  dim = N;

  while (ifs.good()) {
    int i, j;
    double aij;

    ifs >> i >> j >> aij;

    if (ifs.eof())
      break;

    if (i <= 0 || j <= 0)
      return -1;

    if (aij == 0.0)
      continue;

    if (transpose)
      entries[j-1].push_back(std::make_pair(i-1, aij));
    else
      entries[i-1].push_back(std::make_pair(j-1, aij));
  }

  return 0;
}

int matrix::save(const std::string& filename) const {
  std::ofstream ofs(filename);
  if (!ofs.is_open())
    return -1;

  ofs << std::setprecision(default_precision);
  ofs << dim << "\n";

  for (const auto& entry : entries) {
    const size_t i = std::get<0>(entry);
    const auto& pairs = std::get<1>(entry);

    for (const auto& pair : pairs) {
      const size_t j = std::get<0>(pair);
      const double value = std::get<1>(pair);

      if (value == 0.0)
        continue;

      // Remember that the matrix is transposed in memory, so we have
      // to transpose again before writing to disk.
      ofs << (j+1) << " " << (i+1) << " " << value << "\n";
    }
  }

  return 0;
}

void matrix::set(size_t i, size_t j, double value) {
  entries[j].push_back(std::make_pair(i, value));
}

double matrix::get(size_t i, size_t j) const {
  auto row = entries.at(j);
  for (auto elem : row) {
    auto idx = std::get<0>(elem);
    if (idx == i)
      return std::get<1>(elem);
  }

  // abort_unless(false);

  return 0.0;
}

// void matrix::normalize_rows() {
//   for (auto& ij : entries) {
//     // XXX This is a little kludgy.
//     double total = 0.0;
// 
//     for (const auto& j_aij : ij.second) {
//       const double aij = j_aij.second;
//       total += aij;
//     }
// 
//     assert(total >= 0.0);
// 
//     if (total > 0.0) {
//       for (auto& j_aij : ij.second) {
//         double& aij = j_aij.second;
//         aij /= total;
//       }
//     }
//   }
// }

vector matrix::operator*(const vector& vec) const {
  assert(vec.size() == dim);
  assert(dim > 0);

  vector w(dim);

  matrix_vector_multiply(vec.memptr(), w.memptr());

  return w;
}

void matrix::matrix_vector_multiply(const double* input, double* output) const {
  assert(input != nullptr);
  assert(output != nullptr);
  assert(dim > 0);

  std::fill_n(output, dim, 0.0);

  for (const auto& ij : entries) {
    const int i = ij.first;

    for (const auto& j_aij : ij.second) {
      const int j = j_aij.first;
      const double aij = j_aij.second;

      output[i] += aij * input[j];
    }
  }
}

void matrix::subtract_identity() {
  // We use the fact that cholmod sums repeated triplets.
  for (size_t i = 0; i < dim; ++i) {
    auto& pairs = entries[i];
    pairs.push_back(std::make_pair(i, -1.0));
  }
}

std::tuple<std::vector<long>,
           std::vector<long>,
           std::vector<double>> matrix::as_triplets() const {
  std::vector<long> ii, jj;
  std::vector<double> values;

  // XXX DRY (see matrix::save()).

  for (const auto& entry : entries) {
    const size_t i = std::get<0>(entry);
    const auto& pairs = std::get<1>(entry);

    for (const auto& pair : pairs) {
      const size_t j = std::get<0>(pair);
      const double value = std::get<1>(pair);

      if (value == 0.0)
        continue;

      ii.push_back(i);
      jj.push_back(j);
      values.push_back(value);
    }
  }

  assert(ii.size() == jj.size() && jj.size() == values.size());

  return std::make_tuple(ii, jj, values);
}

}
