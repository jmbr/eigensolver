#pragma once

#include <chrono>

class timer {
 public:
  void tic() {
    start = std::chrono::high_resolution_clock::now();
  }

  void toc() {
    stop = std::chrono::high_resolution_clock::now();
  }

  double seconds() {
    std::chrono::duration<double> elapsed = stop - start;
    return elapsed.count();
  }

 private:
  std::chrono::system_clock::time_point start, stop;
};
