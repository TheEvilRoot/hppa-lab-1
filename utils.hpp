#ifndef utils_h
#define utils_h

#include <chrono>

#define MEASURE(VAR_NAME, FUNCTION_CALL) auto TIME_##VAR_NAME = std::chrono::high_resolution_clock::now();\
FUNCTION_CALL;\
auto VAR_NAME = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - TIME_##VAR_NAME)

inline float maxf(float a, float b) {
  return a > b ? a : b;
}

inline bool fcmp(float a, float b, float eps = 0.001) {
  return fabs(a - b) <= maxf(a, b) * eps;
}

inline float frandom() {
  return ((rand() % 10) * 1.0f);
}

auto timeformat(std::chrono::microseconds mcs) {
  long long milliseconds = (mcs.count()) / 1000;
  long long seconds = milliseconds / 1000;
  if (seconds > 0)
    return std::to_string(seconds) + " seconds " + std::to_string(milliseconds % 1000) + " milliseconds";
  if (milliseconds > 0)
    return std::to_string(milliseconds) + " milliseconds " + std::to_string(mcs.count() % 1000) + " microseconds";
  return std::to_string(mcs.count()) + " microseconds";
}

template<typename T, size_t R, size_t C, size_t r, size_t c>
bool matrixCompare(T a[R][C][r][c], T b[R][C][r][c]) {
  for (auto i = 0; i < R; i++) {
    for (auto j = 0; j < C; j++) {
      for (auto k = 0; k < r; k++) {
        for (auto h = 0; h < c; h++) {
          if (!fcmp(a[i][j][k][h], b[i][j][k][h]))
            return false;
        }
      }
    }
  }
  return true;
}

template<typename T, size_t R, size_t C, size_t r, size_t c>
void fillMatrix(T matrix[R][C][r][c], bool doRandomized = false) {
  for (int i = 0; i < R; i++) {
    for (int j = 0; j < C; j++) {
      for (int k = 0; k < r; k++) {
        for (int h = 0; h < c; h++) {
          matrix[i][j][k][h] = (doRandomized ? frandom() : 0.0);
        }
      }
    }
  }
}

#endif