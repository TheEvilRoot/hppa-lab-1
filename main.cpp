#include <ctime>
#include <immintrin.h>

#ifdef SELF_MEASURE
#include <iostream>
#include <iomanip>
#endif

#include "utils.hpp"

constexpr size_t N = 700;
constexpr size_t M = 700;
constexpr size_t L = 750;
constexpr size_t l = 8;
constexpr size_t m = 4;
constexpr size_t n = 8;

template<typename MA, typename MB, typename MR>
void mulAvx(MA matrixA, MB matrixB, MR result) {
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < M; k++) {

        // LOADING COLUMNS TO MULTIPLY WITH ROWS
        // 0: | float1  | float2  | float3  | ...   | float8  |
        // 1: | float9  | ...                       | float16 |
        __m256 cols[m];
        cols[0] = _mm256_load_ps(matrixB[k][j][0]);
        cols[1] = _mm256_load_ps(matrixB[k][j][1]);
        cols[2] = _mm256_load_ps(matrixB[k][j][2]);
        cols[3] = _mm256_load_ps(matrixB[k][j][3]);

        for (int x = 0; x < l; x++) {

          // LOAD ROW AND SET IT TO ALL ELEMENTS OF THE VECTOR
          // 0: | float1 | float 1 | float1 | ... | float1 |
          // 1: | float2 | float 2 | float2 | ... | float2 |
          __m256 rows[m];
          rows[0] = _mm256_broadcast_ss(&matrixA[i][k][x][0]);
          rows[1] = _mm256_broadcast_ss(&matrixA[i][k][x][1]);
          rows[2] = _mm256_broadcast_ss(&matrixA[i][k][x][2]);
          rows[3] = _mm256_broadcast_ss(&matrixA[i][k][x][3]);

          // MARK: - DO NOT MODIFY COLS[], CUZ WE'RE MULTIPLYING
          //         SEVERAL TIMES OVER THEM

          rows[0] = _mm256_mul_ps(cols[0], rows[0]);
          rows[1] = _mm256_mul_ps(cols[1], rows[1]);
          rows[2] = _mm256_mul_ps(cols[2], rows[2]);
          rows[3] = _mm256_mul_ps(cols[3], rows[3]);

          // R1 * C1 + R2 * C2 ...
          rows[0] = _mm256_add_ps(rows[0], rows[1]);
          rows[0] = _mm256_add_ps(rows[0], rows[2]);
          rows[0] = _mm256_add_ps(rows[0], rows[3]);

          // M += <R1 * C1 + R2 * C2 ...>
          __m256 res = _mm256_load_ps(result[i][j][x]);
          res = _mm256_add_ps(rows[0], res);
          _mm256_store_ps(result[i][j][x], res);
        }
      }
    }
  }

}

template<typename MA, typename MB, typename MR>
void mul(MA matrixA, MB matrixB, MR matrixC1) {
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < M; k++) {
        for (int x = 0; x < l; x++) {
          for (int y = 0; y < n; y++) {
            matrixC1[i][j][x][y] += matrixA[i][k][x][0] * matrixB[k][j][0][y] +
              matrixA[i][k][x][1] * matrixB[k][j][1][y] +
              matrixA[i][k][x][2] * matrixB[k][j][2][y] +
              matrixA[i][k][x][3] * matrixB[k][j][3][y];
          }
        }
      }
    }
  }
}

template<typename MA, typename MB, typename MR>
void mulNoVector(MA matrixA, MB matrixB, MR matrixC2) {
#pragma loop(no_vector)
  for (int i = 0; i < L; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < M; k++) {
        for (int x = 0; x < l; x++) {
#pragma loop(no_vector)
          for (int y = 0; y < n; y++) {
            matrixC2[i][j][x][y] += matrixA[i][k][x][0] * matrixB[k][j][0][y] +
              matrixA[i][k][x][1] * matrixB[k][j][1][y] +
              matrixA[i][k][x][2] * matrixB[k][j][2][y] +
              matrixA[i][k][x][3] * matrixB[k][j][3][y];
          }
        }
      }
    }
  }
}

int main() {
  srand(2);

#ifdef SELF_MEASURE
  std::cout << "Allocating...\n";
#endif

  static __declspec(align(32)) float matrixA[L][M][l][m];
  static __declspec(align(32)) float matrixB[M][N][m][n];
  static __declspec(align(32)) float resultVectorized[L][N][l][n];
  static __declspec(align(32)) float resultNotVectorized[L][N][l][n];
  static __declspec(align(32)) float resultAvx[L][N][l][n];

#ifdef SELF_MEASURE
  std::cout << "Filling up...\n";
#endif

  fillMatrix<float, L, M, l, m>(matrixA, true);
  fillMatrix<float, M, N, m, n>(matrixB, true);
  fillMatrix<float, L, N, l, n>(resultVectorized);
  fillMatrix<float, L, N, l, n>(resultNotVectorized);
  fillMatrix<float, L, N, l, n>(resultAvx);

#ifdef SELF_MEASURE
  std::cout << "Calculating...\n";
#endif

  MEASURE(timeVectorized, mul(matrixA, matrixB, resultVectorized));
  MEASURE(timeNotVectorized, mulNoVector(matrixA, matrixB, resultNotVectorized));
  MEASURE(timeAvx, mulAvx(matrixA, matrixB, resultAvx));

#ifdef SELF_MEASURE
  constexpr auto leftPad = 16;
  constexpr auto rightPad = 6;
  std::cout << "Checking results...\n";

  auto doMatch = matrixCompare<float, L, N, l, n>(resultVectorized, resultNotVectorized);
  doMatch &= matrixCompare<float, L, N, l, n>(resultNotVectorized, resultAvx);
  doMatch &= matrixCompare<float, L, N, l, n>(resultVectorized, resultAvx);

  std::cout << "\n";
  std::cout << std::setw(leftPad) << std::setfill(' ') << std::right << "Not Vectorized: ";
  std::cout << std::setw(rightPad) << std::setfill(' ') << std::left << timeformat(timeNotVectorized) << "\n";
  std::cout << std::setw(leftPad) << std::setfill(' ') << std::right << "Vectorized: ";
  std::cout << std::setw(rightPad) << std::setfill(' ') << std::left << timeformat(timeVectorized) << "\n";
  std::cout << std::setw(leftPad) << std::setfill(' ') << std::right << "AVX: ";
  std::cout << std::setw(rightPad) << std::setfill(' ') << std::left << timeformat(timeAvx) << "\n";

  std::cout << std::setw(leftPad) << std::setfill(' ') << std::right << "Match: ";
  std::cout << std::setw(rightPad) << std::setfill(' ') << std::left << std::boolalpha << doMatch << "\n";

  if (!doMatch)
    std::cout << "Checksum: " << std::boolalpha << matrixCompare<float, L, N, l, n>(resultVectorized, resultNotVectorized)
    << matrixCompare<float, L, N, l, n>(resultNotVectorized, resultAvx)
    << matrixCompare<float, L, N, l, n>(resultVectorized, resultAvx) << "\n";
#endif
}