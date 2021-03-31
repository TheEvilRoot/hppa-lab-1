#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <iomanip>

#define eps 0.01
#define N 380
#define M 300
#define L 420
#define l 8
#define m 4
#define n 8

#include "util.h"

void mulAvx(float **** a, float **** b, float **** result) {
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                __m256 cols[m];
                for (int i = 0; i < m; i++) {
                    cols[i] = _mm256_load_ps(b[Y][Z][i]);
                }
                for (int x = 0; x < l; x++) {
                    __m256 rows[m];
                    for (int j = 0; j < m; j++) {
                        rows[j] = _mm256_broadcast_ss(&a[X][Y][x][j]);
                    }

                    for (int j = 0; j < m; j++) {
                        rows[j] = _mm256_mul_ps(cols[j], rows[j]);
                    }
                    
                    for (int j = 1; j < m; j++) {
                        rows[0] = _mm256_add_ps(rows[0], rows[j]);
                    }
                    __m256 res = _mm256_load_ps(result[X][Z][x]);
                    _mm256_store_ps(result[X][Z][x], _mm256_add_ps(rows[0], res));
                }
            }
        }
    }

}

void mul(float **** a, float **** b, float **** result) {
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                for (int x = 0; x < l; x++) {
                    for (int z = 0; z < n; z++) {
                        for (int i = 0; i < m; i++) {
                            result[X][Z][x][z] += a[X][Y][x][i] * b[Y][Z][i][z];
                        }
                    }
                }
            }
        }
    }
}

void mulNoVectorize(float **** a, float **** b, float **** resuly) {
#pragma clang loop vectorize(disable)
    for (int X = 0; X < L; X++) {
#pragma clang loop vectorize(disable)
        for (int Z = 0; Z < N; Z++) {
#pragma clang loop vectorize(disable)
            for (int Y = 0; Y < M; Y++) {
#pragma clang loop vectorize(disable)
                for (int x = 0; x < l; x++) {
#pragma clang loop vectorize(disable)
                    for (int z = 0; z < n; z++) {
#pragma clang loop vectorize(disable)
                        for (int i = 0; i < m; i++) {
                            resuly[X][Z][x][z] += a[X][Y][x][i] * b[Y][Z][i][z];
                        }
                    }
                }
            }
        }
    }
}


int main() {
    srand(time(nullptr) & 0xFFFFFFFF);
    
    fprintf(stderr, "allocating...\n");
    fprintf(stderr, "0 / 5");
    auto matrixA = newMatrix(L, M, l, m, true, true);
    fprintf(stderr, "\r1 / 5");
    auto matrixB = newMatrix(M, N, m, n, true, true);
    fprintf(stderr, "\r2 / 5");
    
    auto resultVectorized = newMatrix(L, N, l, n, false, false);
    fprintf(stderr, "\r3 / 5");
    
    auto resultNotVectorized = newMatrix(L, N, l, n, false, false);
    fprintf(stderr, "\r4 / 5");
    
    auto resultAvx = newMatrix(L, N, l, n, false, true);
    fprintf(stderr, "\r5 / 5\n");
    fprintf(stderr, "running 0 / 3");
    
    auto timeVec = measure<std::chrono::microseconds>(mul, matrixA, matrixB, resultVectorized);
    fprintf(stderr, "\rrunning 1 / 3");
    
    auto timeNoVec = measure<std::chrono::microseconds>(mulNoVectorize, matrixA, matrixB, resultNotVectorized);
    fprintf(stderr, "\rrunning 2 / 3");
    
    auto timeAvx = measure<std::chrono::microseconds>(mulAvx, matrixA, matrixB, resultAvx);
    fprintf(stderr, "\rrunning 3 / 3\n");
    
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "Vectorized: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeVec << "mcs\n";
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "Not Vectorized: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeNoVec << "mcs\n";
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "AVX: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeAvx << "mcs\n";
    
    bool check = matrixCompare(resultVectorized, resultNotVectorized);
    check = check && matrixCompare(resultNotVectorized, resultAvx);
    check = check && matrixCompare(resultVectorized, resultAvx);
    
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "Check: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << std::boolalpha << check << "\n";
    return 0;
}
