#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <iomanip>

#define eps 0.01
#define N 580
#define M 500
#define L 520
#define l 8
#define m 4
#define n 8

#include "util.h"


template<typename MA>
bool matrixCompare(MA a, MA b) {
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                for (int x = 0; x < l; x++) {
                    for (int z = 0; z < n; z++) {
                        if (!fcmp(a[X][Z][x][z], b[X][Z][x][z])) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

template<typename MA, typename MB, typename MR>
__declspec(noinline) void mul(MA matrixA, MB matrixB, MR resVec) {
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                for (int x = 0; x < l; x++) {
                    for (int z = 0; z < n; z++) {
                        resVec[X][Z][x][z] =
                            matrixA[X][Y][x][0] * matrixB[Y][Z][0][z] +
                            matrixA[X][Y][x][1] * matrixB[Y][Z][1][z] +
                            matrixA[X][Y][x][2] * matrixB[Y][Z][2][z] +
                            matrixA[X][Y][x][3] * matrixB[Y][Z][3][z];
                    }
                }
            }
        }
    }
}

template<typename MA, typename MB, typename MR>
__declspec(noinline) void mulNoVec(MA matrixA, MB matrixB, MR resNotVec) {
#pragma loop(no_vector)
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                for (int x = 0; x < l; x++) {
#pragma loop(no_vector)
                    for (int z = 0; z < n; z++) {
                        resNotVec[X][Z][x][z] =
                            matrixA[X][Y][x][0] * matrixB[Y][Z][0][z] +
                            matrixA[X][Y][x][1] * matrixB[Y][Z][1][z] +
                            matrixA[X][Y][x][2] * matrixB[Y][Z][2][z] +
                            matrixA[X][Y][x][3] * matrixB[Y][Z][3][z];
                    }
                }
            }
        }
    }
}

template<typename MA, typename MB, typename MR>
__declspec(noinline) void mulAvx(MA matrixA, MB matrixB, MR resAvx) {
    for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                __m256 cols[m];
                for (int i = 0; i < m; i++) {
                    cols[i] = _mm256_load_ps(matrixB[Y][Z][i]);
                }
                for (int x = 0; x < l; x++) {
                    __m256 rows[m];
                    for (int j = 0; j < m; j++) {
                        rows[j] = _mm256_broadcast_ss(&matrixA[X][Y][x][j]);
                    }

                    for (int j = 0; j < m; j++) {
                        rows[j] = _mm256_mul_ps(cols[j], rows[j]);
                    }

                    for (int j = 1; j < m; j++) {
                        rows[0] = _mm256_add_ps(rows[0], rows[j]);
                    }
                    __m256 res = _mm256_load_ps(resAvx[X][Z][x]);
                    _mm256_store_ps(resAvx[X][Z][x], _mm256_add_ps(rows[0], res));
                }
            }
        }
    }
}


int main() {
    srand(time(nullptr) & 0xFFFFFFFF);

    static __declspec(align(32)) float matrixA[L][M][l][m];
    static __declspec(align(32)) float matrixB[M][N][m][l];
    static __declspec(align(32)) float resVec[L][N][l][n];
    static __declspec(align(32)) float resNotVec[L][N][l][n];
    static __declspec(align(32)) float resAvx[L][N][l][n];

    //auto avx = [&]() {
    //    for (int X = 0; X < L; X++) {
    //        for (int Z = 0; Z < N; Z++) {
    //            for (int Y = 0; Y < M; Y++) {
    //                __m256 cols[m];
    //                for (int i = 0; i < m; i++) {
    //                    cols[i] = _mm256_load_ps(matrixB[Y][Z][i]);
    //                }
    //                for (int x = 0; x < l; x++) {
    //                    __m256 rows[m];
    //                    for (int j = 0; j < m; j++) {
    //                        rows[j] = _mm256_broadcast_ss(&matrixA[X][Y][x][j]);
    //                    }

    //                    for (int j = 0; j < m; j++) {
    //                        rows[j] = _mm256_mul_ps(cols[j], rows[j]);
    //                    }

    //                    for (int j = 1; j < m; j++) {
    //                        rows[0] = _mm256_add_ps(rows[0], rows[j]);
    //                    }
    //                    __m256 res = _mm256_load_ps(resAvx[X][Z][x]);
    //                    _mm256_store_ps(resAvx[X][Z][x], _mm256_add_ps(rows[0], res));
    //                }
    //            }
    //        }
    //    }
    //};

    //avx();

    mul(matrixA, matrixB, resVec);
    mulNoVec(matrixA, matrixB, resNotVec);
    mulAvx(matrixA, matrixB, resAvx);

    /*for (int X = 0; X < L; X++) {
        for (int Z = 0; Z < N; Z++) {
            for (int Y = 0; Y < M; Y++) {
                for (int x = 0; x < l; x++) {
                    for (int z = 0; z < n; z++) {
                        resVec[X][Z][x][z] = 
                            matrixA[X][Y][x][0] * matrixB[Y][Z][0][z] +
                            matrixA[X][Y][x][1] * matrixB[Y][Z][1][z] +
                            matrixA[X][Y][x][2] * matrixB[Y][Z][2][z] +
                            matrixA[X][Y][x][3] * matrixB[Y][Z][3][z];
                    }
                }
            }
        }
    }*/

//#pragma loop(no_vector)
//    for (int X = 0; X < L; X++) {
//        for (int Z = 0; Z < N; Z++) {
//            for (int Y = 0; Y < M; Y++) {
//                for (int x = 0; x < l; x++) {
//                    for (int z = 0; z < n; z++) {
//                        for (int i = 0; i < m; i++) {
//                            resNotVec[X][Z][x][z] += matrixA[X][Y][x][i] * matrixB[Y][Z][i][z];
//                        }
//                    }
//                }
//            }
//        }
//    }

 /*   std::cout << std::setw(20) << std::setfill(' ') << std::right << "Vectorized: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeVec << "mcs\n";
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "Not Vectorized: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeNoVec << "mcs\n";
    std::cout << std::setw(20) << std::setfill(' ') << std::right << "AVX: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << timeAvx << "mcs\n";*/

    bool check = matrixCompare(resVec, resNotVec);
    check = check && matrixCompare(resNotVec, resAvx);
    check = check && matrixCompare(resVec, resAvx);

    std::cout << std::setw(20) << std::setfill(' ') << std::right << "Check: ";
    std::cout << std::setw(6) << std::setfill(' ') << std::left << std::boolalpha << check << "\n";
    return 0;
}