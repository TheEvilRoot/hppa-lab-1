//
//  util.h
//  mmul
//
//  Created by TheEvilRoot on 31.03.21.
//

#ifndef util_h
#define util_h

#include <chrono>

float maxf(float a, float b) {
    return a > b ? a : b;
}

bool fcmp(float a, float b) {
    return fabs(a - b) <= maxf(a, b) * 0.01;
}

float frandom() {
    return (rand() % 100) + (rand() % 100) / 100.0;
}

auto newMatrix(int R, int C, int r, int c, bool randomized, bool aligned) {
    auto matrix = new float*** [R];
    for (int i = 0; i < R; i++) {
        matrix[i] = new  float** [C];
        for (int j = 0; j < C; j++) {
            matrix[i][j] = new  float* [r];
            for (int k = 0; k < r; k++) {
                matrix[i][j][k] = aligned ? (new float[c]) : (new float[c]);
                for (int x = 0; x < c; x++) {
                    matrix[i][j][k][x] = randomized ? frandom() : 0;
                }
            }
        }
    }
    return matrix;
}


template<typename duration, typename Ret, typename ... Aas>
double measure(Ret(*function)(Aas...), Aas... args) {
    auto start = std::chrono::high_resolution_clock::now();
    function(args...);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<duration>(end - start).count();

}

#endif /* util_h */