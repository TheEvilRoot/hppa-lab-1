#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include <chrono>
#include <string>

#include <unistd.h>

#define aOuterCols 300
#define aOuterRows 360
#define bOuterRows 320

#define num_t float
#define dmatrix_t float**

namespace chrono = std::chrono;

std::string time_fmt(unsigned long time)
{
    unsigned long values[3] = {
        time, // mcs
        0,    // ms
        0,    // s
    };
    unsigned long mt = 1000;
    
    for (int current = 0; current < 2; current++)
    {
        if (values[current] < mt)
            break;
        values[current + 1] += values[current] / mt;
        values[current] %= mt;
    }
    
    std::string string;
    if (values[2] > 0)
        string += std::to_string(values[2]) + " seconds ";
    if (values[1] > 0)
        string += std::to_string(values[1]) + " milliseconds ";
    if (values[0] > 0)
        string += std::to_string(values[0]) + " microseconds ";
    return string;
}

template<typename duration, typename Ret, typename ... Aas>
double measure(Ret(*function)(Aas...), Aas ... args) {
    auto start = chrono::high_resolution_clock::now();
    function(args ...);
    auto end = std::chrono::high_resolution_clock::now();
    return chrono::duration_cast<duration>(end - start).count();
}

template<size_t innerACols, size_t innerARows, size_t innerBRows>
void matrix_mul(dmatrix_t mInA, dmatrix_t mInB, dmatrix_t mOut)
{
    // mInA(aOuterCols, aOuterRows)(innerACols, innerARows)
    // *
    // mInB(aOuterRows, bOuterRows)(innerARows, innerBRows)
    // =
    // mOut(aOuterCols, bOuterRows)(innerACols, innerBRows)
    
    constexpr auto oACols = aOuterCols;
    constexpr auto oARows = aOuterRows;
    constexpr auto iACols = innerACols;
    constexpr auto iARows = innerARows;
    
    constexpr auto oBCols = aOuterRows;
    constexpr auto oBRows = bOuterRows;
    constexpr auto iBCols = innerARows;
    constexpr auto iBRows = innerBRows;
    
    constexpr auto oOCols = aOuterCols;
    constexpr auto oORows = bOuterRows;
    constexpr auto iOCols = innerACols;
    constexpr auto iORows = innerBRows;
    
    // mInA[oACols][oARows][iACols][iARows]
    // mInB[oBCols][oBRows][iBCols][iBRows]
    // mOut[oOCols][oORows][iOCols][iORows]
    
    for (size_t m = 0; m < oACols; m++)
    {
        for (size_t r = 0; r < oBRows; r++)
        {
            for (size_t i = 0; i < iOCols * iORows; i++)
                mOut[m * oORows + r][i] = 0;
            
            for (size_t k = 0; k < oBCols; k++)
            {
                //  mOut[m][r] += mInA[m][k] + mInB[k][r]
                for (size_t M = 0; M < iACols; M++)
                {
                    for (size_t R = 0; R < iBRows; R++)
                    {
                        for (size_t K = 0; K < iBCols; K++)
                        {
                            mOut[m * oORows + r][M * iORows + R] =
                                mInA[m * oARows + k][M * iARows + K] +
                                mInB[k * oBRows + r][K * iBRows + R];
                        }
                    }
                }
            }
        }
    }
}

template<size_t innerACols, size_t innerARows, size_t innerBRows>
void matrix_mul_avx(dmatrix_t mInA, dmatrix_t mInB, dmatrix_t mOut)
{
    // mInA(aOuterCols, aOuterRows)(innerACols, innerARows)
    // *
    // mInB(aOuterRows, bOuterRows)(innerARows, innerBRows)
    // =
    // mOut(aOuterCols, bOuterRows)(innerACols, innerBRows)
    
    constexpr auto oACols = aOuterCols;
    constexpr auto oARows = aOuterRows;
    constexpr auto iACols = innerACols;
    constexpr auto iARows = innerARows;
    
    constexpr auto oBCols = aOuterRows;
    constexpr auto oBRows = bOuterRows;
    constexpr auto iBCols = innerARows;
    constexpr auto iBRows = innerBRows;
    
    constexpr auto oOCols = aOuterCols;
    constexpr auto oORows = bOuterRows;
    constexpr auto iOCols = innerACols;
    constexpr auto iORows = innerBRows;
    
    // mInA[oACols][oARows][iACols][iARows]
    // mInB[oBCols][oBRows][iBCols][iBRows]
    // mOut[oOCols][oORows][iOCols][iORows]
    
    for (size_t m = 0; m < oACols; m++)
    {
        for (size_t r = 0; r < oBRows; r++)
        {
            for (size_t i = 0; i < iOCols * iORows; i++)
                mOut[m * oORows + r][i] = 0;
            
            for (size_t k = 0; k < oBCols; k++)
            {
                //  mOut[m][r] += mInA[m][k] + mInB[k][r]
                for (size_t M = 0; M < iACols; M++)
                {
                    for (size_t R = 0; R < iBRows; R++)
                    {
                        for (size_t K = 0; K < iBCols; K++)
                        {
                            mOut[m * oORows + r][M * iORows + R] =
                                mInA[m * oARows + k][M * iARows + K] +
                                mInB[k * oBRows + r][K * iBRows + R];
                        }
                    }
                }
            }
        }
    }
}


template<size_t outerCols, size_t outerRows, size_t innerCols, size_t innerRows>
dmatrix_t alloc_matrix() {
    dmatrix_t matrix = new num_t * [outerCols * outerRows];
    for (int i = 0; i < outerRows * outerCols; ++i)
    {
        matrix[i] = new num_t[innerCols * innerRows];
        for (int j = 0; j < innerCols * innerRows; ++j)
        {
            matrix[i][j] = random() % 10;
        }
    }
    return matrix;
}

void just_mul(dmatrix_t a, dmatrix_t b, dmatrix_t r) {
    matrix_mul<10, 12, 14>(a, b, r);
}

void avx_mul(dmatrix_t a, dmatrix_t b, dmatrix_t r) {
    matrix_mul_avx<10, 12, 14>(a, b, r);
}

int main()
{
    dmatrix_t a = alloc_matrix<aOuterCols, aOuterRows, 10, 12>();
    dmatrix_t b = alloc_matrix<aOuterRows, bOuterRows, 12, 14>();
    dmatrix_t r = alloc_matrix<aOuterCols, bOuterRows, 10, 14>();
    
    printf("no avx: %s\n", time_fmt(measure<chrono::microseconds>(just_mul, a, b, r)).c_str());
    printf("   avx: %s\n", time_fmt(measure<chrono::microseconds>(avx_mul, a, b, r)).c_str());
    
}
