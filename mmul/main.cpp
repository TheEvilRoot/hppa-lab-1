#include <ctime>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

/*#define N 300
#define M 400
#define L 350
#define l 4
#define m 8
#define n 8*/
#define N 300
#define M 400
#define L 350
#define l 4
#define m 8
#define n 8

float matrixA[L][M][l][m];
float matrixB[M][N][m][n];
float matrixC1[L][N][l][n];
float matrixC2[L][N][l][n];
float matrixC3[L][N][l][n];



void equalMatrix() {
	bool b1 = true, b2 = true, b3 = true;
	for (int X = 0; X < L; X++) {
		for (int Z = 0; Z < N; Z++) {
			for (int Y = 0; Y < M; Y++) {
				for (int x = 0; x < l; x++) {
					for (int z = 0; z < n; z++) {
						if (abs(matrixC1[X][Z][x][z] - matrixC2[X][Z][x][z]) > matrixC1[X][Z][x][z]*0.01) b1 = false;
						if (abs(matrixC2[X][Z][x][z] - matrixC3[X][Z][x][z]) > matrixC2[X][Z][x][z]*0.01) b2 = false;
						if (abs(matrixC1[X][Z][x][z] - matrixC3[X][Z][x][z]) > matrixC1[X][Z][x][z]*0.01) b3 = false;
					}
				}
			}
		}
	}
	std::cout << std::boolalpha;
	std::cout << "1 and 2 - " << b1 << std::endl;
	std::cout << "2 and 3 - " << b2 << std::endl;
	std::cout << "1 and 3 - " << b3 << std::endl;

}
//   A=L*M*l*m  
//   B=M*N*m*n  
//   C=L*N*l*n  
void multiplyIntrinsics() {
	for (int X = 0; X < L; X++) {
		for (int Z = 0; Z < N; Z++) {
			for (int Y = 0; Y < M; Y++) {

				__m256 clm0 = _mm256_load_ps(matrixB[Y][Z][0]);
				__m256 clm1 = _mm256_load_ps(matrixB[Y][Z][1]);
				__m256 clm2 = _mm256_load_ps(matrixB[Y][Z][2]);
				__m256 clm3 = _mm256_load_ps(matrixB[Y][Z][3]);
				__m256 clm4 = _mm256_load_ps(matrixB[Y][Z][4]);
				__m256 clm5 = _mm256_load_ps(matrixB[Y][Z][5]);
				__m256 clm6 = _mm256_load_ps(matrixB[Y][Z][6]);
				__m256 clm7 = _mm256_load_ps(matrixB[Y][Z][7]);

				for (int x = 0; x < l; x++) {

					__m256 rw0 = _mm256_broadcast_ss(&matrixA[X][Y][x][0]);
					__m256 rw1 = _mm256_broadcast_ss(&matrixA[X][Y][x][1]);
					__m256 rw2 = _mm256_broadcast_ss(&matrixA[X][Y][x][2]);
					__m256 rw3 = _mm256_broadcast_ss(&matrixA[X][Y][x][3]);
					__m256 rw4 = _mm256_broadcast_ss(&matrixA[X][Y][x][4]);
					__m256 rw5 = _mm256_broadcast_ss(&matrixA[X][Y][x][5]);
					__m256 rw6 = _mm256_broadcast_ss(&matrixA[X][Y][x][6]);
					__m256 rw7 = _mm256_broadcast_ss(&matrixA[X][Y][x][7]);

					rw0 = _mm256_mul_ps(clm0, rw0);
					rw1 = _mm256_mul_ps(clm1, rw1);
					rw2 = _mm256_mul_ps(clm2, rw2);
					rw3 = _mm256_mul_ps(clm3, rw3);
					rw4 = _mm256_mul_ps(clm4, rw4);
					rw5 = _mm256_mul_ps(clm5, rw5);
					rw6 = _mm256_mul_ps(clm6, rw6);
					rw7 = _mm256_mul_ps(clm7, rw7);

					rw0 = _mm256_add_ps(rw0, rw1);
					rw2 = _mm256_add_ps(rw2, rw3);
					rw4 = _mm256_add_ps(rw4, rw5);
					rw6 = _mm256_add_ps(rw6, rw7);

					rw0 = _mm256_add_ps(rw0, rw2);
					rw4 = _mm256_add_ps(rw4, rw6);

					rw0 = _mm256_add_ps(rw0, rw4);

					__m256 res = _mm256_load_ps(matrixC3[X][Z][x]);
					res = _mm256_add_ps(rw0, res);
					_mm256_store_ps(matrixC3[X][Z][x], res);
				}
			}
		}
	}

}

void multiplyVectorized() {
	for (int X = 0; X < L; X++) {
		for (int Z = 0; Z < N; Z++) {
			for (int Y = 0; Y < M; Y++) {
				for (int x = 0; x < l; x++) {
					for (int z = 0; z < n; z++) {
						matrixC1[X][Z][x][z] += matrixA[X][Y][x][0] * matrixB[Y][Z][0][z] +
							matrixA[X][Y][x][1] * matrixB[Y][Z][1][z] +
							matrixA[X][Y][x][2] * matrixB[Y][Z][2][z] +
							matrixA[X][Y][x][3] * matrixB[Y][Z][3][z] +
							matrixA[X][Y][x][4] * matrixB[Y][Z][4][z] +
							matrixA[X][Y][x][5] * matrixB[Y][Z][5][z] +
							matrixA[X][Y][x][6] * matrixB[Y][Z][6][z] +
							matrixA[X][Y][x][7] * matrixB[Y][Z][7][z];

					}
				}
			}
		}
	}
}

void multiplyNotVectorized() {

	for (int X = 0; X < L; X++) {
		for (int Z = 0; Z < N; Z++) {
			for (int Y = 0; Y < M; Y++) {
				for (int x = 0; x < l; x++) {

#pragma loop(no_vector)

					for (int z = 0; z < n; z++) {
						matrixC2[X][Z][x][z] += matrixA[X][Y][x][0] * matrixB[Y][Z][0][z] +
							matrixA[X][Y][x][1] * matrixB[Y][Z][1][z] +
							matrixA[X][Y][x][2] * matrixB[Y][Z][2][z] +
							matrixA[X][Y][x][3] * matrixB[Y][Z][3][z] +
							matrixA[X][Y][x][4] * matrixB[Y][Z][4][z] +
							matrixA[X][Y][x][5] * matrixB[Y][Z][5][z] +
							matrixA[X][Y][x][6] * matrixB[Y][Z][6][z] +
							matrixA[X][Y][x][7] * matrixB[Y][Z][7][z];

					}
				}
			}
		}
	}
}

int main() {
	srand(time(NULL));


	// A filling
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < M; j++) {
			for (int k = 0; k < l; k++) {
				for (int p = 0; p < m; p++) {
					matrixA[i][j][k][p] = (rand() % 10)/1.1;
				}
			}
		}
	}
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < m; k++) {
				for (int p = 0; p < n; p++) {
					matrixB[i][j][k][p] = (rand() % 10)/1.1;
				}
			}
		}
	}

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < l; k++) {
				for (int p = 0; p < n; p++) {
					matrixC1[i][j][k][p] = 0;
					matrixC2[i][j][k][p] = 0;
					matrixC3[i][j][k][p] = 0;
				}
			}
		}
	}

	multiplyVectorized();
	multiplyNotVectorized();
	multiplyIntrinsics();
	equalMatrix();

	std::cout << std::endl;

	//for (int i = 0; i < L; i++) {
	//	for (int j = 0; j < M; j++) {
	//		for (int k = 0; k < l; k++) {
	//			for (int p = 0; p < m; p++) {
	//				std::cout << matrixA[i][j][k][p] << "  ";
	//			}
	//		}
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
	//for (int i = 0; i < M; i++) {
	//	for (int j = 0; j < N; j++) {
	//		for (int k = 0; k < m; k++) {
	//			for (int p = 0; p < n; p++) {
	//				std::cout << matrixB[i][j][k][p] << "  ";
	//			}
	//		}
	//	}
	//	std::cout << std::endl;
	//}
	//std::cout << std::endl;
/*
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < l; k++) {
				for (int p = 0; p < n; p++) {
					std::cout << matrixC1[i][j][k][p] << "  ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << "------------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl;

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < l; k++) {
				for (int p = 0; p < n; p++) {
					std::cout << matrixC2[i][j][k][p] << "  ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << "------------------------------------------------------------------------------"<< std::endl;
	std::cout << std::endl;

	for (int i = 0; i < L; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < l; k++) {
				for (int p = 0; p < n; p++) {
					std::cout << matrixC3[i][j][k][p] << "  ";
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
*/

	return 0;
}