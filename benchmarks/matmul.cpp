#include <blitz/array.h>
#include <ctime>
#include <iostream>
using namespace std;

int main()
{
  static const int ROWS_A = 2000000;
  static const int COLS_A = 3;
  static const int COLS_M = 3;
  static const int ROWS_M = COLS_A;
  clock_t t_start, t_stop;

  double* orig_M = new double[ROWS_M*COLS_M];
  double* orig_A = new double[ROWS_A*COLS_A];

  for (int i = 0; i < ROWS_M*COLS_M; i++)
    orig_M[i] = 20.0 * random()/(RAND_MAX+1.0);
  for (int i = 0; i < ROWS_A*COLS_A; i++)
    orig_A[i] = random()/(RAND_MAX+1.0);

//* start plain array

  double* M1 = new double[ROWS_M*COLS_M];
  double* A1 = new double[ROWS_A*COLS_A];

  for (int i = 0; i < ROWS_M; i++)
    for (int j = 0; j < COLS_M; j++)
      M1[i*COLS_M+j] = orig_M[i*COLS_M+j];

  for (int i = 0; i < ROWS_A; i++)
    for (int j = 0; j < COLS_A; j++)
      A1[i*COLS_A+j] = orig_A[i*COLS_A+j];

  double* B1 = new double[ROWS_A*COLS_M];
  assert(COLS_A == ROWS_M);

  t_start = clock();

  for (int i = 0; i < ROWS_A; i++)
    for (int j = 0; j < COLS_M; j++) {
      B1[i*COLS_M+j] = 0.0;
      for (int k = 0; k < COLS_A; k++)
        B1[i*COLS_M+j] += A1[i*COLS_A+k] * M1[k*COLS_M+j];
    }

  t_stop = clock();
  cout << "plain array: " << t_stop - t_start << endl;
  
  delete[] M1, A1, B1;

// end plain array */

//* start blitz array

  using namespace blitz;
  
  Range rM(0, ROWS_M-1), cM(0, COLS_M-1);
  Range rA(0, ROWS_A-1), cA(0, COLS_A-1);
  Array<double, 2> M2(rM, cM), A2(rA, cA);

  for (int i = 0; i < ROWS_M; i++)
    for (int j = 0; j < COLS_M; j++)
      M2(i,j) = orig_M[i*3+j];

  for (int i = 0; i < ROWS_A; i++)
    for (int j = 0; j < COLS_A; j++)
      A2(i,j) = orig_A[i*3+j];

  Array<double, 2> B2(rA, cM);
  assert(A2.columns() == M2.rows());

  firstIndex  i;
  secondIndex j;
  thirdIndex  k;

  t_start = clock();

  B2 = sum(A2(i,k) * M2(k,j), k);

  t_stop = clock();
  cout << "blitz array: " << t_stop - t_start << endl;

// end blitz array */

  return 0;
}
