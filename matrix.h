#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <chrono>

#define MAX_VAL 4294967295
#define BLOCK_SIZE1 16
#define BLOCK_SIZE2 64
#define BLOCK_SIZEGPU blockDim.x

using namespace std;

void printvec(vector<uint> &a,ofstream &outstr, int m);

int givint(char *buffer);

bool isZero(uint *v, int l);

void matMul(vector<array<int,3>> &mp1, vector<uint> &blksA, vector<array<int,3>> &mp2, vector<uint> &blksB, long long n, long long m, vector<uint> &blksC);
