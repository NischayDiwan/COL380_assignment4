#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;

#define MAX_VAL 4294967295
#define BLOCK_SIZE1 16
#define BLOCK_SIZE2 64
#define BLOCK_SIZEGPU blockDim.x

void printvec(vector<uint> &a,ofstream &outstr, int m);

int givint(char *buffer);

int binSearch(vector<pair<int,int>> &v, pair<int,int> p);

bool isZero(uint *v, int l);

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<uint> &blksC);
