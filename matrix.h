#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
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

void printmap(map<pair<int,int>,vector<uint>> &mp, ofstream &outstr, int m);

int givint(char *buffer);

int binSearch(vector<pair<int,int>> &v, pair<int,int> p);

int binASearch(int *v, int l, pair<int,int> p);

bool isZero(uint *v, int l);

// void inline transpose(vector<uint> &a,vector<uint> &b, int m);

// void inline blockOuter(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m);

// void inline blockInner(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m);

// void inline blockGPUmul(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m);

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<pair<int,int>> &resm, vector<uint> &blksC, ofstream &outstr);
