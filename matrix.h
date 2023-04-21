#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <utility>

using namespace std;

#define MAX_VAL 4294967295
#define BLOCK_SIZE1 16
#define BLOCK_SIZE2 64
#define BLOCK_SIZEGPU blockDim.x

void printvec(vector<uint> &a,ofstream &outstr, int m);

void printmap(map<pair<int,int>,vector<uint>> &mp, ofstream &outstr, int m);

int givint(char *buffer);

void inline transpose(vector<uint> &a,vector<uint> &b, int m);

void inline blockOuter(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m);

void inline blockInner(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m);

bool inline isZero(vector<uint> &v);

__global__ void matMulGPU(void);

void matMul(map<pair<int,int>,vector<uint>> &mp1, map<pair<int,int>,vector<uint>> &mp2, int n, int m, int k1, int k2, map<pair<int,int>,vector<uint>> &resm);
