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

void printvec(vector<uint> &a,ofstream &outstr);

void printmap(map<pair<int,int>,vector<vector<uint>>> &m, ofstream &outstr);

int givint(char *buffer);

void inline transpose(vector<vector<uint>> &a,vector<vector<uint>> &b);

void inline blockOuter(vector<vector<uint>> &v1,vector<vector<uint>> &v2, vector<vector<uint>> &r);

void inline blockInner(vector<vector<uint>> &v1,vector<vector<uint>> &v2, vector<vector<uint>> &r);

bool inline isZero(vector<vector<uint>> &v);

// __global__ void matMulGPU(void);

void matMul(map<pair<int,int>,vector<vector<uint>>> &mp1, map<pair<int,int>,vector<vector<uint>>> &mp2, int n, int m, int k1, int k2, map<pair<int,int>,vector<vector<uint>>> &resm);
