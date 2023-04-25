#include "matrix.h"

void printvec(vector<uint> &a,ofstream &outstr, int m){
	for(int i = 0;i < a.size()/m;i++){
		for(int j=0; j < m; j++)
			outstr << a[i*m + j] << " ";
		outstr << endl;
	}
}

void printmap(map<pair<int,int>,vector<uint>> &mp, ofstream &outstr, int m){
	for(auto i = mp.begin(); i != mp.end(); i++){
		outstr << i->first.first << "," << i->first.second << ":\n";
		printvec((i->second),outstr,m);
	}
}

int givint(char *buffer){
	int *res = (int *)(buffer);
	return *res;
}

int binSearch(vector<pair<int,int>> &v, pair<int,int> p){
    auto it = lower_bound(v.begin(), v.end(), p);
	if(it != v.end()){
		int id = it - v.begin();
		if(v[id] == p)
			return id;
		else
			return -1;
	}
	return -1;
}

void inline transpose(vector<uint> &a,vector<uint> &b, int m){
	for(int i=0; i < m; i++){
		for(int j=0; j < m; j++){
			b[i*m + j] = a[j *m + i];
		}
	}
}

bool inline isZero(vector<uint> &v){
	bool flg = true;
	for (int i = 0; i < v.size(); ++i)
	{	
		if(v[i] != 0){
			flg = false;
			break;
		}
	}
	return flg;
}

void inline blockOuter(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m){
	for (int i = 0; i < v1.size(); ++i)
	{	
		r[i] = v1[i] + v2[i];
	}
}

void inline blockInner(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m){
	for (int i = 0; i < m; ++i)
	{	
		for (int j = 0; j < m; ++j)
		{
			uint temp = 0;
			for (int k = 0; k < m; ++k)
			{
				temp = temp + (v1[i*m + k] * v2[k*m + j]);
			}
			r[i*m + j] = temp;
		}
	}
}

__global__
void matMulGPU(uint *a, uint *b, uint *c, int m){
	extern __shared__ uint dab[];
	// int bid = blockIdx.x;
	int tid = threadIdx.x;
	// int gtid = bid*blockDim.x + tid;
	dab[tid] = a[tid];
	dab[tid + m*m] = b[tid];
	__syncthreads();
	uint temp = 0;
	int i = tid/m;
	int j = tid - i*m;
	for (int k = 0; k < m; ++k)
	{
		temp = temp + (dab[i*m + k] * dab[k*m + j + m*m]);
	}
	// __syncthreads();
	c[tid] = temp;
}

void inline blockGPUmul(vector<uint> &v1,vector<uint> &v2, vector<uint> &r, int m){
	uint *a = &v1[0], *b = &v2[0], *c = &r[0];
	uint *da, *db, *dc;
	int size = m*m*sizeof(int);
	cudaMalloc(&da,size);
	cudaMalloc(&db,size);
	cudaMalloc(&dc,size);
	cudaMemcpy(da,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(db,b,size,cudaMemcpyHostToDevice);
	matMulGPU<<<1,m*m,2*size>>>(da,db,dc,m);
	// cudaDeviceSynchronize();
	cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
}

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<pair<int,int>> &resm, vector<uint> &blksC, ofstream &outstr){
	int nm = n/m;
	
	vector<uint> valV(blksA.begin(),blksA.end());
	vector<uint> colV;
	vector<uint> rofV;
	// vector<uint> rowV;
	int offset = 0;
	int rowno = 0;
	rofV.push_back(offset);
	for(auto i = mp1.begin(); i != mp1.end(); i++){
		colV.push_back(i->second);
		if(i->first > rowno){
			for(int cc = 0;cc<(i->first-rowno);cc++){
				rofV.push_back(offset);
			}
			rowno = i->first;
		}
		offset+=1;
	}
	for(int j = rowno;j<nm;j++){
		rofV.push_back(valV.size()/m/m);
	}

	for(int i = 0; i < nm; ++i)
	{	
		for (int k = 0; k < nm; ++k)
		{
			vector<uint> vtemp(m*m,0);
			for (int j = rofV[i]; j < rofV[i+1]; j++)
			{
				int cl = colV[j];
				if(!(binSearch(mp2,{cl,k}) == -1)){
					vector<uint> vtemp1(m*m,0);
					vector<uint> tval(valV.begin() + j*m*m, valV.begin() + (j+1)*m*m);
					int id = binSearch(mp2,{cl,k});
					vector<uint> tval1(blksB.begin() + id*m*m, blksB.begin() + (id+1)*m*m);
					blockInner(tval,tval1,vtemp1,m);
					// blockGPUmul(valV[j],mp2[{cl,k}],vtemp1,m);
					blockOuter(vtemp,vtemp1,vtemp,m);
				}
			}
			if(!isZero(vtemp)){
				pair<int,int> ptemp = {i,k};
				resm.push_back(ptemp);
				blksC.insert(blksC.end(),vtemp.begin(),vtemp.end());
			}
		}
	}
}
