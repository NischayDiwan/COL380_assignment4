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

int binASearch(int *v, int l, pair<int,int> p){
	// binary search in array
	return -1;
}

bool isZero(uint *v, int l){
	bool flg = true;
	for (int i = 0; i < l; ++i)
	{	
		if(v[i] != 0){
			flg = false;
			break;
		}
	}
	return flg;
}

void inline transpose(vector<uint> &a,vector<uint> &b, int m){
	for(int i=0; i < m; i++){
		for(int j=0; j < m; j++){
			b[i*m + j] = a[j *m + i];
		}
	}
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
void matMulGPU(int id1, int id2, uint *a, uint *b, uint *c, int m, int rx1, int rx2){
	extern __shared__ uint dab[];
	// int bid = blockIdx.x;
	int tid = threadIdx.x;
	// int gtid = bid*blockDim.x + tid;
	dab[tid] = a[tid + id1*m*m];
	dab[tid + m*m] = b[tid + id2*m*m];
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

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<pair<int,int>> &resm, vector<uint> &blksC, ofstream &outstr){
	int nm = n/m;

	// sending data to GPU
	cudaStream_t stream[2];
	// cudaError_t result[2];
	for(int i = 0;i<2;i++){
		cudaStreamCreate(&stream[i]);
	}
	int size = m*m*sizeof(int);
	uint *a = &blksA[0], *b = &blksB[0], *c = &blksC[0];
	uint *da, *db, *dc;
	cudaMalloc(&da,size*blksA.size()/m/m);
	cudaMalloc(&db,size*blksB.size()/m/m);
	cudaMalloc(&dc,size);
	cudaMemcpyAsync(da,a,size*blksA.size()/m/m,cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(db,b,size*blksB.size()/m/m,cudaMemcpyHostToDevice,stream[1]);

	// converting to CSR
	vector<int> colV;
	vector<int> rofV;
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
		rofV.push_back(blksA.size()/m/m);
	}
	cout << "CSR converted\n";

	// thrust::device_vector<uint> dblksA(blksA.begin(), blksA.end());
	// thrust::device_vector<uint> dblksB(blksB.begin(), blksB.end());
	// thrust::device_vector<uint> dblksC(n*n,0);
	// thrust::copy(blksA.begin(), blksA.end(), dblksA.begin());
	// thrust::copy(blksB.begin(), blksB.end(), dblksB.begin());

	cudaDeviceSynchronize();
	cout << "copy done\n";
	// core matrix multiplication
	for(int i = 0; i < nm; ++i)
	{	
		for (int k = 0; k < nm; ++k)
		{
			vector<uint> vtemp(m*m,0);
			for (int j = rofV[i]; j < rofV[i+1]; j++)
			{
				int cl = colV[j];
				if(!(binSearch(mp2,{cl,k}) == -1)){
					int id = binSearch(mp2,{cl,k});
					vector<uint> vtemp1(m*m,0);
					c = &vtemp1[0];
					matMulGPU<<<1,m*m,2*size>>>(j,id,da,db,dc,m,i,k);
					// cudaDeviceSynchronize();
					cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);
					// blockInner(tval,tval1,vtemp1,m);
					blockOuter(vtemp,vtemp1,vtemp,m);
				}
			}
			if(!isZero(&vtemp[0],m*m)){
				pair<int,int> ptemp = {i,k};
				resm.push_back(ptemp);
				blksC.insert(blksC.end(),vtemp.begin(),vtemp.end());
			}
		}
	}
	// thrust::host_vector<uint> hblksC(dblksC.size());
	// thrust::copy(dblksC.begin(), dblksC.end(), hblksC.begin());
	// for(int i = 0; i < nm; ++i)
	// {	
	// 	for (int k = 0; k < nm; ++k)
	// 	{
	// 		vector<uint> vtemp(hblksC.begin() + (i*nm + k)*m*m, hblksC.begin() + (i*nm + k + 1)*m*m);
	// 		if(!isZero(vtemp)){
	// 			pair<int,int> ptemp = {i,k};
	// 			resm.push_back(ptemp);
	// 			blksC.insert(blksC.end(),vtemp.begin(),vtemp.end());
	// 		}
	// 	}
	// }

	// free the cuda memory
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	for(int i = 0;i<2;i++){
		cudaStreamDestroy(stream[i]);
	}
}
