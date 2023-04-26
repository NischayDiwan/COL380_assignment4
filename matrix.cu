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

__device__
int binASearch(int *v, int l, int p1, int p2){
	// binary search in array
	int s = 0;
	int e = l;
	while(s <= e){
		int m = s + (e-s) / 2;
		if(v[2*m] == p1 && v[2*m+1] == p2)
			return m;
		if((v[2*m] < p1) || (v[2*m] == p1 && v[2*m+1] < p2))
			s = m + 1;
		else
			e = m - 1;
	}
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
void matMulGPU(uint *a, uint *b, uint *c, int *ka, int *kb, int m, int n, int k1, int k2){
	extern __shared__ uint dab[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nm = n/m;
	int i = bid / nm;
	int k = bid % nm;
	// int gtid = bid*blockDim.x + tid;
	uint temp = 0;
	for (int j = 0; j < nm; j++)
	{
		int id1 = binASearch(ka,k1,i,j);
		int id2 = binASearch(kb,k2,j,k);
		if(!(id1 == -1 || id2 == -1)){
			dab[tid] = a[tid + id1*m*m];
			dab[tid + m*m] = b[tid + id2*m*m];
			__syncthreads();
			
			int ii = tid/m;
			int jj = tid%m;
			for (int kk = 0; kk < m; ++kk)
			{
				temp = temp + (dab[ii*m + kk] * dab[kk*m + jj + m*m]);
			}
			__syncthreads();
		}
	}
	c[tid + i*m*n + k*m*m] = temp;
}

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<pair<int,int>> &resm, vector<uint> &blksC, ofstream &outstr){
	int nm = n/m;

	// sending data to GPU
	int streamSize = 4;
	cudaStream_t stream[streamSize];
	// cudaError_t result[2];
	for(int i = 0;i<streamSize ;i++){
		cudaStreamCreate(&stream[i]);
	}
	int size = sizeof(uint);
	uint *a = &blksA[0], *b = &blksB[0], *c = new uint[n*n]();
	uint *da, *db, *dc;
	cudaMalloc(&da,size*blksA.size());
	cudaMalloc(&db,size*blksB.size());
	cudaMalloc(&dc,size*n*n);
	cudaMemset(dc,0,size*n*n);
	int *ka, *kb;
	cudaMalloc(&ka,2*sizeof(int)*mp1.size());
	cudaMalloc(&kb,2*sizeof(int)*mp2.size());
	cudaMemcpyAsync(da,a,size*blksA.size(),cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(db,b,size*blksB.size(),cudaMemcpyHostToDevice,stream[1]);
	cudaMemcpyAsync(ka,&mp1[0],2*sizeof(int)*mp1.size(),cudaMemcpyHostToDevice,stream[2]);
	cudaMemcpyAsync(kb,&mp2[0],2*sizeof(int)*mp2.size(),cudaMemcpyHostToDevice,stream[3]);

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
	// std::cout << "CSR converted\n";
	int *rof, *col;

	cudaDeviceSynchronize();
	// std::cout << "gpu multiplication start\n";
	chrono::time_point<std::chrono::system_clock> startg = std::chrono::system_clock::now();
	// core matrix multiplication
	// for(int i = 0; i < nm; ++i)
	// {	
	// 	for (int k = 0; k < nm; ++k)
	// 	{
	// 		for (int j = rofV[i]; j < rofV[i+1]; j++)
	// 		{
	// 			int cl = colV[j];
	// 			int id = binASearch(&mp2[0].first,mp2.size(),{cl,k});
	// 			if(!(id == -1)){
	// 				matMulGPU<<<1,m*m,2*size*m*m>>>(j,id,da,db,dc,m,n,i,k);
	// 			}
	// 		}
	// 	}
	// }
	matMulGPU<<<nm*nm,m*m,2*size*m*m>>>(da,db,dc,ka,kb,m,n,mp1.size(),mp2.size()); // i X k

	cudaDeviceSynchronize();
	// std::cout << "gpu multiplication done\n";
	chrono::time_point<std::chrono::system_clock> endg = std::chrono::system_clock::now();
	chrono::duration<double> elapsed_secondsg = endg-startg;
	std::cout << "gpu multiplication time: " << elapsed_secondsg.count() << "s\n";
	cudaMemcpy(c,dc,size*n*n,cudaMemcpyDeviceToHost);
	
	for(int i = 0; i < nm; ++i)
	{	
		for (int k = 0; k < nm; ++k)
		{
			vector<uint> vtemp(c+(i*nm + k)*m*m,c+(i*nm + k + 1)*m*m);
			if(!isZero(&vtemp[0],m*m)){
				pair<int,int> ptemp = {i,k};
				resm.push_back(ptemp);
				blksC.insert(blksC.end(),vtemp.begin(),vtemp.end());
			}
		}
	}

	// free the memory
	delete c;
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	for(int i = 0;i<streamSize ;i++){
		cudaStreamDestroy(stream[i]);
	}
}
