#include "matrix.h"

using namespace std;
#define timer 0

void printvec(vector<uint> &a,ofstream &outstr, int m){
	for(int i = 0;i < a.size()/m;i++){
		for(int j=0; j < m; j++)
			outstr << a[i*m + j] << " ";
		outstr << endl;
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

__global__
void matMulGPU(uint *a, uint *b, uint *c, int *ka, int *kb, int *rof, int *col, int m, int n, int k1, int k2){
	extern __shared__ uint dab[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nm = n/m;
	int i = bid / nm;
	int k = bid % nm;
	uint temp = 0;
	for (int j = rof[i]; j < rof[i+1]; j++){
		// int id1 = binASearch(ka,k1,i,j);
		int id1 = j;
		int id2 = binASearch(kb,k2,col[id1],k);
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
		__syncthreads();
	}
	__syncthreads();
	c[tid + i*m*n + k*m*m] = temp;

	// if(temp != 0)
	// 	printf("%d\n",c[tid + i*m*n + k*m*m]);
}

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  long long n, long long m, vector<uint> &blksC){
	long long nm = n/m;

	// sending data to GPU
	int streamSize = 6;
	cudaError_t err;
	cudaStream_t stream[streamSize];
	for(int i = 0;i<streamSize ;i++){
		cudaStreamCreate(&stream[i]);
	}
	size_t size = sizeof(uint);
	size_t size2 = sizeof(int);
	size_t size3 = size * n * n;
	uint *a = blksA.data(), *b = blksB.data(), *c = blksC.data();
	uint *da, *db, *dc;
	cudaMalloc(&da,size*blksA.size());
	cudaMalloc(&db,size*blksB.size());
	cudaMalloc(&dc,size3);
	cudaDeviceSynchronize();
	cudaMemset(dc,0,size3);
	int *ka, *kb;
	cudaMalloc(&ka,(size_t)2*size2*(size_t)mp1.size());
	cudaMalloc(&kb,(size_t)2*size2*(size_t)mp2.size());
	cudaMemcpyAsync(da,a,(size_t)size*(size_t)blksA.size(),cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(db,b,(size_t)size*(size_t)blksB.size(),cudaMemcpyHostToDevice,stream[1]);
	cudaMemcpyAsync(ka,&mp1[0],(size_t)2*size2*(size_t)mp1.size(),cudaMemcpyHostToDevice,stream[2]);
	cudaMemcpyAsync(kb,&mp2[0],(size_t)2*size2*(size_t)mp2.size(),cudaMemcpyHostToDevice,stream[3]);

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
	cudaMalloc(&rof,(size_t)rofV.size()*size2);
	cudaMalloc(&col,(size_t)colV.size()*size2);
	cudaMemcpyAsync(rof,&rofV[0],(size_t)rofV.size()*size2,cudaMemcpyHostToDevice,stream[4]);
	cudaMemcpyAsync(col,&colV[0],(size_t)colV.size()*size2,cudaMemcpyHostToDevice,stream[5]);

	int stride = (int)(nm*nm);
	if(timer){
		cudaDeviceSynchronize();
		chrono::time_point<std::chrono::system_clock> startg = std::chrono::system_clock::now();
		matMulGPU<<<stride,m*m,2*size*m*m,0>>>(da,db,dc,ka,kb,rof,col,m,n,mp1.size(),mp2.size()); // i X k
		cudaDeviceSynchronize();
		chrono::time_point<std::chrono::system_clock> endg = std::chrono::system_clock::now();
		chrono::duration<double> elapsed_secondsg = endg-startg;
		std::cout << "gpu multiplication time: " << elapsed_secondsg.count() << "s\n";
	}else{
		cudaDeviceSynchronize();
		matMulGPU<<<stride,m*m,2*size*m*m,0>>>(da,db,dc,ka,kb,rof,col,m,n,mp1.size(),mp2.size()); // i X k
	}

	cudaDeviceSynchronize();
	err = cudaMemcpy((void *)c,(void *)dc,size3,cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) 
	    printf("Error: %s\n", cudaGetErrorString(err));

	// free the memory
	cudaFree(da);
	cudaFree(db);
	cudaFree(ka);
	cudaFree(kb);
	cudaFree(rof);
	cudaFree(col);
	cudaFree(dc);
	for(int i = 0;i<streamSize ;i++){
		cudaStreamDestroy(stream[i]);
	}
}
