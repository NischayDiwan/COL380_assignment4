#include "matrix.h"

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
	// int gtid = bid*blockDim.x + tid;
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
	}
	c[tid + i*m*n + k*m*m] = temp;
}

void matMul(vector<pair<int,int>> &mp1, vector<uint> &blksA, vector<pair<int,int>> &mp2, vector<uint> &blksB,  int n, int m, vector<uint> &blksC){
	int nm = n/m;

	// sending data to GPU
	int streamSize = 6;
	cudaStream_t stream[streamSize];
	for(int i = 0;i<streamSize ;i++){
		cudaStreamCreate(&stream[i]);
	}
	int size = sizeof(uint);
	uint *a = &blksA[0], *b = &blksB[0], *c = &blksC[0];
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
	cudaMalloc(&rof,rofV.size()*sizeof(int));
	cudaMalloc(&col,colV.size()*sizeof(int));
	cudaMemcpyAsync(rof,&rofV[0],rofV.size()*sizeof(int),cudaMemcpyHostToDevice,stream[4]);
	cudaMemcpyAsync(col,&colV[0],colV.size()*sizeof(int),cudaMemcpyHostToDevice,stream[5]);
	
	if(timer){
		cudaDeviceSynchronize();
		chrono::time_point<std::chrono::system_clock> startg = std::chrono::system_clock::now();
		matMulGPU<<<nm*nm,m*m,2*size*m*m>>>(da,db,dc,ka,kb,rof,col,m,n,mp1.size(),mp2.size()); // i X k
		cudaDeviceSynchronize();
		chrono::time_point<std::chrono::system_clock> endg = std::chrono::system_clock::now();
		chrono::duration<double> elapsed_secondsg = endg-startg;
		std::cout << "gpu multiplication time: " << elapsed_secondsg.count() << "s\n";
	}else{
		cudaDeviceSynchronize();
		matMulGPU<<<nm*nm,m*m,2*size*m*m>>>(da,db,dc,ka,kb,rof,col,m,n,mp1.size(),mp2.size()); // i X k
	}

	int chunk = n*n/4;
	cudaMemcpyAsync(c,dc,chunk*size,cudaMemcpyDeviceToHost,stream[0]);
	cudaMemcpyAsync(c+chunk,dc+chunk,size*chunk,cudaMemcpyDeviceToHost,stream[1]);
	cudaMemcpyAsync(c+2*chunk,dc+2*chunk,size*chunk,cudaMemcpyDeviceToHost,stream[2]);
	cudaMemcpyAsync(c+3*chunk,dc+3*chunk,size*chunk,cudaMemcpyDeviceToHost,stream[3]);
	cudaDeviceSynchronize();

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
