#include <cuda.h>
#include <cuda_runtime.h>
#include <bits/stdc++.h>
#include <chrono>

#define MAX_VAL 4294967295

using namespace std;
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

__global__
void matMulGPU(uint *a, uint *b, uint *c, int *ka, int *kb, int *rof, int *col, int* row, int *cof, int m, int n, int k1, int k2){
	extern __shared__ uint dab[];
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int nm = n/m;
	int i = bid / nm;
	int k = bid % nm;
	uint64_t temp = 0;
	int j1 = rof[i];
	int j2 = cof[k];
	while (j1 < rof[i+1] && j2 < cof[k+1]){
		int id1 = ka[j1];
		int id2 = kb[j2];
		int cl = col[j1];
		int rw = row[j2];
		int flag = 0;
		if(cl == rw){
			flag = 1;
			j1++;
			j2++;
		}
		else if(cl < rw)
			j1++;
		else
			j2++;
		// if(tid == 0 && bid == 9)
		// 	printf("%d %d : %d %d : %d %d\n",i,cl,cl,k,id1,id2);
		if(flag){
			__syncthreads();
			dab[tid] = (uint)a[tid + id1*m*m];
			dab[tid + m*m] = (uint)b[tid + id2*m*m];
			__syncthreads();
			int ii = tid/m;
			int jj = tid%m;
			for (int kk = 0; kk < m; ++kk)
			{
				temp = temp + (uint64_t)(dab[ii*m + kk] * dab[kk*m + jj + m*m]);
			}
		}
	}
	c[tid + i*m*n + k*m*m] = min(temp,MAX_VAL);
}

void matMul(vector<array<int,3>> &mp1, vector<uint> &blksA, vector<array<int,3>> &mp2, vector<uint> &blksB,  long long n, long long m, vector<uint> &blksC){
	long long nm = n/m;
	// sending data to GPU
	int streamSize = 2;
	cudaError_t err;
	cudaStream_t stream[streamSize];
	for(int i = 0;i<streamSize ;i++){
		cudaStreamCreate(&stream[i]);
	}
	size_t size = sizeof(uint);
	size_t size2 = sizeof(int);
	size_t size3 = sizeof(uint) * n * n;
	uint *a = &blksA[0], *b = &blksB[0], *da, *db;
	uint *c = &blksC[0], *dc;
	cudaMalloc(&da,size*blksA.size());
	cudaMalloc(&db,size*blksB.size());
	cudaMalloc(&dc,size3);
	cudaDeviceSynchronize();
	cudaMemset(dc,0,size3);
	cudaMemcpyAsync(da,a,(size_t)size*(size_t)blksA.size(),cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(db,b,(size_t)size*(size_t)blksB.size(),cudaMemcpyHostToDevice,stream[1]);

	// converting to CSR
	vector<int> valV;
	vector<int> colV;
	vector<int> rofV;
	int offset = 0;
	int rowno = 0;
	rofV.push_back(offset);
	for(int i = 0; i <mp1.size() ; i++){
		colV.push_back(mp1[i][1]);
		valV.push_back(mp1[i][2]);
		if(mp1[i][0] > rowno){
			for(int cc = 0;cc<(mp1[i][0]-rowno);cc++){
				rofV.push_back(offset);
			}
			rowno = mp1[i][0];
		}
		offset+=1;
	}
	for(int j = rowno;j<nm;j++){
		rofV.push_back(blksA.size()/m/m);
	}
	vector<int> valV2;
	vector<int> cofV;
	vector<int> rowV;
	offset = 0;
	int colno = 0;
	cofV.push_back(offset);
	for(int i = 0; i <mp2.size() ; i++){
		rowV.push_back(mp2[i][0]);
		valV2.push_back(mp2[i][2]);
		if(mp2[i][1] > colno){
			for(int cc = 0;cc<(mp2[i][1]-colno);cc++){
				cofV.push_back(offset);
			}
			colno = mp2[i][1];
		}
		offset+=1;
	}
	for(int j = colno;j<nm;j++){
		cofV.push_back(blksB.size()/m/m);
	}
	std::cout << "CSR converted\n";

	int *ka;
	cudaMalloc(&ka,(size_t)size2*(size_t)mp1.size());
	cudaMemcpyAsync(ka,valV.data(),(size_t)size2*(size_t)mp1.size(),cudaMemcpyHostToDevice,stream[0]);
	int *kb;
	cudaMalloc(&kb,(size_t)size2*(size_t)mp2.size());
	cudaMemcpyAsync(kb,valV2.data(),(size_t)size2*(size_t)mp2.size(),cudaMemcpyHostToDevice,stream[1]);
	int *rof, *col, *cof, *row;
	cudaMalloc(&rof,(size_t)rofV.size()*size2);
	cudaMalloc(&col,(size_t)colV.size()*size2);
	cudaMalloc(&cof,(size_t)cofV.size()*size2);
	cudaMalloc(&row,(size_t)rowV.size()*size2);
	cudaMemcpyAsync(rof,&rofV[0],(size_t)rofV.size()*size2,cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(col,&colV[0],(size_t)colV.size()*size2,cudaMemcpyHostToDevice,stream[1]);
	cudaMemcpyAsync(cof,&cofV[0],(size_t)cofV.size()*size2,cudaMemcpyHostToDevice,stream[0]);
	cudaMemcpyAsync(row,&rowV[0],(size_t)rowV.size()*size2,cudaMemcpyHostToDevice,stream[1]);

	int stride = (int)(nm*nm);
	// chrono::time_point<std::chrono::system_clock> startg = std::chrono::system_clock::now();
	cudaDeviceSynchronize();
	matMulGPU<<<stride,m*m,2*size*m*m,0>>>(da,db,dc,ka,kb,rof,col,row,cof,m,n,mp1.size(),mp2.size()); // i X k
	// cudaDeviceSynchronize();
	// chrono::time_point<std::chrono::system_clock> endg = std::chrono::system_clock::now();
	// chrono::duration<double> elapsed_secondsg = endg-startg;
	// std::cout << "gpu multiplication time: " << elapsed_secondsg.count() << "s\n";

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

bool compmy(const array<int,3> &a, const array<int,3> &b){
	if(a[1] == b[1]){
		if(a[0] == b[0]){
			return a[2] < b[2];
		}
		return a[0] < b[0];
	}
	return a[1] < b[1];
};

int main(int argc, char const *argv[])
{
	string inputA = "";
	string inputB = "";
	string outputC = "";

	// int thrds = 8;
	// std::ofstream myFile;
	// myFile.open("Mat.txt");
	chrono::time_point<std::chrono::system_clock> start, end;
	chrono::duration<double> elapsed_seconds;
	start = chrono::system_clock::now();
	
	if(argc == 4){
		inputA = argv[1];
		inputB = argv[2];
		outputC = argv[3];
	}else{
		printf("Incorrect number of arguments\n");
		cout << "Given: " << argc - 1 << " Expected: 3" << endl;
		return 0;
	}
	char buffer1[4];
	char buffer2[4];

    std::ifstream fileA;
    fileA.open(inputA,std::ios::binary);
	std::ifstream fileB;
	fileB.open(inputB,std::ios::binary);
    std::ofstream outC;
    outC.open(outputC);


    fileA.read(buffer1,4);
    int n1 = givint(buffer1);
    fileA.read(buffer1,4);
    int m1 = givint(buffer1);
    fileA.read(buffer1,4);
    int k1 = givint(buffer1);

	fileB.read(buffer2,4);
	int n2 = givint(buffer2);
	fileB.read(buffer2,4);
	int m2 = givint(buffer2);
	fileB.read(buffer2,4);
	int k2 = givint(buffer2);

	assert(n1 == n2);
	assert(m1 == m2);
	int n = n1, m = m1;

	vector<array<int,3>> keysa;
	vector<uint> vala;
	vector<array<int,3>> keysb;
	vector<uint> valb;

    // std::map<pair<int,int>,vector<uint>> inMapA;
    for (int i = 0; i < k1; ++i){
	    fileA.read(buffer1,4);
	    int a = givint(buffer1);
	    fileA.read(buffer1,4);
	    int b = givint(buffer1);
		array<int,3> pt = {a,b,i};
		keysa.push_back(pt);
	    vector<uint> vt;
	    for (int j = 0; j < m; ++j){
		  	for (int h = 0; h < m; ++h){
				int e = 0;
			  	fileA.read((char *)&e,2);
			    vt.push_back(e);
		  	}
	    }
		vala.insert(vala.end(),vt.begin(),vt.end());
    }
	fileA.close();
	// std::map<pair<int,int>,vector<uint>> inMapB;
	for (int i = 0; i < k2; ++i)
    {
	    fileB.read(buffer2,4);
	    int a = givint(buffer2);
	    fileB.read(buffer2,4);
	    int b = givint(buffer2);
		array<int,3> pt = {a,b,i};
		keysb.push_back(pt);
	    vector<uint> vt;
	    for (int j = 0; j < m; ++j){
		  	for (int h = 0; h < m; ++h){
				int e = 0;
			  	fileB.read((char *)&e,2);
			    vt.push_back(e);
		  	}
	    }
		valb.insert(valb.end(),vt.begin(),vt.end());
    }
	fileB.close();
    cout << "Input reading done, n: " << n << ", m: " << m <<  ", k-values: " << keysa.size() << " and " << keysb.size() << endl;
	
	sort(keysa.begin(), keysa.end());
	sort(keysb.begin(), keysb.end(), compmy);
	
	assert(keysa.size() == vala.size()/m/m);
	assert(keysb.size() == valb.size()/m/m);
	assert(keysa.size() == k1);
	assert(keysb.size() == k2);
	// inMapA.clear();
	// inMapB.clear();
	end = chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "Sorted input: " << elapsed_seconds.count() << endl;

	// vector<pair<int,int>> keysc;
	vector<uint> valc((size_t)n*(size_t)n,0);
	start = chrono::system_clock::now();
	matMul(keysa, vala, keysb, valb, n, m, valc);
	end = chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "Time taken for matrix multiplication: " << elapsed_seconds.count() << endl;

	// readable output
	int nm = n/m;
    cout << "Writing output" << endl;
    outC.write((char *)&n,4);
    outC.write((char *)&m,4);
    long kout = 0;
    outC.write((char *)&kout,4);
	for(int i = 0; i < nm; ++i)
	{	
		for (int j = 0; j < nm; ++j)
		{
			vector<uint> vtemp(valc.begin()+(i*nm + j)*m*m,valc.begin()+(i*nm + j + 1)*m*m);
			if(!isZero(&vtemp[0],m*m)){
				int aout = i;
				outC.write((char *)&aout,4);
				int bout = j;
				outC.write((char *)&bout,4);
				for (int k = 0; k < m*m; ++k){
					uint eout = vtemp[k];
					outC.write((char *)&eout,4);
				}
				kout += 1;
			}
		}
	}
    cout << "Number of output non-zero blocks: " << kout << endl;
    outC.seekp(8);
    outC.write((char *)&kout,4);
    cout << "Writing done" << endl;
    outC.close();
	// myFile.close();
	return 0;
}