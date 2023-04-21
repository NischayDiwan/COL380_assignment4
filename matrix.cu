#include "matrix.h"

void printvec(vector<uint> &a,ofstream &outstr){
   	for(int i=0; i < a.size(); i++)
		outstr << a.at(i) << " ";
	outstr << endl;
}

void printmap(map<pair<int,int>,vector<uint>> &m, ofstream &outstr){
	for(auto i = m.begin(); i != m.end(); i++){
		outstr << i->first.first << "," << i->first.second << ":\n";
		printvec((i->second),outstr);
	}
}

void inline transpose(vector<uint> &a,vector<uint> &b, int m){
	for(int i=0; i < m; i++){
		for(int j=0; j < m; j++){
			b[i*m + j] = a[j *m + i];
		}
	}
}

int givint(char *buffer){
	int *res = (int *)(buffer);
	return *res;
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

__global__
void matMulGPU(void){
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int gtid = bid*blockDim.x + tid;
}

void matMul(map<pair<int,int>,vector<uint>> &mp1, map<pair<int,int>,vector<uint>> &mp2, int n, int m, int k1, int k2, map<pair<int,int>,vector<uint>> &resm){
	int nm = n/m;
	matMulGPU<<<m,m>>>();

	vector<vector<uint>> valV;
	vector<uint> colV;
	vector<uint> rofV;
	// vector<uint> rowV;
	int offset = 0;
	int rowno = 0;
	rofV.push_back(offset);
	for(auto i = mp1.begin(); i != mp1.end(); i++){
		valV.push_back(i->second);
		colV.push_back(i->first.second);
		// rowV.push_back(i->first.first);
		if(i->first.first > rowno){
			for(int cc = 0;cc<(i->first.first-rowno);cc++){
				rofV.push_back(offset);
			}
			rowno = i->first.first;
		}
		offset+=1;
	}
	for(int j = rowno;j<nm;j++){
		rofV.push_back(valV.size());
	}	
	for(int i = 0; i < nm; ++i)
	{	
		for (int k = 0; k < nm; ++k)
		{
			vector<uint> vtemp(m*m,0);
			for (int j = rofV[i]; j < rofV[i+1]; j++)
			{
				int cl = colV[j];
				if(!(mp2.find({cl,k}) == mp2.end())){
					vector<uint> vtemp1(m*m,0);
					blockInner(valV[j],mp2[{cl,k}],vtemp1,m);
					blockOuter(vtemp,vtemp1,vtemp,m);
				}
			}
			if(!isZero(vtemp)){
				pair<int,int> ptemp = {i,k};
				resm[ptemp] = vtemp;
			}
		}
	}
}
