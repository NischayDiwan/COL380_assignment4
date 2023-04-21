#include "matrix.h"

void printvec(vector<uint> &a,ofstream &outstr){
   	for(int i=0; i < a.size(); i++)
		outstr << a.at(i) << " ";
	outstr << endl;
}

void printmap(map<pair<int,int>,vector<vector<uint>>> &m, ofstream &outstr){
	for(auto i = m.begin(); i != m.end(); i++){
		outstr << i->first.first << "," << i->first.second << ":\n";
		for (int j = 0; j < (i->second).size(); ++j)
		{
			printvec((i->second)[j],outstr);
		}
	}
}

void inline transpose(vector<vector<uint>> &a,vector<vector<uint>> &b){
	for(int i=0; i < a.size(); i++){
		for(int j=0; j < a.size(); j++){
			b[i][j] = a[j][i];
		}
	}
}

int givint(char *buffer){
	int *res = (int *)(buffer);
	return *res;
}

void inline blockOuter(vector<vector<uint>> &v1,vector<vector<uint>> &v2, vector<vector<uint>> &r){
	for (int i = 0; i < v1.size(); ++i)
	{	
		for (int j = 0; j < v1.size(); ++j)
		{
			r[i][j] = v1[i][j] + v2[i][j];
		}
	}
}

void inline blockInner(vector<vector<uint>> &v1,vector<vector<uint>> &v2, vector<vector<uint>> &r){
	for (int i = 0; i < v1.size(); ++i)
	{	
		for (int j = 0; j < v1.size(); ++j)
		{
			uint temp = 0;
			for (int k = 0; k < v1.size(); ++k)
			{
				temp = temp + (v1[i][k] * v2[k][j]);
			}
			r[i][j] = temp;
		}
	}
}

bool inline isZero(vector<vector<uint>> &v){
	bool flg = true;
	for (int i = 0; i < v.size(); ++i)
	{	
		if(flg == false){
			break;
		}
		for (int j = 0; j < v.size(); ++j)
		{
			if(v[i][j] != 0){
				flg = false;
				break;
			}
		}
	}
	return flg;
}

// __global__
// void matMulGPU(void){
// 	int bid = blockIdx.x;
// 	int tid = threadIdx.x;
// 	int gtid = bid*blockDim.x + tid;
// }

void matMul(map<pair<int,int>,vector<vector<uint>>> &mp1, map<pair<int,int>,vector<vector<uint>>> &mp2, int n, int m, int k1, int k2, map<pair<int,int>,vector<vector<uint>>> &resm){
	int nm = n/m;
	// matMulGPU<<<m,m>>>();

	vector<vector<vector<uint>>> valV;
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
			vector<vector<uint>> vtemp(m,vector<uint>(m,0));
			for (int j = rofV[i]; j < rofV[i+1]; j++)
			{
				int cl = colV[j];
				if(!(mp2.find({cl,k}) == mp2.end())){
					vector<vector<uint>> vtemp1(m,vector<uint>(m,0));
					blockInner(valV[j],mp2[{cl,k}],vtemp1);
					blockOuter(vtemp,vtemp1,vtemp);
				}
			}
			if(!isZero(vtemp)){
				pair<int,int> ptemp = {i,k};
				resm[ptemp] = vtemp;
			}
		}
	}
}
