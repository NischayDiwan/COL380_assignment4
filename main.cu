#include "matrix.h"

using namespace std;
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

    std::map<pair<int,int>,vector<uint>> inMapA;
    for (long long i = 0; i < k1; ++i){
	    fileA.read(buffer1,4);
	    int a = givint(buffer1);
	    fileA.read(buffer1,4);
	    int b = givint(buffer1);
	    pair<int,int> pt = {a,b};
	    vector<uint> vt;
	    for (int j = 0; j < m; ++j){
		  	for (int h = 0; h < m; ++h){
				int e = 0;
			  	fileA.read((char *)&e,2);
			    vt.push_back(e);
		  	}
	    }
	    inMapA[pt] = vt; 
    }
	fileA.close();
	std::map<pair<int,int>,vector<uint>> inMapB;
	for (long long i = 0; i < k2; ++i)
    {
	    fileB.read(buffer2,4);
	    int a = givint(buffer2);
	    fileB.read(buffer2,4);
	    int b = givint(buffer2);
	    pair<int,int> pt = {a,b};
	    vector<uint> vt;
	    for (int j = 0; j < m; ++j){
		  	for (int h = 0; h < m; ++h){
				int e = 0;
			  	fileB.read((char *)&e,2);
			    vt.push_back(e);
		  	}
	    }
	    inMapB[pt] = vt; 
    }
	fileB.close();
    cout << "Input reading done, k values: " << inMapA.size() << " and " << inMapB.size() << endl;

	vector<pair<int,int>> keysa;
	vector<uint> vala;
	for(auto i = inMapA.begin(); i != inMapA.end(); i++){
		keysa.push_back(i->first);
		vala.insert(vala.end(),i->second.begin(),i->second.end());
	}

	vector<pair<int,int>> keysb;
	vector<uint> valb;
	for(auto i = inMapB.begin(); i != inMapB.end(); i++){
		keysb.push_back(i->first);
		valb.insert(valb.end(),i->second.begin(),i->second.end());
	}

	assert(keysa.size() == vala.size()/m/m);
	assert(keysb.size() == valb.size()/m/m);
	assert(keysa.size() == k1);
	assert(keysb.size() == k2);
	inMapA.clear();
	inMapB.clear();

	end = chrono::system_clock::now();
	elapsed_seconds = end-start;
	std::cout << "Map to vector conversion: " << elapsed_seconds.count() << endl;

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
					uint eout = min(vtemp[k],(uint)MAX_VAL);
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