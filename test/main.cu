#include "matrix.h"

int main(int argc, char const *argv[])
{
	string inputA = "";
	string inputB = "";
	string outputC = "";

	// int thrds = 8;
	std::ofstream myFile;
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

    std::map<pair<int,int>,vector<vector<uint>>> inMapA;
    for (int i = 0; i < k1; ++i){
	    fileA.read(buffer1,4);
	    int a = givint(buffer1);
	    fileA.read(buffer1,4);
	    int b = givint(buffer1);
	    pair<int,int> pt = {a,b};
	    vector<vector<uint>> vt;
	    for (int j = 0; j < m; ++j){
	    	vector<uint> vt1;
		  	for (int h = 0; h < m; ++h){
				int e = 0;
			  	fileA.read((char *)&e,2);
			    vt1.push_back(e);
		  	}
		  	vt.push_back(vt1);
	    }
	    inMapA[pt] = vt; 
    }
	fileA.close();
	std::map<pair<int,int>,vector<vector<uint>>> inMapB;
	for (int i = 0; i < k2; ++i)
    {
	    fileB.read(buffer2,4);
	    int a = givint(buffer2);
	    fileB.read(buffer2,4);
	    int b = givint(buffer2);
	    pair<int,int> pt = {a,b};
	    vector<vector<uint>> vt;
	    for (int j = 0; j < m; ++j)
	    {
	    	vector<uint> vt1;
		  	for (int h = 0; h < m; ++h)
		  	{
				int e = 0;
			  	fileB.read((char *)&e,2);
			    vt1.push_back(e);
		  	}
		  	vt.push_back(vt1);
	    }
	    inMapB[pt] = vt; 
    }
	fileB.close();
    cout << "Input reading done, k values: " << inMapA.size() << " and " << inMapB.size() << endl;

    std::map<pair<int,int>,vector<vector<uint>>> outMap;
	matMul(inMapA, inMapB, n, m, k1, k2, outMap);

	// readable output
    myFile.open("Mat.txt");
    printmap(outMap,myFile);
	myFile.close();

    cout << "Writing output" << endl;
    outC.write((char *)&n,4);
    outC.write((char *)&m,4);
    int kout = 0;
    outC.write((char *)&kout,4);
    for(auto i = outMap.begin(); i != outMap.end(); i++){
	    int aout = i->first.first;
	    outC.write((char *)&aout,4);
	    int bout = i->first.second;
	    outC.write((char *)&bout,4);
	    for (int j = 0; j < m; ++j){
		  	for (int h = 0; h < m; ++h){
			  	uint eout = min(i->second[j][h],(uint)MAX_VAL);
			  	outC.write((char *)&eout,4);
		  	}
	    }
	    kout += 1;
    }

    cout << "Number of output non-zero blocks: " << kout << endl;
    outC.seekp(8);
    outC.write((char *)&kout,4);
    cout << "Writing done" << endl;
    outC.close();
	return 0;
}