make 
time ./exec input1 input2 outFile.bin
python checker.py -f output1 outFile.bin -e 4
# time ./exec input1m input2m outFile.bin
# time ./exec ../input1l ../input2l outFile.bin
# cuda-gdb --args ./exec input1 input2 outFile.bin