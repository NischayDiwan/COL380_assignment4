make
for i in 10 50 100 500 1000 5000 10000
do
    python gen-input.py -n 8192 -m 4 -z ${i} -o input1
    python gen-input.py -n 8192 -m 4 -z 7889 -o input2
    python gen-output.py -f input1 input2 -e 2 -o output1
    time ./exec input1 input2 outFile.bin
    python checker.py -f output1 outFile.bin -e 4
done
# time ./exec input1v input2v outFile.bin
# python checker.py -f output1v outFile.bin -e 4
# cuda-gdb --args ./exec input1 input2 outFile.bin