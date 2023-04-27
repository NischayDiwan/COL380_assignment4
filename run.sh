python gen-input.py -n 8192 -m 8 -z 500000 -o input1
python gen-input.py -n 8192 -m 8 -z 50 -o input2
python gen-output.py -f input1 input2 -e 2 -o output1
# python gen-output.py -f input2 input1 -e 2 -o output2
# python checker.py -f output1 output1 -e 4
# python checker.py -f output1 output2 -e 4
