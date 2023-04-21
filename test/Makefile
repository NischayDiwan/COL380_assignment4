CC=nvcc
CFLAGS=-std=c++17 -O3 -g

sources=main.cu matrix.cu
objects=$(sources:.cu=.o)

exec:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: exec
	./exec input1 input2 outFile.bin

%.o: %.cu matrix.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm *.o exec outFile.bin Mat.txt mprof*