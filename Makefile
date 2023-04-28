CC=nvcc 
CFLAGS=-std=c++11 -arch=sm_35 -O3 -g 

sources=main.cu
objects=$(sources:.cu=.o)

exec:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: exec
	./exec inputFile1.bin inputFile2.bin outFile.bin

%.o: %.cu
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o exec outFile.bin Mat.txt inputFile1.bin inputFile2.bin
