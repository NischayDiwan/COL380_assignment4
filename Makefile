CC=nvcc
CFLAGS=--compiler-bindir "/usr/bin/g++-10" -O3 -g 

sources=main.cu matrix.cu
objects=$(sources:.cu=.o)

exec:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: exec
	./exec inputFile1.bin inputFile2.bin outFile.bin

%.o: %.cu matrix.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o exec outFile.bin Mat.txt inputFile1.bin inputFile2.bin