bench: simple.hcu streams.hcu graphs.hcu kernels.hcu kernels.cu main.cu util.hpp threading.hpp threading.cpp
	nvcc -DCASE=${CASE} -c -O3 -std=c++17 threading.cpp
	nvcc -DCASE=${CASE} -c -O3 -std=c++17 kernels.cu
	nvcc -DCASE=${CASE} -c -O3 -std=c++17 main.cu
	nvcc -DCASE=${CASE} -O3 kernels.o main.o threading.o -o $@

.PHONY: clean
clean:
	rm -rf bench a.out *.o
