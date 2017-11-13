cuda_build:
	mkdir -p bin out
	nvcc -o bin/s_sparse_sampler -arch=sm_32 -lm -ccbin clang-3.8 s_sparse_sampler.cu

cuda: cuda_build
	# /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-cuda 1000;
bin/mm-cuda in/10int.dat;