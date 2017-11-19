cuda_build:
	mkdir -p bin out
	nvcc -o bin/s_sparse_sampler -arch=sm_32 -lm -g s_sparse_sampler.cu

cuda: cuda_build
	# /usr/bin/time -f "Real Time %e\nCPU Time %U \nKernel Time %S\nContext Switches %w" bin/mm-cuda 1000;
	bin/s_sparse_sampler data_100k.txt;

cuda_memcheck: cuda_build
	cuda-memcheck bin/s_sparse_sampler data_100k.txt;

cuda_gdb: cuda_build
	cuda-gdb bin/s_sparse_sampler;
