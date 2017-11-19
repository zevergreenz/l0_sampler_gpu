make_folder:
	mkdir -p bin out
cuda_build: make_folder
	nvcc -o bin/s_sparse_sampler -arch=sm_32 -lm -g s_sparse_sampler.cu
l0_build: make_folder
	nvcc -o bin/l0_sampler -arch=sm_32 -lm -g l0_sampler.cu

cuda: cuda_build
	bin/s_sparse_sampler data_100k.txt;

cuda_large: cuda_build
	bin/s_sparse_sampler data_stream.txt;

cuda_memcheck: cuda_build
	cuda-memcheck bin/s_sparse_sampler data_100k.txt;

cuda_gdb: cuda_build
	cuda-gdb bin/s_sparse_sampler;

l0_sampler: l0_build
	bin/l0_sampler data_100k.txt;

l0_sampler_large: l0_build
	bin/l0_sampler data_stream.txt;