make_folder:
	mkdir -p bin out
cuda_build: make_folder
	nvcc -o bin/s_sparse_sampler -arch=sm_32 -lm -g s_sparse_sampler.cu
l0_build: make_folder
	nvcc -o bin/l0_sampler -arch=sm_32 -lm -g l0_sampler.cu

cuda: cuda_build
	bin/s_sparse_sampler data_stream.txt;

l0_sampler: l0_build
	bin/l0_sampler data_stream.txt;