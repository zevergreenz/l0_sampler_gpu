# l0_sampler_gpu
A CUDA implementation of L0-sampler.

To test performance, edit `stream_generator.py` as appropriate, then generate stream data file with `python3 stream_generator.py`. By default, it generates 1 million updates, with indices in the range (0, 100,000).

Then run `make l0_sampler`, which will compile and run the l0 sampler to process the generated data file.