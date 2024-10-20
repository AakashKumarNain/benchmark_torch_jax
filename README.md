# benchmark_torch_jax
The idea behind this repository is to build a set of models both in JAX and torch so that they can be used to compare the
performance difference between the two frameworks on different hardwares. The objective is to help the developers of the
frameowrks to identify the potential gaps/bugs that can be addressed at the library level

**Note:** The code in this repository sevres the purpose of benchmarking. For most cases, we will try to use standalone py file for each
framework. Once the benchmark is done, we can refactor and format the code to make it more prettier, but it's not the end goal of this
particular codebase


# How to run the benchmarks?

1. `git clone https://github.com/AakashKumarNain/benchmark_torch_jax.git`
2. `cd benchmark_torch_jax/`
3. `pip3 install -r requirements.txt`
4. `cd gpt2/`
5. `python jax_single_gpu.py input.txt` and `python torch_single_gpu.py input.txt`


# Results

Please check the [results.md](./results.md) file for the initial results. We will store the logs properly once some initial benchmarks results are complete. 
 

# References

1. [nanoGPT](https://github.com/karpathy/build-nanogpt/tree/master) by Karpathy