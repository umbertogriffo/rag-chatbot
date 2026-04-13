# KV Cache

## Integer Mapping Table

The integer values are defined by the underlying GGML library:

| Integer Value | GGML Type      | Description                                                 |
|---------------|----------------|-------------------------------------------------------------|
| 0             | GGML_TYPE_F32  | 32-bit float (Highest precision, high VRAM)                 |
| 1             | GGML_TYPE_F16  | 16-bit float (Default)                                      |
| 2             | GGML_TYPE_Q4_0 | 4-bit quantization                                          |
| 3             | GGML_TYPE_Q4_1 | 4-bit quantization (variant 1)                              |
| 4             | GGML_TYPE_Q4_2 | 4-bit quantization (variant 2) **support has been removed** |
| 5             | GGML_TYPE_Q4_3 | 4-bit quantization (variant 3) **support has been removed** |
| 6             | GGML_TYPE_Q5_0 | 5-bit quantization                                          |
| 7             | GGML_TYPE_Q5_1 | 5-bit quantization (variant 1)                              |
| 8             | GGML_TYPE_Q8_0 | 8-bit quantization (Best balance of memory/speed)           |


## Install Flash Attention

Installing [flash-attn](https://github.com/dao-ailab/flash-attention) is resource-intensive, requiring over 16.8 GB of RAM per job during compilation and takes hours:
- https://github.com/Dao-AILab/flash-attention/issues/1043
- https://github.com/Dao-AILab/flash-attention/issues/2051

Found out a trick here to skip the compilation installing directly a pre-build wheels https://www.reddit.com/r/LocalLLaMA/comments/1no4ho1/some_things_i_learned_about_installing_flashattn/

Find the corresponding version of a wheel:
https://mjunya.com/flash-attention-prebuild-wheels/?package=FA2&flash=2.8.3&python=3.12&torch=2.10&cuda=12.6&platform=Linux+x86_64

and then:

```
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu126torch2.10-cp312-cp312-linux_x86_64.whl
```

When the model is loaded you can check the kv cache:
```
llama_kv_cache: size =  272.00 MiB (  4096 cells,  32 layers,  1/1 seqs), K (q8_0):  136.00 MiB, V (q8_0):  136.00 MiB
```

With the following configuration we doubled the context window using the same amount of VRAM, and the inference is faster:
```python
class Llama31Settings(ModelSettings):
    url = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    file_name = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    config = {
        "n_ctx": 8096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        "n_threads": 8,  # The number of CPU threads to use, tailor to your system and the resulting performance
        "n_gpu_layers": 33,  # The number of layers to offload to GPU, if you have GPU acceleration available
        "flash_attn": True,  # Whether to use flash attention, which can speed up inference on compatible hardware
        "type_k": 2, # Q4_0
        "type_v": 2, # Q4_0
    }
    config_answer = {"temperature": 0.7, "stop": []}
```
