import torch

import tensorrt_llm

print("tensorrt_llm:", tensorrt_llm.__file__)
print("block ops:", [n for n in dir(torch.ops.trtllm) if "block_scale" in n])
