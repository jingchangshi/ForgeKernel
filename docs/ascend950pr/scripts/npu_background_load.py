import os, time, sys
import torch, torch_npu
torch.npu.set_device(int(sys.argv[1]))
n = 8192 * 4096
a = torch.randn(n, dtype=torch.bfloat16, device='npu')
b = torch.randn(n, dtype=torch.bfloat16, device='npu')
open(sys.argv[2], 'w').write(str(os.getpid()))
deadline = time.time() + float(sys.argv[3])
while time.time() < deadline:
    for _ in range(50):
        c = a * b
    torch.npu.synchronize()
print('load done')
