# Control: do the macOS counters respond to a plain anonymous 512 MB allocation, and does
# load_state_dict() from an mmap'd state dict leave the file pages reclaimable?
import os, sys, gc
import torch, psutil
from safetensors.torch import save_file, load_file
p = psutil.Process()
def fp(tag):
    mi = p.memory_full_info(); print(f"{tag:45s} rss={mi.rss/2**20:8.0f} MB  uss={mi.uss/2**20:8.0f} MB")
n = 128 * 1024 * 1024
fp("start")
x = torch.ones(n)                       # 512 MB anonymous
fp("after torch.ones(512MB) anonymous")
del x; gc.collect()
fp("after del anonymous")
path = os.path.join(sys.argv[1], "ctl.safetensors")
save_file({"w": torch.randn(n)}, path); gc.collect()
fp("after save_file")
m = torch.nn.Module(); m.w = torch.nn.Parameter(torch.zeros(n), requires_grad=False)
fp("after creating 512MB param (zeros, touched)")
sd = load_file(path, device="cpu")
fp("after load_file cpu (mmap, untouched)")
m.load_state_dict(sd, strict=True)      # what the strict loader / safetensors.load_model do
fp("after load_state_dict (copies file->param)")
del sd; gc.collect()
fp("after del state_dict")
print("param checksum", float(m.w[::65536].sum()))
os.remove(path)
