# Empirical check: does safetensors.torch.load_file(device="cpu") copy tensor bytes into
# anonymous memory, or return views over a file-backed mapping?
import os, resource, sys, gc, time
import torch
from safetensors.torch import save_file, load_file

path = os.path.join(sys.argv[1], "st_mmap_test.safetensors")
n = 128 * 1024 * 1024  # 128M fp32 = 512 MB
save_file({"w": torch.randn(n)}, path)
gc.collect()

def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024

def footprint_mb():
    try:
        import psutil
        p = psutil.Process()
        mi = p.memory_full_info()
        return {"rss": mi.rss / 2**20, "uss": getattr(mi, "uss", float("nan")) / 2**20}
    except Exception as e:
        return {"err": repr(e)}

print("file size MB", os.path.getsize(path) / 2**20)
print("before load:", footprint_mb(), "maxrss", round(rss_mb()))
sd = load_file(path, device="cpu")
t = sd["w"]
print("after load (untouched):", footprint_mb(), "maxrss", round(rss_mb()))
st = t.untyped_storage()
print("storage nbytes MB", st.nbytes() / 2**20, "storage.filename", getattr(st, "filename", None), "is_shared", t.is_shared())
s = float(t[::4096].sum())  # touch every page
print("after touching pages:", footprint_mb(), "maxrss", round(rss_mb()))
dst = torch.empty_like(t)
dst.copy_(t)  # what load_state_dict does
print("after copy_ into a fresh param:", footprint_mb(), "maxrss", round(rss_mb()))
del sd, t, st
gc.collect()
print("after del state_dict:", footprint_mb(), "maxrss", round(rss_mb()))
os.remove(path)
