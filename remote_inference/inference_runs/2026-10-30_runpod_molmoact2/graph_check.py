"""On-pod sanity check for the served MolmoAct2 checkpoint (run with the policy server stopped).

1. Loads the policy exactly like lerobot-policy-server (deployment.py) does.
2. Builds one observation from two training frames + the training start pose, like PolicyRunner._batch.
3. Predicts the chunk 3x with the flow-matching CUDA graph enabled and 3x with it disabled, same noise:
   replays must be identical, graph vs eager must agree to ~1e-3 (normalized units); prints the
   unnormalized chunk so it can be compared with a demonstration start.
Usage: python graph_check.py <model_dir> <top.jpg> <wrist.jpg>
"""

import sys
import time

import numpy as np
import torch
from PIL import Image

from lerobot.policies import get_policy_class, make_pre_post_processors

try:
    from lerobot.configs.policies import PreTrainedConfig
except ImportError:  # older layout
    from lerobot.policies.pretrained import PreTrainedConfig

path, top_path, wrist_path = sys.argv[1], sys.argv[2], sys.argv[3]
np.set_printoptions(precision=2, suppress=True, linewidth=160)
torch.set_printoptions(precision=3, sci_mode=False, linewidth=160)

t0 = time.time()
cfg = PreTrainedConfig.from_pretrained(path)
cfg.pretrained_path = path
cfg.device = "cuda"
policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(
    policy_cfg=cfg, pretrained_path=path,
    preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": {}}},
)
print(f"loaded in {time.time() - t0:.0f}s; chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
      f"cuda_graph={getattr(cfg, 'enable_inference_cuda_graph', None)} max_action_dim={getattr(cfg, 'expected_max_action_dim', None)}")


def load(p):
    return np.asarray(Image.open(p).convert("RGB").resize((640, 480)), dtype=np.uint8)


state = np.array([-1.37, -100.31, 96.28, 55.08, -3.58, 1.26], dtype=np.float32)  # mean training start pose
batch = {
    "observation.state": torch.from_numpy(state).unsqueeze(0),
    "observation.images.top cam": torch.from_numpy(load(top_path)).permute(2, 0, 1).contiguous().float().div_(255).unsqueeze(0),
    "observation.images.wrist cam": torch.from_numpy(load(wrist_path)).permute(2, 0, 1).contiguous().float().div_(255).unsqueeze(0),
    "task": ["pickup jenga and place inside the white box"],
    "robot_type": "so101_follower",
}
max_dim = int(getattr(cfg, "expected_max_action_dim", 32) or 32)
noise = torch.randn(1, cfg.chunk_size, max_dim, generator=torch.Generator().manual_seed(0)).cuda()


def run(label, n=3):
    outs, times = [], []
    for _ in range(n):
        prepared = pre({k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()})
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            a = policy.predict_action_chunk(prepared, noise=noise.clone())
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
        outs.append(a.detach().float().cpu())
    d_replay = max((outs[i] - outs[0]).abs().max().item() for i in range(1, n))
    print(f"{label}: times {[round(x, 3) for x in times]} s; shape {tuple(outs[0].shape)}; "
          f"replay max|diff| {d_replay:.2e}; finite={torch.isfinite(outs[0]).all().item()}")
    return outs[0]


policy._set_inference_cuda_graph_enabled(True)
graph = run("cuda graph ON ")
policy._set_inference_cuda_graph_enabled(False)
eager = run("cuda graph OFF")
print(f"graph vs eager max|diff| = {(graph - eager).abs().max().item():.2e} (normalized units; expect ~1e-3 or less)")

try:
    un = post(graph.cuda())
    un = un.detach().float().cpu().numpy()[0]
    print("unnormalized chunk (deg / gripper %), rows = steps 0,5,...,29:")
    for i in range(0, un.shape[0], 5):
        print(f"  step {i:2d} {un[i]}   delta vs state {un[i] - state}")
    print("  step 29", un[-1])
    print("per-step |delta| max over chunk:", np.abs(np.diff(un, axis=0)).max(0))
except Exception as e:  # noqa: BLE001
    print("postprocess failed:", type(e).__name__, e)
print("CHECK_DONE")
