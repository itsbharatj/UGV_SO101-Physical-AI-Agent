# Time GR00T preprocessing and action-chunk prediction on realistic 480x640 inputs.
import time, torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors

path = "/root/models/groot"
cfg = PreTrainedConfig.from_pretrained(path); cfg.pretrained_path = path; cfg.device = "cuda"
policy = get_policy_class("groot").from_pretrained(path, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})
dtypes = {}
for p in policy.parameters(): dtypes[p.dtype] = dtypes.get(p.dtype, 0) + p.numel()
print("param dtypes:", {str(k): f"{v/1e9:.2f}B" for k, v in dtypes.items()})

def obs():
    return {"observation.state": torch.rand(1, 6) * 50,
            "observation.images.top cam": torch.rand(1, 3, 480, 640),
            "observation.images.wrist cam": torch.rand(1, 3, 480, 640),
            "task": ["pickup jenga and place inside the white box"]}

for label, ctx in [("as saved", torch.autocast("cuda", enabled=False)), ("bf16 autocast", torch.autocast("cuda", dtype=torch.bfloat16))]:
    for i in range(4):
        policy.reset(); torch.cuda.synchronize(); t0 = time.time()
        with torch.inference_mode(), ctx:
            b = pre(obs()); torch.cuda.synchronize(); t1 = time.time()
            a = policy.predict_action_chunk(b); torch.cuda.synchronize(); t2 = time.time()
        print(f"{label} run {i}: preprocess {1000*(t1-t0):.0f} ms, predict {1000*(t2-t1):.0f} ms, chunk {tuple(a.shape)}")
