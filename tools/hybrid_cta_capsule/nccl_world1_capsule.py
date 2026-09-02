import os, torch, torch.distributed as dist
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
dist.init_process_group("nccl", rank=0, world_size=1)
x = torch.ones(1024, device="cuda")
y = dist.all_reduce(x)
torch.cuda.synchronize()
v = x[0].item()
print("ALLREDUCE_OK", v, flush=True)
assert v == 1.0
dist.destroy_process_group()
print("NCCL_WORLD1_PASS", flush=True)
