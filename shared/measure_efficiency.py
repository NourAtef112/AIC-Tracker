import torch, time, os
from thop import profile

def measure_all(model, template_size=128, search_size=256, n_runs=200):
    model.eval().cuda()
    t = torch.randn(1, 3, template_size, template_size).cuda()
    s = torch.randn(1, 3, search_size,   search_size).cuda()

    # FLOPs & Params
    flops, params = profile(model, inputs=(t, s), verbose=False)
    print(f"FLOPs  : {flops/1e9:.2f} GFLOPs  (budget: 30)")
    print(f"Params : {params/1e6:.2f} M        (budget: 50M)")

    # Latency – warm up first
    for _ in range(20):
        with torch.no_grad():
            model(t, s)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(t, s)
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start) / n_runs * 1000
    print(f"Latency: {latency_ms:.2f} ms       (budget: 30ms)")

    # Model size
    torch.save(model.state_dict(), "_tmp.pth")
    size_gb = os.path.getsize("_tmp.pth") / 1e9
    os.remove("_tmp.pth")
    print(f"Size   : {size_gb:.4f} GB       (budget: 0.5GB)")

    return flops/1e9, params/1e6, latency_ms, size_gb
