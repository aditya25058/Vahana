import os, sys, time, random
D = sys.argv[1]
fs = sorted(os.path.join(D, f) for f in os.listdir(D) if f.endswith(".safetensors"))
# pick a shard region far from recent use; read 3 GB in 9.4 MB chunks at random offsets (expert-sized), then sequential
f = fs[len(fs) // 2]
sz = os.path.getsize(f)
CH = 9_437_184
for mode in ("random 9.4MB", "sequential 64MB"):
    fd = os.open(f, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    t = time.perf_counter(); n = 0
    offs = [random.randrange(0, sz - CH) for _ in range(300)] if mode.startswith("random") else list(range(0, min(sz, 3 << 30), 64 << 20))
    for o in offs:
        os.lseek(fd, o, 0)
        n += len(os.read(fd, CH if mode.startswith("random") else 64 << 20))
    dt = time.perf_counter() - t
    os.close(fd)
    print(mode, round(n / dt / 2**30, 2), "GB/s over", round(n / 2**30, 1), "GB", flush=True)
    f = fs[len(fs) // 2 + 1]; sz = os.path.getsize(f)
