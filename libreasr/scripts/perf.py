from libreasr.lib.imports import *

# load config
cfgs = ["./config/base.yaml"]

# load everything
conf, lang, builder_train, builder_valid, db, m, learn = parse_and_apply_config(
    path=cfgs
)
(conf, lang, db, m, learn)

# move model to gpu
dev = conf["cuda"]["device"]
m = m.to(dev)


# generate useful input data
def gen_data(N=32, T=320000, U=50, V=4096, to="cpu"):
    x = torch.randn(N, T, 1, 1)
    xl = torch.randint(T // 2, T, (N,)).long()
    y = torch.randint(0, V, (N, U)).long()
    yl = torch.randint(U // 2, U, (N,)).long()
    (x, xl, y, yl) = map(lambda x: x.to(to), (x, xl, y, yl))
    return (x, y, xl, yl)


# perform one full learning step
def step(opt, N=1, T=16000 * 20):
    try:
        m.train()
        tpl = gen_data(N=N, T=T, to=dev)
        loss = m(tpl)
        loss = loss["loss"].mean()
        loss.backward()
        opt.step()
        print(f"step N={N} T={T}")
        return True
    except:
        print(f"step N={N} T={T} OOM")
        return False


# profile everything
def profile():

    # store
    profs = []

    # create opt
    opt = torch.optim.Adam(m.parameters(), lr=3e-4)

    # dry runs
    for i in range(5):
        step(opt)

    # measure
    for N in range(1, 24):
        for T in map(lambda x: x * 16000, [0.5, 1.0, 2.0, 4.0, 8.0]):
            T = int(T)
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                step(opt, N=N, T=T)
            profs.append((N, T, prof))

    print(profs[-3][-1])

    # print
    # for N, T, prof in profs[::-1]:
    #     print()
    #     print(N, T)
    #     print(prof)


profile()
