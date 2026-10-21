"""Measure single-frame inference latency of the RTSR EDSR variants.

    emt-env/bin/python tools/bench_latency.py --exp ../experiment_rtsr_m2_26feats \
        --blocks 8 --feats 26 --lr 960 540

Reports mean/median latency, FPS and peak GPU memory, so the paper's
"real-time" claim can be checked against 30/60/90 FPS budgets.
"""

import argparse
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))


def build(blocks, feats, scale, colors=3, rgb_range=255):
    from importlib import import_module

    class _A(object):
        pass

    a = _A()
    a.model = 'EDSR'
    a.n_resblocks = blocks
    a.n_feats = feats
    a.scale = [scale]
    a.n_colors = colors
    a.rgb_range = rgb_range
    a.res_scale = 1
    a.shift_mean = True
    return import_module('model.edsr').make_model(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp', required=True, help='experiment dir holding model/model_best.pt')
    ap.add_argument('--blocks', type=int, required=True)
    ap.add_argument('--feats', type=int, required=True)
    ap.add_argument('--scale', type=int, default=2)
    ap.add_argument('--lr', nargs=2, type=int, required=True, metavar=('W', 'H'))
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu else 'cuda')
    net = build(args.blocks, args.feats, args.scale).to(device).eval()

    ckpt = os.path.join(args.exp, 'model', 'model_best.pt')
    if os.path.isfile(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
        loaded = 'yes'
    else:
        loaded = 'NO (random weights)'

    w, h = args.lr
    x = torch.rand(1, 3, h, w, device=device) * 255.0

    with torch.no_grad():
        for _ in range(args.warmup):
            net(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            net(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = statistics.mean(times)
    med_ms = statistics.median(times)
    peak = (torch.cuda.max_memory_allocated() / 1024 ** 2) if device.type == 'cuda' else 0.0
    print('model          blocks={} feats={} scale={}'.format(args.blocks, args.feats, args.scale))
    print('input/output   {}x{} -> {}x{}'.format(w, h, w * args.scale, h * args.scale))
    print('weights        {}'.format(loaded))
    print('latency        mean {:.1f} ms | median {:.1f} ms | p95 {:.1f} ms'.format(
        mean_ms, med_ms, sorted(times)[int(0.95 * len(times)) - 1]))
    print('throughput     {:.1f} FPS  (60 FPS budget = 16.7 ms)'.format(1000.0 / mean_ms))
    print('gpu peak       {:.0f} MiB'.format(peak))
    print('json           {{"blocks": {}, "feats": {}, "input": [{}, {}], "mean_ms": {:.2f}, "fps": {:.1f}}}'.format(
        args.blocks, args.feats, w, h, mean_ms, 1000.0 / mean_ms))


if __name__ == '__main__':
    main()