"""Reproduce the paper's Fig.4: original vs recovered frame, with zoomed crops.

Runs the trained EDSR on one LR/HR pair and writes a side-by-side panel
(bicubic-upscaled LD | RTSR output | GT), each annotated with its PSNR, plus a
2x nearest-neighbour zoom strip.  The crop is chosen automatically as the region
with the most detail (highest Laplacian variance) unless --crop is given.

    emt-env/bin/python tools/make_fig4.py --exp ../experiment_t2_4k_b3000 \
        --data dataset/4k_b3000/DIV2K --frame 00420 --out results/fig4_snapshot.png
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))


def build(blocks, feats, scale):
    from importlib import import_module

    class _A(object):
        pass

    a = _A()
    a.model = 'EDSR'
    a.n_resblocks = blocks
    a.n_feats = feats
    a.scale = [scale]
    a.n_colors = 3
    a.rgb_range = 255
    a.res_scale = 1
    a.shift_mean = True
    return import_module('model.edsr').make_model(a)


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def auto_crop(hr, w, h, step=64):
    gray = cv2.cvtColor(hr, cv2.COLOR_BGR2GRAY)
    best, best_xy = -1.0, (hr.shape[1] // 2 - w // 2, hr.shape[0] // 2 - h // 2)
    for y in range(0, hr.shape[0] - h, step):
        for x in range(0, hr.shape[1] - w, step):
            v = cv2.Laplacian(gray[y:y + h, x:x + w], cv2.CV_64F).var()
            if v > best:
                best, best_xy = v, (x, y)
    return best_xy[0], best_xy[1], w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--exp', default='../experiment_t2_4k_b3000')
    ap.add_argument('--data', default='dataset/4k_b3000/DIV2K')
    ap.add_argument('--frame', default='00420')
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--feats', type=int, default=26)
    ap.add_argument('--crop', nargs=4, type=int, default=None,
                    help='x y w h in the 2160p frame (default: auto)')
    ap.add_argument('--out', default='results/fig4_snapshot.png')
    args = ap.parse_args()

    lr_path = os.path.join(args.data, 'DIV2K_train_LR_bicubic/X2/{}x2.png'.format(args.frame))
    hr_path = os.path.join(args.data, 'DIV2K_train_HR/{}.png'.format(args.frame))
    lr_bgr = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    hr_bgr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    if lr_bgr is None or hr_bgr is None:
        raise SystemExit('missing frames: {} / {}'.format(lr_path, hr_path))

    net = build(args.blocks, args.feats, 2).cuda().eval()
    ckpt = os.path.join(HERE, args.exp, 'model', 'model_best.pt')
    if os.path.isfile(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False), strict=False)

    lr_rgb = np.ascontiguousarray(lr_bgr[:, :, ::-1].transpose(2, 0, 1))
    x = torch.from_numpy(lr_rgb).float().unsqueeze(0).cuda()
    with torch.no_grad():
        y = net(x)
    sr = y.clamp(0, 255).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1]

    lr_up = cv2.resize(lr_bgr, (hr_bgr.shape[1], hr_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)

    if args.crop:
        cx, cy, w, h = args.crop
    else:
        cx, cy, w, h = auto_crop(hr_bgr, 640, 360)
    print('crop x={} y={} w={} h={}'.format(cx, cy, w, h))

    def cut(img):
        return img[cy:cy + h, cx:cx + w]

    p_lr, p_sr = psnr(cut(lr_up), cut(hr_bgr)), psnr(cut(sr), cut(hr_bgr))
    tiles = [label(cut(lr_up), 'LD bicubic  {:.2f} dB'.format(p_lr)),
             label(cut(sr), 'RTSR M2  {:.2f} dB'.format(p_sr)),
             label(cut(hr_bgr), 'Original 2160p')]
    panel = np.hstack(tiles)
    zoom = np.hstack([cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2),
                                 interpolation=cv2.INTER_NEAREST) for t in tiles])

    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    cv2.imwrite(os.path.join(ROOT, args.out), panel)
    cv2.imwrite(os.path.join(ROOT, 'results/fig4_zoom.png'), zoom)
    print('full-frame PSNR: bicubic {:.3f} dB | RTSR {:.3f} dB'.format(
        psnr(lr_up, hr_bgr), psnr(sr, hr_bgr)))
    print('written:', os.path.join(ROOT, args.out))


if __name__ == '__main__':
    main()