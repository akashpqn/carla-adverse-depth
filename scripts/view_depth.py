"""
Render a depth PNG as a viewable colour image.

The dataset's depth maps are 16-bit PNGs holding CENTIMETRES (value / 100 = metres). Opened in an
ordinary image viewer they look almost black: the viewer stretches 0..65535 to black..white, but a
120 m scene only reaches 12000, and unmeasured pixels (sky, beyond range, between LiDAR scan lines)
are 0. This rescales to the actual depth range and colours it - near is red, far is blue, no-data
stays black.

    python view_depth.py <depth.png> [output.png] [--max-depth 80]
    python view_depth.py <folder> --all          # every depth map in a folder
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image


def colourise(depth_cm, max_depth_m, cmap_name="turbo"):
    """
    Depth -> colour image. Unmeasured pixels (0) stay black so they read as 'no data' rather than
    'zero distance'. Depth is mapped on a square-root scale: most pixels are close, so a linear ramp
    crushes all the near detail into one colour.
    """
    depth_m = depth_cm.astype(np.float32) / 100.0
    valid = depth_m > 0
    scaled = np.sqrt(np.clip(depth_m, 0, max_depth_m) / max_depth_m)

    try:
        from matplotlib import cm
        lut = (cm.get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:  # matplotlib absent - fall back to a plain near-red / far-blue ramp
        ramp = np.linspace(0, 1, 256)
        lut = np.stack([255 * (1 - ramp), 255 * ramp * 0.6, 255 * ramp], axis=1).astype(np.uint8)

    idx = np.clip((scaled * 255).astype(np.int32), 0, 255)
    out = np.zeros(depth_m.shape + (3,), dtype=np.uint8)
    out[valid] = lut[idx[valid]]
    return Image.fromarray(out)


def convert(path, out_path, max_depth_m, cmap="turbo"):
    depth_cm = np.array(Image.open(path))
    colourise(depth_cm, max_depth_m, cmap).save(out_path)
    valid = depth_cm[depth_cm > 0]
    coverage = 100.0 * valid.size / depth_cm.size
    span = "%.1f-%.1f m" % (valid.min() / 100.0, valid.max() / 100.0) if valid.size else "no data"
    print("%s -> %s  (%.1f%% measured, %s)" % (os.path.basename(path), out_path, coverage, span))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="depth PNG, or a folder with --all")
    parser.add_argument("output", nargs="?", help="output image (default: alongside, _view.png)")
    parser.add_argument("--all", action="store_true", help="convert every PNG in the folder")
    parser.add_argument("--max-depth", default=200.0, type=float, help="metres mapped to the far colour")
    parser.add_argument("--cmap", default="turbo", help="matplotlib colourmap name (turbo, viridis, magma, ...)")
    args = parser.parse_args()

    if args.all:
        out_dir = args.output or os.path.join(args.path, "_view")
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(args.path, "*.png"))):
            convert(f, os.path.join(out_dir, os.path.basename(f)), args.max_depth, args.cmap)
    else:
        out = args.output or os.path.splitext(args.path)[0] + "_view.png"
        convert(args.path, out, args.max_depth, args.cmap)


if __name__ == "__main__":
    main()
