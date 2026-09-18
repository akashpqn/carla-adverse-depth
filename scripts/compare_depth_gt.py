"""
Compare the depth ground-truth sources in a captured set.

Reports, per source: coverage, min/max/percentile range, how much sits at the clipping limit, and -
where two sources measure the same pixel - how closely they agree.

    python compare_depth_gt.py <town/weather folder> [--camera front] [--limit 30]
"""

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

SOURCES = ["carla_depth_u16", "sparse_depth_u16", "semi_dense_depth_u16"]


def load_m(path):
    """16-bit PNG in centimetres -> float32 metres, 0 = no measurement."""
    return np.array(Image.open(path)).astype(np.float32) / 100.0


def describe(name, values, total_px, max_depth):  # total_px = pixels across all sampled frames
    if values.size == 0:
        print("  %-22s no measured pixels" % name)
        return
    at_limit = 100.0 * (values >= max_depth - 0.05).mean()
    print("  %-22s coverage %5.1f%% | min %6.2f m | p50 %6.2f m | p99 %6.2f m | max %6.2f m | at clip %4.1f%%"
          % (name, 100.0 * values.size / total_px, values.min(), np.percentile(values, 50),
             np.percentile(values, 99), values.max(), at_limit))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--camera", default="front")
    ap.add_argument("--limit", default=30, type=int, help="frames to sample")
    ap.add_argument("--max-depth", default=200.0, type=float, help="the capture's clip distance")
    args = ap.parse_args()

    meta_files = sorted(glob.glob(os.path.join(args.folder, "metadata", "*.json")))
    if not meta_files:
        raise SystemExit("no frames in %s" % args.folder)
    meta = json.load(open(meta_files[0], encoding="utf-8"))
    max_depth = args.max_depth  # the capture's --max-depth; every source is clipped to it
    print("set: %s" % args.folder)
    print("frames: %d (sampling %d), camera: %s, weather: %s"
          % (len(meta_files), min(args.limit, len(meta_files)), args.camera, meta["weather"]))
    print("intrinsics: %dx%d, fov %.0f deg | LiDAR range %s m"
          % (meta["intrinsics"]["width"], meta["intrinsics"]["height"], meta["intrinsics"]["fov"],
             meta.get("lidar_profile", "?")))
    print()

    stems = [os.path.splitext(os.path.basename(f))[0] for f in meta_files][: args.limit]
    pooled = {s: [] for s in SOURCES}
    frame_px = 0
    per_frame_agreement = []

    for stem in stems:
        frame = {}
        for src in SOURCES:
            path = os.path.join(args.folder, src, "%s_%s.png" % (stem, args.camera))
            if os.path.exists(path):
                d = load_m(path)
                frame[src] = d
                pooled[src].append(d[d > 0])
                frame_px = d.size
        if "carla_depth_u16" in frame and "sparse_depth_u16" in frame:
            a, b = frame["carla_depth_u16"], frame["sparse_depth_u16"]
            both = (a > 0) & (b > 0)
            if both.sum():
                per_frame_agreement.append(np.abs(a[both] - b[both]))

    print("RANGE AND COVERAGE (pooled over sampled frames)")
    for src in SOURCES:
        vals = np.concatenate(pooled[src]) if pooled[src] else np.array([])
        describe(src, vals, frame_px * max(1, len(pooled[src])), max_depth)

    if per_frame_agreement:
        diff = np.concatenate(per_frame_agreement)
        print()
        print("AGREEMENT, dense CARLA depth vs sparse LiDAR (pixels measured by both)")
        print("  compared pixels : %d" % diff.size)
        print("  mean abs diff   : %.3f m" % diff.mean())
        print("  median abs diff : %.3f m" % np.median(diff))
        print("  95th percentile : %.3f m" % np.percentile(diff, 95))
        print("  max abs diff    : %.3f m" % diff.max())
        print("  within 0.10 m   : %.2f%%" % (100.0 * (diff < 0.10).mean()))
        print("  beyond 1.00 m   : %.2f%%  (occlusion-boundary disagreements)" % (100.0 * (diff > 1.0).mean()))


if __name__ == "__main__":
    main()
