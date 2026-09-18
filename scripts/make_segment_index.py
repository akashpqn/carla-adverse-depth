"""
Write segments.csv into each <town>/<weather> folder of the dataset.

Every crash-recovery restart begins a new scene (the simulator state is lost), so a folder is a series
of continuous segments rather than one drive. Each frame's metadata records the scene it came from in
`run_seed`; this groups consecutive frames by that value.

    python make_segment_index.py [dataset_root]
"""

import csv
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else r"D:\adver_city_depth_dataset\dataset"

for folder in sorted(glob.glob(os.path.join(root, "*", "*"))):
    frames = sorted(glob.glob(os.path.join(folder, "metadata", "*.json")))
    if not frames:
        continue
    segments = []
    for path in frames:
        stem = os.path.splitext(os.path.basename(path))[0]
        seed = json.load(open(path, encoding="utf-8")).get("run_seed")
        if not segments or segments[-1]["run_seed"] != seed:
            segments.append({"run_seed": seed, "first_frame": stem, "last_frame": stem, "length": 0})
        segments[-1]["last_frame"] = stem
        segments[-1]["length"] += 1

    with open(os.path.join(folder, "segments.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["segment_id", "run_seed", "first_frame", "last_frame", "length"])
        writer.writeheader()
        for i, seg in enumerate(segments):
            writer.writerow(dict(segment_id=i, **seg))

    rel = os.path.relpath(folder, root)
    print("%-28s %4d frames, %3d segments" % (rel, len(frames), len(segments)))
