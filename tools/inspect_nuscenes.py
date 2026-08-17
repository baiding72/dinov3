#!/usr/bin/env python3
"""Inspect downloaded nuScenes-family mini datasets with the official devkit.

Usage:
    PYTHONPATH=tools/nuscenes-devkit/python-sdk \
        /Users/baiding/miniconda3/envs/nuscenes/bin/python tools/inspect_nuscenes.py

Prints dataset stats for nuScenes v1.0-mini and nuImages v1.0-mini,
and optionally the SQLite table list of a nuPlan log database.
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

try:
    from nuscenes.nuscenes import NuScenes
    from nuimages.nuimages import NuImages
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Run with PYTHONPATH pointing at the devkit python-sdk folder.", file=sys.stderr)
    sys.exit(2)


def inspect_nuscenes(dataroot: str) -> None:
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    print("== nuScenes v1.0-mini ==")
    print("tables:", sorted(nusc.table_names))
    for t in ["scene", "sample", "sample_data", "sample_annotation", "category", "instance", "attribute", "map"]:
        print(f"  {t}: {len(getattr(nusc, t))}")
    chans = Counter(sd["channel"] for sd in nusc.sample_data)
    print("  sample_data channels:", dict(chans))
    mods = Counter(sd["sensor_modality"] for sd in nusc.sample_data)
    print("  sample_data modalities:", dict(mods))
    cats = Counter(c["name"].split(".")[0] for c in nusc.category)
    print("  top-level categories:", dict(cats))


def inspect_nuimages(dataroot: str) -> None:
    nuim = NuImages(version="v1.0-mini", dataroot=dataroot, verbose=False)
    print("== nuImages v1.0-mini ==")
    print("tables:", sorted(nuim.table_names))
    for t in ["log", "sample", "sample_data", "object_ann", "surface_ann", "category", "attribute", "sensor"]:
        print(f"  {t}: {len(getattr(nuim, t))}")
    sens = {s["token"]: s for s in nuim.sensor}
    cals = {c["token"]: c for c in nuim.calibrated_sensor}
    chans = Counter()
    for sd in nuim.sample_data:
        cal = cals[sd["calibrated_sensor_token"]]
        chans[sens[cal["sensor_token"]]["channel"]] += 1
    print("  sample_data channels:", dict(chans))
    keys = [sd for sd in nuim.sample_data if sd["is_key_frame"]]
    print(f"  keyframes: {len(keys)}, sweeps: {len(nuim.sample_data) - len(keys)}")


def inspect_nuplan_db(db_path: str) -> None:
    print("== nuPlan log database ==")
    if not os.path.exists(db_path):
        print(f"  file not found: {db_path}")
        return
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    print(f"  tables ({len(tables)}):", tables)
    for t in tables:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"    {t}: {n} rows")
        except sqlite3.Error as e:
            print(f"    {t}: <error: {e}>")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes-root", default=os.path.expanduser("~/nuscenes"))
    ap.add_argument("--nuimages-root", default=os.path.expanduser("~/nuimages"))
    ap.add_argument("--nuplan-db", default="")
    args = ap.parse_args()
    inspect_nuscenes(args.nuscenes_root)
    print()
    inspect_nuimages(args.nuimages_root)
    if args.nuplan_db:
        print()
        inspect_nuplan_db(args.nuplan_db)


if __name__ == "__main__":
    main()
