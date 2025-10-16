#!/usr/bin/env python3
import argparse, os, re, glob, math, csv, json
from pathlib import Path
from typing import Optional

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch compute per-column max/min consecutive diffs (n+1 - n) across episodes."
    )
    p.add_argument("root", help="Root directory containing demo_* subfolders")
    p.add_argument("--glob", default="demo_*/end_effector_pose_right_arm.txt",
                   help="Glob (relative to root) for files to analyze (default: demo_*/end_effector_pose_right_arm.txt)")
    p.add_argument("--cols", default="2,3,4", help="1-based column indices to analyze (default: 2,3,4)")
    p.add_argument("--delimiter", default=None, help="Explicit delimiter (default: any whitespace)")
    p.add_argument("--out_csv", default="column_diff_stats.csv", help="Output CSV path (default: column_diff_stats.csv)")
    p.add_argument("--out_json", default="column_diff_stats.json", help="Output JSON path (default: column_diff_stats.json)")
    p.add_argument("--abs", action="store_true", help="Use absolute differences |n+1 - n| for extrema")
    return p.parse_args()

def load_rows(path: str, delimiter: Optional[str]):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split() if delimiter is None else s.split(delimiter)
            try:
                row = [float(x) for x in parts]
            except ValueError:
                # skip non-numeric lines
                continue
            rows.append(row)
    return rows

def stats_for_column(rows, col_idx: int, use_abs: bool):
    series = [r[col_idx] for r in rows if len(r) > col_idx]
    if len(series) < 2:
        return None
    # time column if present in col 0 (same length)
    timecol = [r[0] for r in rows] if all(len(r) > 0 for r in rows) else None

    def diff(i):
        return series[i+1] - series[i]

    best_max = -math.inf
    best_min = math.inf
    best_max_i = None
    best_min_i = None
    for i in range(len(series) - 1):
        d = diff(i)
        key = abs(d) if use_abs else d
        if key > best_max:
            best_max = key
            best_max_i = i
        if key < best_min:
            best_min = key
            best_min_i = i

    # Prepare record
    def pack(i, key):
        rec = {
            "index_n": i,
            "index_n1": i + 1,
            "n_val": series[i],
            "n1_val": series[i + 1],
            "diff": series[i + 1] - series[i],
            "score": key,
        }
        if timecol is not None and len(timecol) == len(rows):
            rec["n_time"] = timecol[i]
            rec["n1_time"] = timecol[i + 1]
        return rec

    return {
        "count": len(series),
        "num_diffs": len(series) - 1,
        "extreme_max": pack(best_max_i, best_max),
        "extreme_min": pack(best_min_i, best_min),
        "used_absolute": use_abs,
    }

def extract_episode_id(path: str) -> str:
    m = re.search(r"(?:^|/)demo_(\d+)(?:/|$)", path.replace("\\", "/"))
    return m.group(1) if m else ""

def main():
    args = parse_args()
    root = Path(args.root)
    rel_glob = args.glob
    paths = sorted(root.glob(rel_glob)) if "**" in rel_glob else sorted(Path(p) for p in glob.glob(str(root / rel_glob)))
    cols = [int(x.strip()) for x in args.cols.split(",") if x.strip()]
    col_idxs = [c - 1 for c in cols]

    rows_out = []
    for p in paths:
        if not p.is_file():
            continue
        rows = load_rows(str(p), args.delimiter)
        if len(rows) < 2:
            continue
        ep = extract_episode_id(str(p))
        for c, ci in zip(cols, col_idxs):
            st = stats_for_column(rows, ci, args.abs)
            if st is None:
                continue
            row = {
                "episode": ep,
                "file": str(p),
                "column": c,
                "rows": st["count"],
                "diffs": st["num_diffs"],
                "used_absolute": st["used_absolute"],
                "max_index_n": st["extreme_max"]["index_n"],
                "max_index_n1": st["extreme_max"]["index_n1"],
                "max_n_val": st["extreme_max"]["n_val"],
                "max_n1_val": st["extreme_max"]["n1_val"],
                "max_diff": st["extreme_max"]["diff"],
                "max_score": st["extreme_max"]["score"],
                "max_n_time": st["extreme_max"].get("n_time"),
                "max_n1_time": st["extreme_max"].get("n1_time"),
                "min_index_n": st["extreme_min"]["index_n"],
                "min_index_n1": st["extreme_min"]["index_n1"],
                "min_n_val": st["extreme_min"]["n_val"],
                "min_n1_val": st["extreme_min"]["n1_val"],
                "min_diff": st["extreme_min"]["diff"],
                "min_score": st["extreme_min"]["score"],
                "min_n_time": st["extreme_min"].get("n_time"),
                "min_n1_time": st["extreme_min"].get("n1_time"),
            }
            rows_out.append(row)

    # Write CSV / JSON
    if rows_out:
        fieldnames = list(rows_out[0].keys())
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows_out:
                w.writerow(r)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(rows_out, f, indent=2)
        print(f"Wrote {len(rows_out)} rows to {args.out_csv} and {args.out_json}")
    else:
        print("No results. Check your --root and --glob.")
if __name__ == "__main__":
    main()