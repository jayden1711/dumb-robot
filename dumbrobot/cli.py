"""python -m dumbrobot.cli score <body_script.py> [--course bumps] [--seeds 0 1] [--brain A] [--out DIR]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate import evaluate_script


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("script")
    sc.add_argument("--course", nargs="+", default=["bumps"])
    sc.add_argument("--seeds", nargs="+", type=int, default=[0])
    sc.add_argument("--brain", default="A")
    sc.add_argument("--out", default=None, help="directory for score.json, failure_log.txt, trace.npz")
    a = ap.parse_args(argv)
    out = evaluate_script(a.script, courses=tuple(a.course), seeds=tuple(a.seeds), brain=a.brain,
                          record_trace=a.out is not None)
    if not out["valid"] and out["validity"]:
        print("INVALID:", "; ".join(f"{f['name']}: {f['detail']}" for f in out["validity"]))
    for cname, c in out["per_course"].items():
        for e in c["episodes"]:
            print(f"[{cname} seed={e['seed']}] score={e['score']['score']:.3f}\n{e['summary']}")
    print(f"mean score={out['score']:.3f} valid={out['valid']}")
    if a.out:
        d = Path(a.out)
        d.mkdir(parents=True, exist_ok=True)
        for c in out["per_course"].values():
            for i, e in enumerate(c["episodes"]):
                tr = e.pop("trace", None)
                if tr:
                    np.savez(d / f"trace_{i}.npz", **tr)
        (d / "score.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
