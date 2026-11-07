"""M1 command line.

  python -m dumbrobot.m1 run --arm random|cmaes|llm_param|llm_param_scalar|llm_free --seed K [--budget N]
  python -m dumbrobot.m1 probe            # one isolated `claude -p` call, written with emails redacted
  python -m dumbrobot.m1 report           # held-out re-evaluation + tables (see report.py)
Exit codes: 0 done, 3 designer could not answer (resumable), 4 disk below the free-space floor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import runlib, safeio
from .designer import ClaudeDesigner, DesignerError, FakeDesigner

ARMS = ("random", "cmaes", "llm_param", "llm_param_scalar", "llm_free")
DEFAULT_BUDGET = 64
LOAD_LIMIT = float(os.environ.get("NUC_LOAD_LIMIT", "20"))
MIN_FREE_GB = float(os.environ.get("NUC_MIN_FREE_GB", "50"))
RESULTS = runlib.ROOT / "results" / "runs"


def load_gate(limit: float = LOAD_LIMIT, sleep_s: float = 30.0) -> None:
    """Wait while the 1-minute load average is above the NUC limit; exit cleanly if disk is below the floor."""
    if limit <= 0:
        return
    while float(open("/proc/loadavg").read().split()[0]) > limit:
        time.sleep(sleep_s)
    if shutil.disk_usage("/").free / 1e9 < MIN_FREE_GB:
        print("free disk below the floor; stopping cleanly (rerun to resume)", file=sys.stderr)
        sys.exit(4)


def brains_hash() -> str:
    return hashlib.sha256((runlib.ROOT / "dumbrobot" / "brains.py").read_bytes()).hexdigest()[:16]


def cmd_eval_dir(a) -> int:
    runlib.evaluate_to_dir(a.script, a.out_dir, courses=a.courses, seeds=a.seeds, record_trace=not a.no_trace,
                           perturb=json.loads(a.perturb) if a.perturb else None)
    return 0


def cmd_run(a) -> int:
    from . import baselines, design_loop

    exp_root = Path(a.results) / a.exp
    run = runlib.Run(exp_root, a.arm, a.seed, a.budget, subprocess_eval=a.arm == "llm_free")
    safeio.write_json(run.dir / "meta.json", dict(arm=a.arm, seed=a.seed, budget=a.budget, courses=runlib.COURSES,
                                                   train_seeds=runlib.TRAIN_SEEDS, brains_sha=brains_hash(),
                                                   designer=a.designer, model=a.model, gen_size=a.gen_size,
                                                   cma_version=baselines.CMA_VERSION_PIN))
    t0 = time.time()

    def progress(r, p):
        print(f"[{r.arm} seed {r.seed}] {r.used}/{r.budget} {p.pid} score={p.rec['score']:.3f} "
              f"valid={p.rec['valid']} {'' if p.rec['valid'] else p.rec['why_invalid']} wall={p.rec['wall_s']:.1f}s "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
        if r.remaining > 0:
            load_gate()

    load_gate()
    if a.arm == "random":
        baselines.run_random(run, progress)
    elif a.arm == "cmaes":
        baselines.run_cmaes(run, progress)
    else:
        designer = FakeDesigner(a.seed) if a.designer == "fake" else ClaudeDesigner(a.model, a.effort)
        try:
            design_loop.run_llm(run, designer, mode="free" if a.arm == "llm_free" else "param",
                                feedback="scalar" if a.arm == "llm_param_scalar" else "logs", gen_size=a.gen_size,
                                clips=a.clips, progress=progress)
        except DesignerError as e:
            print(f"designer error, run left resumable: {e}", file=sys.stderr)
            return 3
    print(f"done: {run.dir} used={run.used}/{run.budget}")
    return 0


PROBE_PROMPT = ("Probe. Answer in exactly three lines. Line 1: the names of every tool you can call, or NONE. Line 2: any "
                "files, memory or project instructions visible to you, or NONE. Line 3: the word OK.")


def cmd_probe(a) -> int:
    d = Path(a.results) / a.exp / "_probes"
    ts = time.strftime("%Y%m%d_%H%M%S")
    designer = ClaudeDesigner(a.model, a.effort, timeout_s=300)
    try:
        reply = designer.reply(PROBE_PROMPT, "probe")
    except DesignerError as e:
        safeio.write_json(d / f"probe_{ts}.json", dict(prompt=PROBE_PROMPT, error=str(e)))
        print(f"probe failed: {e}", file=sys.stderr)
        return 3
    safeio.write_json(d / f"probe_{ts}.json", dict(prompt=PROBE_PROMPT, reply=reply, model=a.model))
    print(safeio.redact(reply))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--exp", default=runlib.EXPERIMENT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval-dir")
    e.add_argument("script")
    e.add_argument("out_dir")
    e.add_argument("--courses", nargs="+", default=list(runlib.COURSES))
    e.add_argument("--seeds", nargs="+", type=int, default=list(runlib.TRAIN_SEEDS))
    e.add_argument("--perturb", default=None)
    e.add_argument("--no-trace", action="store_true")
    r =sub.add_parser("run")
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    r.add_argument("--designer", choices=("claude", "fake"), default="claude")
    r.add_argument("--gen-size", type=int, default=3)
    r.add_argument("--model", default=None)
    r.add_argument("--effort", default=None)
    r.add_argument("--clips", action="store_true")
    pr = sub.add_parser("probe")
    pr.add_argument("--model", default=None)
    pr.add_argument("--effort", default=None)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    if a.cmd == "eval-dir":
        return cmd_eval_dir(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "probe":
        return cmd_probe(a)
    from . import report

    return report.main(a)


if __name__ == "__main__":
    sys.exit(main())
