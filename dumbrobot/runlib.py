"""One run = one (arm, seed) directory. Every proposal is evaluated into its own directory (PLAN section 8), charged to
the ledger, and cached, so any run can be killed and resumed without re-spending budget."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import safeio
from .assembly import BuildError, load_script
from .evaluate import Ledger, evaluate

sys.dont_write_bytecode = True  # importing a proposal's body.py must not leave a .pyc inside the run directory

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "m1"
COURSES = ("bumps", "corridor")
TRAIN_SEEDS = (0, 1)  # start-yaw jitter seeds seen by every optimizer
HELD_OUT_SEEDS = (1000, 1001, 1002, 1003)  # never seen by any optimizer
POP_SIZE = 8
EVAL_TIMEOUT_S = 300.0

ALLOWED_IMPORTS = {"dumbrobot", "math", "numpy", "mujoco"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr"}


def genome_script(genome: dict) -> str:
    """Parametric-arm body script: the default template with this genome."""
    g = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v)
         for k, v in genome.items()}
    return ("from dumbrobot.body_template import build  # noqa: F401\n\nGENOME = " + json.dumps(g, indent=1) + "\n")


def check_source(source: str) -> str | None:
    """Static guard for designer-written scripts (not a sandbox: the subprocess wall-clock limit is the other half).
    Returns an error string or None."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]] if node.level == 0 else ["dumbrobot"]
        else:
            mods = []
        for m in mods:
            if m not in ALLOWED_IMPORTS:
                return f"import of '{m}' is not allowed (allowed: {sorted(ALLOWED_IMPORTS)})"
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or (node.id.startswith("__") and node.id != "__name__")):
            return f"use of '{node.id}' is not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"attribute '{node.attr}' is not allowed"
    return None


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()[:12]


# ---- evaluation into a directory ---------------------------------------------------------------------------------------
def _invalid(name: str, detail: str, t0: float) -> dict:
    return dict(valid=False, validity=[dict(name=name, detail=detail)], validity_s=time.time() - t0, score=0.0,
                per_course={}, sim_s=0.0, wall_s=time.time() - t0)


def write_outputs(out_dir: Path, out: dict) -> dict:
    """Write validity.json, score.json, failure_log.json, trace.npz (redacted text) and return the compact record."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traces, failure, per_course = {}, {}, {}
    for c, cv in out["per_course"].items():
        eps = []
        failure[c] = []
        for e in cv["episodes"]:
            tr = e.get("trace")
            if tr:
                for k, v in tr.items():
                    traces[f"{c}__{e['seed']}__{k}"] = np.asarray(v)
            eps.append(dict(seed=e["seed"], **e["score"]))
            failure[c].append(dict(seed=e["seed"], summary=e["summary"], segments=e["result"]["segments"],
                                   ending=e["result"]["ending"]))
        per_course[c] = dict(mean_score=cv["mean_score"], episodes=eps)
    rec = dict(score=out["score"], valid=out["valid"], why_invalid=[f["name"] for f in out["validity"]],
               validity_detail="; ".join(f"{f['name']}: {f['detail']}" for f in out["validity"]) or "ok",
               per_course=per_course, sim_s=out["sim_s"], wall_s=out["wall_s"], validity_s=out["validity_s"])
    safeio.write_json(out_dir / "validity.json", out["validity"] or "ok")
    safeio.write_json(out_dir / "score.json", rec)
    safeio.write_json(out_dir / "failure_log.json", failure)
    if traces:
        np.savez_compressed(out_dir / "trace.npz", **traces)
    rec["failure_text"] = {c: [f["summary"] for f in fl] for c, fl in failure.items()}
    return rec


def load_record(out_dir: Path) -> dict | None:
    sp, fp = out_dir / "score.json", out_dir / "failure_log.json"
    if not (sp.exists() and fp.exists()):
        return None
    rec = json.loads(sp.read_text())
    rec["failure_text"] = {c: [f["summary"] for f in fl] for c, fl in json.loads(fp.read_text()).items()}
    return rec


def evaluate_to_dir(script_path, out_dir, courses=COURSES, seeds=TRAIN_SEEDS, perturb=None, record_trace=True) -> dict:
    t0 = time.time()
    out_dir = Path(out_dir)
    err = check_source(Path(script_path).read_text())
    if err:
        return write_outputs(out_dir, _invalid("build", err, t0))
    try:
        genome, build_fn = load_script(script_path)
    except BuildError as e:
        return write_outputs(out_dir, _invalid("build", str(e), t0))
    out = evaluate(build_fn, genome, courses=tuple(courses), seeds=tuple(seeds), record_trace=record_trace,
                   perturb=perturb)
    return write_outputs(out_dir, out)


def evaluate_to_dir_subprocess(script_path, out_dir, courses=COURSES, seeds=TRAIN_SEEDS, timeout_s=EVAL_TIMEOUT_S,
                               perturb=None, record_trace=True) -> dict:
    """Untrusted (free-form) scripts run in a child process with a wall-clock limit."""
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "dumbrobot.m1", "eval-dir", str(script_path), str(out_dir), "--courses", *courses,
           "--seeds", *map(str, seeds)]
    if perturb:
        cmd += ["--perturb", json.dumps(perturb)]
    if not record_trace:
        cmd += ["--no-trace"]
    try:
        cp = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return write_outputs(out_dir, _invalid("build", f"wall-clock limit of {timeout_s:.0f} s exceeded", t0))
    rec = load_record(out_dir)
    if cp.returncode != 0 or rec is None:
        return write_outputs(out_dir, _invalid("build", "evaluation process failed: " + cp.stderr.strip()[-300:], t0))
    return rec


# ---- a run --------------------------------------------------------------------------------------------------------------
@dataclass
class Prop:
    pid: str
    gen: int
    j: int
    parent: str | None
    reason: str
    source: str
    genome: dict | None
    rec: dict
    dir: Path = field(repr=False, default=None)


class Run:
    def __init__(self, exp_root, arm: str, seed: int, budget: int, courses=COURSES, train_seeds=TRAIN_SEEDS,
                 subprocess_eval: bool = False):
        self.exp_root, self.arm, self.seed, self.budget = Path(exp_root), arm, seed, budget
        self.courses, self.train_seeds, self.subprocess_eval = tuple(courses), tuple(train_seeds), subprocess_eval
        self.dir = self.exp_root / arm / f"seed_{seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger = Ledger(self.dir / "ledger.jsonl", budget)
        self.charged = set()
        lp = self.dir / "ledger.jsonl"
        if lp.exists():
            for line in lp.read_text().splitlines():
                r = json.loads(line)
                self.charged.add((r["gen"], r["j"]))
        self.props: list[Prop] = []

    @property
    def used(self) -> int:
        return self.ledger.used

    @property
    def remaining(self) -> int:
        return self.budget - self.ledger.used

    def pdir(self, gen: int, j: int) -> Path:
        return self.dir / f"gen_{gen:03d}" / f"proposal_{j}"

    def propose(self, gen: int, j: int, source: str, genome: dict | None = None, parent: str | None = None,
                reason: str = "", raw: str = "", freeform: bool = False, no_proposal: str | None = None) -> Prop | None:
        """Evaluate (or reload) one proposal and charge it. Returns None only when the budget is exhausted."""
        if (gen, j) not in self.charged and self.remaining <= 0:
            return None
        d = self.pdir(gen, j)
        pid = f"g{gen:03d}p{j}"
        rec = load_record(d)
        if rec is None:
            safeio.write_text(d / "body.py", source)
            safeio.write_json(d / "genome.json", genome if genome is not None else "see body.py")
            if reason or raw:
                safeio.write_text(d / "reasoning.md", reason or "(no reason given)")
            if raw:
                safeio.write_text(d / "designer_raw.txt", raw)
            if no_proposal:
                rec = write_outputs(d, _invalid("no_proposal", no_proposal, time.time()))
            elif freeform or self.subprocess_eval:
                rec = evaluate_to_dir_subprocess(d / "body.py", d, self.courses, self.train_seeds)
            else:
                rec = evaluate_to_dir(d / "body.py", d, self.courses, self.train_seeds)
        if (gen, j) not in self.charged:
            out = dict(valid=rec["valid"], validity=[dict(name=n) for n in rec["why_invalid"]], score=rec["score"],
                       sim_s=rec["sim_s"], wall_s=rec["wall_s"], validity_s=rec["validity_s"])
            if not self.ledger.charge(self.arm, self.seed, out, gen=gen, j=j, pid=pid, parent=parent,
                                      src=source_hash(source)):
                return None
            self.charged.add((gen, j))
        p = Prop(pid, gen, j, parent, reason, source, genome, rec, d)
        self.props.append(p)
        self.write_population()
        return p

    def population(self) -> list[Prop]:
        """Best POP_SIZE valid, distinct designs by train score."""
        seen, out = set(), []
        for p in sorted((p for p in self.props if p.rec["valid"]), key=lambda p: (-p.rec["score"], p.gen, p.j)):
            h = source_hash(p.source)
            if h not in seen:
                seen.add(h)
                out.append(p)
        return out[:POP_SIZE]

    def write_population(self) -> None:
        safeio.write_json(self.dir / "population.json",
                          [dict(pid=p.pid, score=p.rec["score"], gen=p.gen, j=p.j, src=source_hash(p.source))
                           for p in self.population()])

    def finish(self) -> None:
        safeio.assert_clean(self.dir)
        safeio.write_json(self.dir / "done.json", dict(arm=self.arm, seed=self.seed, used=self.used, budget=self.budget))
