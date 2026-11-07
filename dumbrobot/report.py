"""M1 report: held-out re-evaluation of each run's winner, sim-artefact checks, medians and spreads per arm, an exact rank
test, every run listed (nothing dropped), and findings candidates for hand review. Output goes to git-ignored results/."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from . import runlib, safeio
from .runlib import HELD_OUT_SEEDS, COURSES, evaluate_to_dir, evaluate_to_dir_subprocess, load_record

ARM_ORDER = ("random", "cmaes", "llm_param", "llm_param_scalar", "llm_free")
PERTURBATIONS = {"dt_half": {"timestep_scale": 0.5}, "pgs_solver": {"solver": "PGS"}, "mu_x0.8": {"friction_scale": 0.8},
                 "mu_x1.2": {"friction_scale": 1.2}}
ARTEFACT_RATIO = 0.5  # tagged sim-artefact if a perturbed held-out score falls below this fraction of the nominal one
CHECKPOINTS = (8, 16, 24, 32, 48, 64)
IMPROVEMENT = 0.05  # a child "improved" if it beat its parent's score by this much (findings candidates)


# ---- statistics ---------------------------------------------------------------------------------------------------------
def spread(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return dict(n=0)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return dict(n=int(a.size), median=float(med), q1=float(q1), q3=float(q3), min=float(a.min()), max=float(a.max()))


def mann_whitney(x, y, max_combos: int = 400_000) -> dict:
    """U statistic of x over y and the exact permutation p-values (one-sided x > y, two-sided), mid-ranks for ties.
    Falls back to a normal approximation only when enumeration would be too large."""
    x, y = list(map(float, x)), list(map(float, y))
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return dict(U=None, p_greater=None, p_two_sided=None)
    allv = np.array(x + y)
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rx = float(ranks[:nx].sum())
    U = rx - nx * (nx + 1) / 2
    mean_r = nx * (nx + len(y) + 1) / 2
    if math.comb(nx + ny, nx) <= max_combos:
        sums = np.array([ranks[list(c)].sum() for c in itertools.combinations(range(nx + ny), nx)])
        p_gt = float(np.mean(sums >= rx - 1e-9))
        p2 = float(np.mean(np.abs(sums - mean_r) >= abs(rx - mean_r) - 1e-9))
    else:
        sd = math.sqrt(nx * ny * (nx + ny + 1) / 12)
        z = (rx - mean_r) / sd if sd else 0.0
        p_gt = 0.5 * math.erfc(z / math.sqrt(2))
        p2 = math.erfc(abs(z) / math.sqrt(2))
    return dict(U=float(U), nx=nx, ny=ny, p_greater=p_gt, p_two_sided=p2)


# ---- run discovery ------------------------------------------------------------------------------------------------------
def ledger_rows(run_dir: Path) -> list[dict]:
    p = run_dir / "ledger.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def best_of_run(run_dir: Path):
    """Winner = best train score among valid proposals in the ledger (ties: earliest). Returns (pid, gen, j) or None."""
    rows = [r for r in ledger_rows(run_dir) if r["valid"]]
    if not rows:
        return None
    r = max(rows, key=lambda r: (r["score"], -r["proposal"]))
    return r["pid"], r["gen"], r["j"], r["score"]


def curve(rows: list[dict], budget: int) -> list[float]:
    best, out = -1e9, []
    for r in rows[:budget]:
        best = max(best, r["score"])
        out.append(best)
    return out + [out[-1] if out else 0.0] * (budget - len(out))


def discover(exp_root: Path) -> dict[str, list[tuple[int, Path]]]:
    out = {}
    for arm in ARM_ORDER:
        for sd in sorted((exp_root / arm).glob("seed_*"), key=lambda p: int(p.name.split("_")[1])) if (exp_root / arm).exists() else []:
            out.setdefault(arm, []).append((int(sd.name.split("_")[1]), sd))
    return out


# ---- held-out evaluation -----------------------------------------------------------------------------------------------
def _eval(script: Path, out_dir: Path, free: bool, seeds, perturb=None):
    rec = load_record(out_dir)
    if rec is not None:
        return rec
    if free:
        return evaluate_to_dir_subprocess(script, out_dir, COURSES, seeds, perturb=perturb, record_trace=False)
    return evaluate_to_dir(script, out_dir, COURSES, seeds, perturb=perturb, record_trace=False)


def heldout(run_dir: Path, arm: str, with_perturb: bool = True) -> dict | None:
    b = best_of_run(run_dir)
    if b is None:
        return dict(pid=None, score=0.0, valid=False, train=None, per_course={}, perturbed={}, artefact=False)
    pid, gen, j, train = b
    script = run_dir / f"gen_{gen:03d}" / f"proposal_{j}" / "body.py"
    free = arm == "llm_free"
    nominal = _eval(script, run_dir / "heldout" / "nominal", free, HELD_OUT_SEEDS)
    out = dict(pid=pid, train=train, score=nominal["score"], valid=nominal["valid"],
               per_course={c: dict(mean_score=v["mean_score"], finished=sum(e["done"] for e in v["episodes"]),
                                   n=len(v["episodes"]), progress=float(np.mean([e["progress"] for e in v["episodes"]])))
                           for c, v in nominal["per_course"].items()}, perturbed={}, artefact=False)
    if with_perturb:
        for name, pt in PERTURBATIONS.items():
            r = _eval(script, run_dir / "heldout" / name, free, HELD_OUT_SEEDS[:2], perturb=pt)
            out["perturbed"][name] = r["score"]
        nom2 = _eval(script, run_dir / "heldout" / "nominal_2seeds", free, HELD_OUT_SEEDS[:2])["score"]
        out["nominal_2seeds"] = nom2
        out["artefact"] = bool(nom2 > 0.05 and min(out["perturbed"].values()) < ARTEFACT_RATIO * nom2)
    safeio.write_json(run_dir / "heldout" / "summary.json", out)
    return out


REFERENCE = {"default_template": None, "hand_big_wheels": "examples/hand_made_big_wheels.py",
             "hand_small_wheels": "examples/hand_made_small_wheels.py"}


def reference_rows(exp_root: Path) -> dict:
    out = {}
    for name, rel in REFERENCE.items():
        d = exp_root / "_reference" / name
        script = d / "body.py"
        if rel is None:
            safeio.write_text(script, runlib.genome_script(__import__("dumbrobot.schema", fromlist=["x"]).default_genome()))
        else:
            safeio.write_text(script, (runlib.ROOT / rel).read_text())
        rec = _eval(script, d / "heldout", False, HELD_OUT_SEEDS)
        out[name] = dict(score=rec["score"], per_course={c: v["mean_score"] for c, v in rec["per_course"].items()})
    return out


# ---- findings candidates -------------------------------------------------------------------------------------------------
def findings_candidates(exp_root: Path, runs) -> str:
    lines = ["# Findings candidates (auto-extracted; each needs hand review against the clip and failure log before it is "
             "counted)\n",
             "A child is listed when it beat its parent's train score by at least " + f"{IMPROVEMENT}" + ". The reason is the "
             "designer's own statement, not a verified mechanism. Classify each as `sensing` or `mechanics` and say whether "
             "it survives the held-out check and the sim-artefact perturbations.\n"]
    for arm in ("llm_param", "llm_param_scalar", "llm_free"):
        for sd, rd in runs.get(arm, []):
            rows = {r["pid"]: r for r in ledger_rows(rd)}
            for r in rows.values():
                par = rows.get(r.get("parent"))
                if par and r["valid"] and r["score"] >= par["score"] + IMPROVEMENT:
                    pdir = rd / f"gen_{r['gen']:03d}" / f"proposal_{r['j']}"
                    reason = (pdir / "reasoning.md").read_text().strip() if (pdir / "reasoning.md").exists() else ""
                    lines.append(f"- {arm} seed {sd} gen {r['gen']} {r['pid']} (parent {par['pid']}): "
                                 f"train {par['score']:.3f} -> {r['score']:.3f}. Reason: {reason}  Dir: {pdir}")
    return "\n".join(lines) + "\n"


# ---- main ---------------------------------------------------------------------------------------------------------------
def build_report(exp_root: Path, with_perturb: bool = True, budget: int | None = None) -> str:
    runs = discover(exp_root)
    table, by_arm = [], {}
    for arm in ARM_ORDER:
        for sd, rd in runs.get(arm, []):
            meta = json.loads((rd / "meta.json").read_text()) if (rd / "meta.json").exists() else {}
            rows = ledger_rows(rd)
            bud = budget or meta.get("budget") or len(rows)
            h = heldout(rd, arm, with_perturb)
            info = dict(arm=arm, seed=sd, used=len(rows), budget=bud, complete=(rd / "done.json").exists() and len(rows) >= bud,
                        invalid=sum(not r["valid"] for r in rows), sim_s=sum(r["sim_s"] for r in rows),
                        wall_s=sum(r["wall_s"] for r in rows), curve=curve(rows, bud), **h)
            table.append(info)
            by_arm.setdefault(arm, []).append(info)
    ref = reference_rows(exp_root)
    L = ["# M1 report (SIMULATION ONLY; MuJoCo; untracked results)\n",
         f"Courses: {', '.join(COURSES)}. Train jitter seeds {list(runlib.TRAIN_SEEDS)} (seen by every designer); held-out "
         f"jitter seeds {list(HELD_OUT_SEEDS)} (never seen). Budget unit = one proposal, invalid ones included. All claims use "
         "the held-out score of each run's winner (best train score in that run). Seeds are not pooled.\n",
         "## Reference designs (held-out score; headroom check, CRITIQUES 4)\n", "| design | score | " +
         " | ".join(COURSES) + " |", "|---|---|" + "---|" * len(COURSES)]
    for n, r in ref.items():
        L.append(f"| {n} | {r['score']:.3f} | " + " | ".join(f"{r['per_course'].get(c, float('nan')):.3f}" for c in COURSES) + " |")
    L += ["\n## Every run (nothing dropped)\n",
          "| arm | seed | proposals used/budget | complete | invalid | train best | held-out | held-out per course (finished/n) | "
          "artefact | winner |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in table:
        pc = ", ".join(f"{c}: {v['mean_score']:.2f} ({v['finished']}/{v['n']})" for c, v in t["per_course"].items())
        L.append(f"| {t['arm']} | {t['seed']} | {t['used']}/{t['budget']} | {t['complete']} | {t['invalid']} | "
                 f"{t['train'] if t['train'] is None else round(t['train'], 3)} | {t['score']:.3f} | {pc} | "
                 f"{t['artefact']} | {t['pid']} |")
    L += ["\n## Per arm: held-out score of the winners (median, IQR, min-max over seeds)\n",
          "| arm | seeds | median | Q1 | Q3 | min | max | median train best | median invalid frac | median sim-s | median wall-s |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    summ = {}
    for arm in ARM_ORDER:
        ts = by_arm.get(arm, [])
        if not ts:
            continue
        s = spread([t["score"] for t in ts])
        summ[arm] = s
        L.append(f"| {arm} | {s['n']} | {s['median']:.3f} | {s['q1']:.3f} | {s['q3']:.3f} | {s['min']:.3f} | {s['max']:.3f} | "
                 f"{np.median([t['train'] or 0 for t in ts]):.3f} | {np.median([t['invalid'] / max(1, t['used']) for t in ts]):.2f} | "
                 f"{np.median([t['sim_s'] for t in ts]):.0f} | {np.median([t['wall_s'] for t in ts]):.0f} |")
    L += ["\n## Best-so-far train score vs proposals (median [Q1, Q3] over seeds)\n",
          "| arm | " + " | ".join(f"n={k}" for k in CHECKPOINTS) + " |", "|---|" + "---|" * len(CHECKPOINTS)]
    for arm in ARM_ORDER:
        ts = by_arm.get(arm, [])
        if not ts:
            continue
        cells = []
        for k in CHECKPOINTS:
            v = [t["curve"][min(k, len(t["curve"])) - 1] for t in ts if t["curve"]]
            q = spread(v)
            cells.append(f"{q['median']:.2f} [{q['q1']:.2f}, {q['q3']:.2f}]" if q["n"] else "-")
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L += ["\n## Rank tests on held-out winner scores (exact permutation Mann-Whitney)\n",
          "| A vs B | n_A, n_B | U_A | p(A > B) one-sided | p two-sided |", "|---|---|---|---|---|"]
    pair_res = {}
    for a, b in (("llm_param", "random"), ("llm_param", "cmaes"), ("cmaes", "random"), ("llm_param", "llm_param_scalar"),
                 ("llm_free", "random"), ("llm_free", "cmaes"), ("llm_free", "llm_param")):
        if a in by_arm and b in by_arm:
            r = mann_whitney([t["score"] for t in by_arm[a]], [t["score"] for t in by_arm[b]])
            pair_res[(a, b)] = r
            L.append(f"| {a} vs {b} | {r['nx']}, {r['ny']} | {r['U']:.1f} | {r['p_greater']:.3f} | {r['p_two_sided']:.3f} |")
    L += ["\n## Claim check (pre-declared rule)\n",
          "\"The LLM wins\" requires, for the arm in question, a higher median held-out score than BOTH random and CMA-ES and a "
          "one-sided exact p < 0.05 against each, at the same budget, with every seed reported above. Failure-log asymmetry "
          "(the LLM sees per-segment logs and a plain-words task description, the baselines see the scalar only) is the "
          "thing under test. Runs tagged artefact (a perturbed held-out score under "
          f"{ARTEFACT_RATIO:.0%} of nominal) are listed but excluded from claims by hand.\n"]
    for arm in ("llm_param", "llm_free"):
        if arm in summ and "random" in summ and "cmaes" in summ:
            beats = all(summ[arm]["median"] > summ[b]["median"] and pair_res.get((arm, b), {}).get("p_greater", 1) < 0.05
                        for b in ("random", "cmaes"))
            L.append(f"- {arm}: median {summ[arm]['median']:.3f} vs random {summ['random']['median']:.3f}, cmaes "
                     f"{summ['cmaes']['median']:.3f}; rule satisfied: {beats}")
    L.append("\nMechanical tricks the designer invented: see findings_candidates.md (to be hand-reviewed into findings.md).\n")
    text = "\n".join(L)
    safeio.write_text(exp_root / "report.md", text)
    safeio.write_json(exp_root / "report.json", dict(runs=table, reference=ref, summary=summ,
                                                      pairs={f"{a}|{b}": v for (a, b), v in pair_res.items()}))
    safeio.write_text(exp_root / "findings_candidates.md", findings_candidates(exp_root, runs))
    csv = ["arm,seed,n,best_so_far"] + [f"{t['arm']},{t['seed']},{i + 1},{v:.4f}" for t in table for i, v in enumerate(t["curve"])]
    safeio.write_text(exp_root / "curves.csv", "\n".join(csv))
    return text


def main(a) -> int:
    print(build_report(Path(a.results) / a.exp))
    return 0
