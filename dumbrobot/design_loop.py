"""The LLM design loop (PLAN section 10): population of best designs, generations of 1-4 children, per-segment failure
logs in the prompt, every reply cached before it is evaluated so a run can be killed and resumed for free."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import catalog as C
from . import safeio, schema, validity, worlds
from .designer import DesignerError, parse_reply
from .runlib import Prop, Run, check_source, genome_script, source_hash
from .scoring import FLIP_CAP, FLIP_PENALTY, W_ENERGY, W_TIME

ROOT = Path(__file__).resolve().parents[1]
MAX_GEN_SIZE = 4
SHOW_LOGS_FOR_TOP = 3


# ---- prompt pieces (facts only, no design advice) -----------------------------------------------------------------------
def course_text(courses) -> str:
    out = []
    for name in courses:
        c = worlds.get_course(name)
        segs = ", ".join(f"{s.name}" for s in c.segments)
        if name == "bumps":
            bumps = ", ".join(f"{h * 1000:.0f} mm" for h in worlds.BUMP_HEIGHTS)
            desc = (f"A straight lane {2 * worlds.LANE_HALF:.2f} m wide with side walls. The robot starts at x={c.start_x} m "
                    f"facing +x. Five round bumps (horizontal cylinders of radius {worlds.BUMP_RADIUS} m lying across the lane) "
                    f"stick out of the floor by {bumps} at x = "
                    f"{', '.join(f'{worlds.BUMP_X0 + i * worlds.BUMP_DX:.1f}' for i in range(5))} m. Finish line at "
                    f"x={c.finish_x:.2f} m.")
        elif name == "flat":
            desc = f"Flat floor, start x={c.start_x} m, finish x={c.finish_x} m."
        else:
            (a1, a2), (b1, b2) = worlds.BAFFLES
            desc = (f"A corridor {2 * worlds.CORR_HALF:.2f} m wide with side walls, flat floor. The robot starts at "
                    f"x={c.start_x} m, y=0 facing +x. A perpendicular wall at x={worlds.PILLAR_X} m runs from the right wall "
                    f"(y=-{worlds.CORR_HALF}) to y={worlds.PILLAR_Y1:+.2f} m, leaving a free gap of "
                    f"{worlds.CORR_HALF - worlds.PILLAR_Y1:.2f} m on the left. Then two slanted baffles (straight walls, {2 * worlds.WALL_H * 1000:.0f} mm high): baffle_1 from "
                    f"({a1[0]}, {a1[1]:+.2f}) to ({a2[0]}, {a2[1]:+.2f}) m, baffle_2 from ({b1[0]}, {b1[1]:+.2f}) to "
                    f"({b2[0]}, {b2[1]:+.2f}) m. Progress is measured along the centre-line path that threads the "
                    f"gaps ({c.length:.2f} m long); the finish is the end of that path.")
        out.append(f"- {name}: {desc} Time limit {c.t_limit:.0f} s. Failure-log segments: {segs}.")
    return "\n".join(out)


def rules_text() -> str:
    lo, hi = validity.MASS_RANGE
    return (
        "FIXED FACTS (the simulator is MuJoCo, 2 ms step; none of this can be changed by a design)\n"
        "- The brain is frozen: both wheel motors get full forward command (+1) for the whole run, no sensors, no steering, no "
        "memory. Any steering, backing off or obstacle handling has to come from the body.\n"
        "- Motors follow a DC torque-speed line and are chosen from a catalog: "
        + "; ".join(f"{k} (stall {v['stall']} N m, no-load {v['w0']} rad/s, {v['mass'] * 1000:.0f} g)" for k, v in C.MOTORS.items())
        + ".\n"
        "- Chassis effective density (kg/m^3): " + ", ".join(f"{k} {v:g}" for k, v in C.CHASSIS_MATERIALS.items()) + ". "
        f"A fixed {C.BATTERY_MASS * 1000:.0f} g battery sits at the chassis centre.\n"
        "- Friction robot-vs-terrain is the geometric mean of the two coefficients. Terrain mu is 0.8. Wheel mu is a design "
        f"parameter; chassis, bumper, whisker and tail rod mu is {C.MU_BODY}; casters are free-rolling balls.\n"
        "- A mechanical part is infinitely stiff and unbreakable unless it is a sprung joint.\n"
        "VALIDITY GATE (a design that fails it is rejected, scores 0 and still uses one evaluation)\n"
        f"- schema: every parameter present, right type, inside its range (out of range is an error, not clipped).\n"
        f"- mass: total between {lo} and {hi} kg. envelope: bounding box at most "
        f"{validity.ENVELOPE[0]} x {validity.ENVELOPE[1]} x {validity.ENVELOPE[2]} m (length x width x height).\n"
        f"- intersect: parts on different bodies may not overlap by more than {validity.PENETRATION_TOL * 1000:.1f} mm at the "
        f"rest pose. ground_clip: no part more than {validity.GROUND_TOL * 1000:.0f} mm below the floor at rest.\n"
        "- settle: dropped on flat ground it must rest upright. drive: on flat ground it must move forward more than 1 cm in "
        "1 s without flipping.\n"
        "SCORING (per course, then the mean over courses; progress and finish are about the chassis centre)\n"
        f"- score = progress_fraction + finished * ({W_TIME} * (1 - t/T_limit) + {W_ENERGY} * (1 - min(E/E_ref, 1))) "
        f"- {FLIP_PENALTY} * min(flips, {FLIP_CAP}). Finishing counts only upright. The episode ends at the finish, the time "
        "limit, when stuck (under 1 cm gained in 2 s, accumulated for 4 s), or after 3 s inverted. Each design is run with "
        "two different small random start-heading offsets per course; the score is the mean.\n"
    )


def format_genome_diff(genome: dict, base: dict) -> str:
    d = {k: v for k, v in genome.items() if base.get(k) != v}
    return json.dumps(d) if d else "{} (identical to the default template)"


def result_line(p: Prop, feedback: str) -> str:
    r = p.rec
    if not r["valid"] and r["why_invalid"] and r["per_course"] == {}:
        if feedback == "scalar":
            return "score=0.000 (rejected)"
        return f"REJECTED by the validity gate, score=0.000: {r['validity_detail']}"
    return f"score={r['score']:.3f}" + ("" if r["valid"] else " (an episode exploded)")


def failure_block(p: Prop, indent="    ") -> str:
    lines = []
    for c, texts in p.rec["failure_text"].items():
        for seed, t in zip(p.rec["per_course"][c]["episodes"], texts):
            lines.append(f"{indent}[start offset seed {seed['seed']}] " + t.replace("\n", "\n" + indent))
    return "\n".join(lines)


def design_text(p: Prop, mode: str, base: dict) -> str:
    if mode == "param":
        if p.genome is None:
            return "no design (the child was missing, malformed or had no valid parent)"
        return "genome changes vs the default template: " + format_genome_diff(p.genome, base)
    return "body script:\n```python\n" + p.source.rstrip() + "\n```"


def build_prompt(run: Run, mode: str, feedback: str, gen: int, n: int, last: list[Prop], pop: list[Prop]) -> str:
    base = schema.default_genome()
    parts = [
        "You are the designer in a closed loop. You propose changes to a small simulated two-wheel-drive robot's body "
        "(SIMULATION ONLY) so that a frozen, trivially dumb controller completes the courses below. Each proposal is built, "
        "checked, simulated and scored; you then see the result and propose again.\n",
        rules_text(),
        "COURSES\n" + course_text(run.courses) + "\n",
        "DESIGN SPACE (the parameters of the default template body; the template maps each onto physical parts: chassis box, "
        "two drive wheels with motors on the sides, optional front/rear caster balls, optional two-plate bumper (rigid or a "
        "sprung switch; angle>0 is a V pointing forward, angle<0 a funnel), 0-4 whiskers (sprung hinged rods), a tail rod with "
        "tip mass, and a ballast mass)\n" + schema.describe() + "\n",
    ]
    if mode == "free":
        parts.append("BODY API (source of dumbrobot/bodylib.py, the part helpers a script may call, and the default template "
                     "script dumbrobot/body_template.py)\n```python\n" + (ROOT / "dumbrobot" / "bodylib.py").read_text()
                     + "\n```\n```python\n" + (ROOT / "dumbrobot" / "body_template.py").read_text() + "\n```\n"
                     "A body script defines GENOME (a dict that must still pass the schema above) and build(world, g). It "
                     "may only import dumbrobot, math, numpy, mujoco, may not use open/exec/eval/getattr or dunder names, "
                     "and must create the chassis and the two drive wheels (add_chassis, add_drive). You may add new geoms "
                     "and joints through world.spec / the helpers, with real masses (density or mass), subject to the same "
                     "validity gate.\n")
    budget = (f"BUDGET: {run.budget} evaluations in total for this whole design effort; {run.used} used so far, "
              f"{run.remaining} left. Every child you propose costs one evaluation, including a child that is rejected, "
              f"malformed or missing. Generation {gen}: you must answer with exactly {n} child(ren).\n")
    parts.append(budget)
    parts.append(f"CURRENT BEST DESIGNS (best first; score is the mean over courses of the course scores)\n")
    if not pop:
        parts.append("(none yet)\n")
    shown = set()
    for i, p in enumerate(pop):
        parts.append(f"* {p.pid}: score={p.rec['score']:.3f} (generation {p.gen}). {design_text(p, mode, base)}")
        if feedback == "logs" and i < SHOW_LOGS_FOR_TOP:
            parts.append("  Failure log of its last evaluation:\n" + failure_block(p))
            shown.add(p.pid)
        parts.append("")
    if last:
        parts.append("LAST GENERATION (what you proposed and what happened)")
        for p in last:
            parts.append(f"* {p.pid} (child of {p.parent}): stated reason: {p.reason or '(none)'}\n  {design_text(p, mode, base)}\n"
                         f"  Result: {result_line(p, feedback)}")
            if feedback == "logs" and p.rec["failure_text"]:
                parts.append("  Failure log: (shown above)" if p.pid in shown else "  Failure log:\n" + failure_block(p))
            parts.append("")
    fmt = ("```json\n{\"param_name\": new_value, ...}\n```  (only the parameters that differ from the PARENT design; "
           "JSON numbers/strings; ints must be ints)" if mode == "param" else
           "```python\n<the complete body script>\n```")
    parts.append(
        "YOUR ANSWER\nFor each child write exactly this block (children numbered from 1), nothing else outside the blocks:\n"
        "=== CHILD <k> ===\nPARENT: <id of the design this child is derived from, one of the ids above>\n"
        "REASON: <one sentence: the mechanical idea and what it should change in the failure log>\n" + fmt + "\n\n"
        f"CHILDREN_REQUESTED: {n}\nALLOWED_PARENTS: {' '.join(p.pid for p in pop) if pop else 'g000p0'}\nMODE: {mode}\n")
    return "\n".join(parts)


# ---- the loop -----------------------------------------------------------------------------------------------------------
def _resolve(run: Run, pid: str | None, pop: list[Prop]) -> Prop | None:
    by = {p.pid: p for p in run.props}
    if pid in by and by[pid].rec["valid"]:
        return by[pid]
    return pop[0] if pop else None


def _clips(run: Run, gdir: Path, best: Prop) -> list[str]:
    from .render import render_clip  # lazy: renderer is optional for tests

    made = []
    tr = np.load(best.dir / "trace.npz")
    for c in run.courses:
        s = run.train_seeds[0]
        trace = {k: tr[f"{c}__{s}__{k}"] for k in ("t", "pos", "quat", "wheel")}
        out = gdir / f"clip_{c}.gif"
        render_clip(best.dir / "body.py", c, trace, out)
        made.append(out.name)
    return made


def run_llm(run: Run, designer, mode: str = "param", feedback: str = "logs", gen_size: int = 3, clips: bool = False,
            progress=None) -> None:
    """Raises DesignerError (after leaving everything resumable) if the designer cannot answer."""
    assert mode in ("param", "free") and feedback in ("logs", "scalar") and 1 <= gen_size <= MAX_GEN_SIZE
    free = mode == "free"
    base = schema.default_genome()
    seed_src = ((ROOT / "dumbrobot" / "body_template.py").read_text() if free else genome_script(base))
    p0 = run.propose(0, 0, seed_src, genome=base, reason="seed design: the default template (counts as one evaluation)",
                     freeform=free)
    last = [p0] if p0 else []
    if progress and p0:
        progress(run, p0)
    gen = 1
    while run.remaining > 0:
        n = min(gen_size, run.remaining)
        gdir = run.dir / f"gen_{gen:03d}"
        reply_path = gdir / "designer_reply.txt"
        pop = run.population()
        if reply_path.exists():
            raw = reply_path.read_text()
        else:
            prompt = build_prompt(run, mode, feedback, gen, n, last, pop)
            safeio.write_text(gdir / "prompt.txt", prompt)
            try:
                raw = designer.reply(prompt, tag=f"{run.arm}/seed_{run.seed}/gen_{gen:03d}")
            except DesignerError as e:
                safeio.write_json(gdir / "designer_error.json", dict(error=str(e)))
                raise
            safeio.write_text(reply_path, raw)
        children = parse_reply(raw, mode)
        made = []
        for j in range(n):
            ch = children[j] if j < len(children) else None
            if ch is None:
                p = run.propose(gen, j, "# no proposal\n", reason="", no_proposal="designer returned no child for this slot",
                                freeform=free)
            elif ch.error:
                p = run.propose(gen, j, "# malformed child\n", parent=ch.parent, reason=ch.reason, raw=ch.raw,
                                no_proposal="malformed child: " + ch.error, freeform=free)
            else:
                parent = _resolve(run, ch.parent, pop)
                if parent is None:
                    p = run.propose(gen, j, "# no parent\n", reason=ch.reason, raw=ch.raw, no_proposal="no valid parent",
                                    freeform=free)
                elif free:
                    p = run.propose(gen, j, ch.source, parent=parent.pid, reason=ch.reason, raw=ch.raw, freeform=True)
                else:
                    g = {**parent.genome, **ch.changes}
                    p = run.propose(gen, j, genome_script(g), genome=g, parent=parent.pid, reason=ch.reason, raw=ch.raw)
            if p is None:
                break
            made.append(p)
            if progress:
                progress(run, p)
        best = max((p for p in made if p.rec["valid"]), key=lambda p: p.rec["score"], default=None)
        clip_files, clip_err = [], None
        if clips and best is not None and not (gdir / "clip_done").exists():
            try:
                clip_files = _clips(run, gdir, best)
                safeio.write_text(gdir / "clip_done", "ok")
            except Exception as e:  # noqa: BLE001 - a render failure must not cost a paid generation
                clip_err = f"{type(e).__name__}: {e}"
        safeio.write_json(gdir / "summary.json", dict(
            gen=gen, used=run.used, best_child=best.pid if best else None, clips=clip_files, clip_error=clip_err,
            children=[dict(pid=p.pid, parent=p.parent, reason=p.reason, score=p.rec["score"], valid=p.rec["valid"],
                           why_invalid=p.rec["why_invalid"], src=source_hash(p.source)) for p in made]))
        last = made
        gen += 1
    run.finish()
