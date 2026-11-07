# dumb-robot: plan

**Everything here is simulation (MuJoCo). No hardware, no claims about real robots.**

Premise: the brain is frozen and almost empty; a designer (an LLM, or a baseline optimizer) may only change the body.
The question is how much task competence a smart designer can move from software into passive mechanics, and whether an
LLM designer does that better than random search and CMA-ES given the same evaluation budget. See CRITIQUES.md for the
objections this plan has to survive.

No result numbers appear in tracked files. Scores, logs, clips and renders live in git-ignored `results/`.

## 1. Brains (frozen forever)

| Brain | Behaviour | Inputs |
|---|---|---|
| A | both drive motors at full forward command (`ctrl = [+1, +1]`) for the whole episode | none |
| B (M2) | as A, but when the bumper switch reads "pressed", command `[-1, +1]`-style reverse-and-turn for 0.5 s, then resume | one boolean (bumper) |

Brain B is implemented and unit-tested in M0 but not used in any experiment before M2. Brains live in `dumbrobot/brains.py`, are pure functions of (time, sensor booleans), and have no parameters. A test
asserts that Brain A's output is constant and that nothing in `brains.py` is imported by the design or optimizer code.
Brain B's turn direction is fixed (always the same way) so it contains no left/right decision.

## 2. Body builder (mjSpec)

`dumbrobot/bodylib.py` holds part helpers on top of `mujoco.MjSpec`: chassis, drive wheel pair with motors, caster ball,
bumper (rigid or sprung switch, flat or wedge), whiskers (sprung hinges), tail, ballast. It enforces the physical
honesty rules in section 4.

A **body script** is an ordinary Python file the designer edits. It defines:

```python
GENOME = {...}                       # parameter values, schema in section 3
def build(world, g):                 # world: bodylib.Builder, g: validated genome dict
    ...                              # calls bodylib helpers or raw mjSpec calls
```

`dumbrobot/body_template.py` is the default script: it maps the full genome onto the helpers. The runner loads a body
script by path, builds the robot into a fresh spec, attaches a course, compiles, and validates. The script is untrusted
input: it is executed in a subprocess with a wall-clock limit (M1; M0 runs it in-process because only hand-written
scripts exist).

Two designer arms, reported separately:

1. **Parametric arm** (fair fight): everybody edits only `GENOME` for the template script. Random, CMA-ES and the LLM all
   search the identical space.
2. **Free-form arm**: the LLM may also edit the script code (new part shapes, new mountings) inside the bodylib/mjSpec
   API and the same validity gate. This is a superset of the parametric space, so it is never compared against the
   baselines without the parametric-arm result next to it.

Drive is differential: two motorized wheels (hinge about the body y axis). Legs and treads are not in M0; a tread is
approximated later as a wheel pair plus a belt contact patch (M2, listed so nobody assumes it exists).

## 3. Design-space schema (shared by LLM, random, CMA-ES)

`dumbrobot/schema.py` holds an ordered list of `Param(name, kind, lo, hi | choices, default, unit)` with
`kind in {float, int, enum}`. One genome is a dict `{name: value}`. Three codecs, all defined by the schema:

- `validate(g)` returns a list of schema errors (missing/extra key, wrong type, out of bounds, unknown enum). Out-of-bounds
  is an error, not silently clipped, so the LLM sees what it did wrong.
- `to_unit(g)` / `from_unit(u)`: map to and from the unit hypercube `[0,1]^d`. Floats are linear, ints and enums are binned.
  This is the representation CMA-ES and random search use (CMA-ES clips to the cube; the decode is deterministic).
- `sample(rng)`: uniform in the unit cube then decoded. This is the random-search proposal distribution.

Parameter groups (names are final in `schema.py`; bounds there are the source of truth):

- chassis: length, width, height, effective-density material (enum), centre height above axle
- drive: wheel radius, width, axle x-position, tread friction, motor (enum from a small gearmotor catalog)
- casters: front/rear present, radius, lift above ground (free-rolling balls on ball joints; a level caster that rides up
  a bump lifts the drive wheels off the ground, so lift is a design lever)
- bumper: present, kind (rigid | switch), width, reach, wedge half-angle (negative = funnel), height, ground clearance
- whiskers: count (0..4), length, outward angle, spring stiffness
- tail: length, drop angle, tip mass
- ballast: mass, x, z

The same genome is what the LLM sees in each prompt, with the schema table, so a proposal is a small JSON/dict diff plus a
stated reason. CMA-ES works on the continuous relaxation (about 30 dimensions); ints and enums are binned, which is
a known weakness and is called out in the report rather than hidden.

## 4. Physical honesty and validity

Honesty rules (builder side):

- Motors follow a DC torque-speed line, natively in MuJoCo: `tau = stall * u - (stall / w_noload) * w`, then clamped to
  `[-stall, stall]`, `u in [-1, 1]`. Catalog entries (stall torque, no-load speed, mass, reflected armature) are
  representative hobby gearmotor figures, not datasheet copies [verify]. The designer picks from the catalog; it cannot
  invent a motor.
- Every part gets mass from `density * volume` (effective densities for hollow/shelled materials are stated in
  `catalog.py` and are approximations [verify]) or from an explicit catalog mass. Motors add their catalog mass at the
  axles. No body with mass below 1e-4 kg is allowed (tiny masses explode a 2 ms integrator).
- Friction between a robot geom and a terrain geom is the geometric mean `sqrt(mu_robot * mu_terrain)`, applied through
  explicit contact pairs, because MuJoCo's default is the maximum of the two coefficients, which would make ice
  irrelevant to a grippy wheel. Robot and terrain use separate contype/conaffinity bits so only those pairs (plus robot
  self-contacts between non-adjacent bodies) collide.
- Fixed timestep 2 ms, control at 100 Hz, pyramidal friction cone, MuJoCo default solver settings. Nothing in a body
  script can change these (the runner owns options).

Validity gate (`dumbrobot/validity.py`), run before any scoring, returns a list of named failures; any failure means the
proposal is rejected and the designer is told why:

1. `schema`: schema errors from section 3.
2. `build`: the script raised, or MuJoCo failed to compile the spec.
3. `mass`: total mass outside [0.15, 3.0] kg, or any body below 1e-4 kg.
4. `envelope`: bounding box over 0.45 m long, 0.40 m wide or 0.25 m tall at the rest pose (arena fit).
5. `intersect`: any two geoms on **different** bodies (parent/child included) penetrating more than 1.5 mm at the rest
   pose, measured by the engine through explicit contact pairs (`mj_geomDistance` reports 0 for deep overlaps, so it is
   not used). Geoms welded to one body may overlap (that is how fixed parts attach); a caster ball may overlap the chassis
   (an unmodelled housing). Range-of-motion sweeps are not checked (known gap; the explosion check catches the worst
   case). Structural strength is not modelled: a rigid bumper or tail is infinitely stiff and unbreakable, mass only.
5b. `ground_clip`: any robot part more than 2 mm below the ground plane at the rest pose.
6. `settle`: dropped on flat ground with motors off for 1 s: any NaN, speed above 5 m/s, or not resting upright (chassis up
   axis within about 25 degrees of vertical) is a failure (`explode` or `unstable`).
7. `drive`: on flat ground, Brain A for 1 s moves the body forward more than 1 cm and does not flip. This catches
   mirrored axes and designs that cannot move at all.
8. `runtime` (during scoring, not before): NaN or speed above 20 m/s aborts the episode as `exploded`; it is scored zero
   and flagged invalid.

## 5. Courses

Courses are built by `dumbrobot/worlds.py` as extra geoms on the same spec. Each returns a `Course`: geometry, start pose,
finish rule, time limit, terrain friction, and an ordered list of **segments** (name, x-interval) used for per-segment
logs. Progress is the furthest distance along the course path (x in course 1; path distance in the maze).

Staged set (M0 built course 1, M1 adds course 2):

1. **bumps**: 0.9 m wide lane with side walls, 5 round bumps of increasing height (8 to 40 mm), 25 s limit.
2. **walls/corridor** (M1): a 1.1 m wide corridor. A perpendicular wall spans from the right side wall to just past the
   centre line, leaving a gap on the left; then two slanted baffles (a two-turn chicane). A centred wall with gaps on both
   sides was tried first and no valid design could pass it (a symmetric blind robot driving at a symmetric wall just
   stalls), so the wall is one-sided, which lets a forward-pointing wedge deflect around its edge (the CRITIQUES 4 headroom rule).
   Progress is the arc length along a centre-line path threaded through the gaps, measured by the nearest path sample within a
   window around the furthest progress so far (a robot cannot be credited across a wall). Segments: approach, wall,
   corridor, chicane_1, chicane_2, run_out. Geometry constants are in `worlds.py`.
3. **simple maze** (M2): grid maze, 3 junctions, one dead end; progress is geodesic distance to the exit.
4. **ramp + gap** (M2): ramp up, a flat platform, a gap narrower than the longest allowed body.

Darwin's-finches worlds (M3): sand/low friction, ice, stairs, maze, steep hill, narrow corridor. Each uses the same
code path with different terrain geoms and friction.

Finish counts only when the chassis is upright (up axis above the flip threshold) at or past the line.

Termination: finish reached; time limit; `stuck` (less than 1 cm progress over a 2 s window, accumulated for 4 s);
`flipped_stuck` (inverted for 3 s); `exploded`.

## 6. Scoring

Per episode, all on fixed constants defined in `scoring.py` before any designer sees them:

```
p        = clip(progress / course_length, 0, 1)
score    = p
         + done * ( 0.30 * (1 - t / T_limit) + 0.15 * (1 - min(E / E_ref, 1)) )
         - 0.05 * min(flips, 4)
```

`done` is 1 if the finish was reached. `E` is electrical energy (below), `E_ref` is a fixed course constant. Completion
dominates; time and energy break ties among finishers; flips are a small penalty. Raw components (progress, completion,
time, energy, flips) are always logged next to the scalar so a different weighting can be applied afterwards.

Electrical energy per motor: `P = max(0, tau * w) + tau^2 * w_noload / stall` (mechanical power plus resistive loss
from the DC motor model, with no regeneration credit), integrated over the episode.

A flip is a transition of the chassis up axis below 0.2 (about 78 degrees from vertical) after it had been above 0.5.

Multi-course score (M1+): mean of per-course scores over the courses in the experiment, with a design that fails to
complete course k simply scoring its progress there.

Determinism: the simulator is deterministic. To stop designs from overfitting one exact trajectory, the start yaw is
jittered from a seed. Optimization uses a fixed set of jitter seeds per evaluation; final reporting re-evaluates the
best designs on held-out jitter seeds that no optimizer ever saw.

## 7. Evaluation budget accounting

Unit of budget: **one proposal**. Every proposal (a genome from CMA-ES or random search, a diff from the LLM) costs 1 unit
whether it is invalid or not, so rejection sampling is not free for the baselines and a malformed LLM edit is not free
for the LLM. A proposal that passes validity is evaluated on the experiment's whole course set with the fixed jitter
seeds; that bundle is the one unit. The ledger (`ledger.jsonl`, one line per proposal) records: arm, seed, proposal
index, valid/invalid and why, score, sim seconds, wall seconds, validity seconds. Secondary comparisons report
sim-seconds and wall-clock so that an arm cannot win by being cheap per proposal.

Fairness rules:

- Same budget N per run for every arm in a comparison, several independent seeds per arm (at least 5 for the baselines,
  as many as the subscription budget allows for the LLM; the report states the count and does not pool seeds).
- Same genome schema (parametric arm), same validity gate, same scorer, same course set.
- The LLM sees per-segment failure logs; the baselines see the scalar only. This asymmetry is the thing under test and
  is stated in every report. An extra ablation gives the LLM the scalar only.
- LLM batches: the LLM proposes in generations of 1 to 4 children; the budget counts children, not generations. A slot the
  LLM leaves empty or fills with a malformed reply is charged as an invalid proposal (`no_proposal`). The LLM arm's seed
  design (the default template, whose log it needs to start) is evaluated as proposal 1 and charged to its budget; the
  baselines get no such evaluation, so this slightly favours the baselines. CMA-ES starts from the same default design.
- M1 experiment: courses bumps and corridor, budget 64 proposals per run, train jitter seeds 0 and 1, held-out jitter seeds
  1000 to 1003, 10 seeds per baseline, as many LLM seeds as the subscription allows (stated in the report, never pooled).
  The winner of a run is its best train score; only its held-out score is used for claims. The report also re-runs each
  winner under a halved timestep, the PGS solver and plus/minus 20 percent friction (CRITIQUES 3) and tags a winner
  `sim-artefact` if a perturbed score falls below half of its nominal score.
- What the LLM is told that the baselines are not: the per-segment failure logs (the asymmetry under test) and a
  plain-words description of the task (course geometry, scoring, validity rules, the physical facts). That second item is
  also a privilege the baselines lack, and the scalar-only ablation keeps it, so the ablation isolates the logs only.
- Winner's-curse control: only the held-out re-evaluation is used for claims.
- Statistics: report the median and interquartile range of best-so-far curves across seeds, plus a rank test between
  arms; "the LLM wins" requires beating both baselines on held-out score at equal budget, not on one lucky seed.

## 8. Logging layout (under git-ignored `results/`)

```
results/runs/<experiment>/<arm>/seed_<k>/
    ledger.jsonl                    # one line per proposal (section 7)
    population.json                 # best-so-far designs kept (top 8 by held-in score)
    gen_<NNN>/
        proposal_<j>/
            body.py                 # exact source that was scored
            genome.json
            reasoning.md            # designer's stated reason (LLM arm), emails redacted
            designer_raw.txt        # raw reply, emails redacted before saving
            validity.json           # named failures or "ok"
            score.json              # scalar + raw components per course
            failure_log.json        # per-segment log (section 9)
            trace.npz               # 20 Hz chassis pose + wheel angles (clip source for rendering)
results/runs/<experiment>/findings.md   # mechanical tricks the designer invented (section 10)
```

`trace.npz` is the clip source. M1 also renders a short CPU clip of each generation's best child (`gen_NNN/clip_<course>.gif`,
top view plus side view, own software rasterizer, single-threaded, no encoder dependency); the Blender renders stay in M3.
Per generation the loop also writes `prompt.txt`, `designer_reply.txt` (emails redacted) and `summary.json`; the
designer's stated reason is in each proposal's `reasoning.md` and the exact scored source in `body.py`.
Every proposal is cached, so any run resumes without re-simulating or re-charging; a designer reply is saved before its
children are evaluated.

## 9. Per-segment failure log

For every segment: first/last time inside, duration, forward progress, mean speed, stuck seconds, flips, minimum chassis
up-axis value, seconds of wheel spin without forward motion (slip), and energy. For the segment where the episode
ended without finishing, also the robot geoms and terrain geoms in contact at the moment `stuck` was declared. A
generated one-paragraph plain-text summary is what goes into the LLM prompt; it states facts (where, how long, what was
touching) and gives no design advice.

## 10. Designer loop (M1)

Claude through the subscription CLI only (`claude -p`), never a paid API. Isolation per call: working directory is a fresh
directory under `/var/tmp`, `--setting-sources project,local` (the fresh directory has none, so no user or project memory
loads), tools disabled (`--tools ""`; the prompt carries everything the loop needs), no MCP (`--strict-mcp-config` with an
empty config), no session persistence, a short fixed system prompt, and any email address is redacted from every saved
reply and probe. All writes go through `safeio`, which reads each file back and stops the run if an address survives.
Two arms: `llm_param` (the design JSON per the schema; the fair fight), `llm_free` (a whole edited body script, run in a
subprocess with a wall-clock limit after a static import/name check; a guard, not a sandbox), and the ablation
`llm_param_scalar` (score only, no failure logs). The prompt contains the schema, the current best body scripts, their score and failure
summaries, and asks for edits plus a one-sentence reason each. Batches of `claude -p` calls are launched by the organizer
from a prepared runner script (written to HANDOFF.md under "Organizer: run these") if the session permission system
refuses them. A findings list is maintained by hand-reviewing reasoning against failure logs after each run; each entry
has the trick, the generation it first appeared, the evidence (failure log before vs after), and whether it survives on a
held-out world.

## 11. Baselines (M1)

- **Random search**: `schema.sample` per proposal, same budget.
- **CMA-ES** (`cma` package, version pinned in THIRD_PARTY.md and logged): unit-cube parametrization, population 8, initial
  sigma 0.3, start point = the default design (the same design the LLM is shown first), fixed seeds, same budget, invalid
  proposals get a fixed bad fitness (the worst valid score seen so far, or 0) and still consume budget. A final partial
  generation is evaluated but not told.

## 12. Tests

M0 tests (`tests/`): schema round-trip and bound rejection; every validity failure triggered by a purpose-built bad
design; deterministic builds and episodes; motor model (steady-state wheel speed against `w_noload * r`, torque at
stall); energy accounting on a synthetic trace; flip counter on synthetic orientations; segment bookkeeping; scoring
formula on hand-made episode results; a hand-made large-wheel design completes course 1 and a small-wheel one fails at a
known bump; Brain A constant output.

M1 tests (`tests/test_m1.py`, `tests/test_render.py`): corridor segment bookkeeping and path progress; headroom (default
fails the corridor, a hand-made wedge finishes held-out seeds); redaction and the refuse-to-continue check; the `claude`
command line is isolated and runs in a fresh empty `/var/tmp` directory (subprocess mocked); reply parsing (good,
malformed, free-form); the static script guard and the wall-clock limit; budget accounting (invalid and missing children are
charged, the budget is never exceeded); resume of random search, CMA-ES and the LLM loop without new charges or new
designer calls; survival of a designer failure mid-run; the failure logs are in the prompt and absent in the scalar
ablation; the optimizer code never touches `brains`; exact rank test; report end to end on tiny runs; GIF writer
round trip. No test calls `claude`.

## 13. Visuals (M3)

Evolution timelapse and final-bodies grid via Blender, GPU only through `gpu-run --preemptible`. Reference tools are in
the arena repo at `~/projects/wt/robot-arena-sim-blender-render/tools/blender` (`render_shot.py`, `shot_plan.py`,
`export_poses.py`, `encode.sh`). That repo has no LICENSE file at the time of writing, so nothing is copied from it; we
call or re-implement against our own trace format after checking its license status again [verify].

## 14. Milestones

- **M0**: PLAN/README/THIRD_PARTY/CRITIQUES; body builder, Brain A, course 1, validity gate, scoring, one hand-made design
  scoring correctly, tests.
- **M1**: LLM loop vs random vs CMA-ES on courses 1 and 2, seeds, report.
- **M2**: all four staged courses, Brain B.
- **M3**: Darwin's finches and renders.
