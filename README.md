# dumb-robot

**Simulation only (MuJoCo). No hardware.**

A robot with a deliberately dumb, frozen brain, and a designer that may only change its body. Brain A drives both motors
full forward with no sensors. Brain B adds a single bumper reflex. An LLM designer (Claude through the subscription CLI)
edits a Python body script built on MuJoCo's `mjSpec` API: wheels, casters, bumpers, whiskers, tails, springs, ballast.
Failure logs tell it where each attempt got stuck, flipped or slowed. Random search and CMA-ES search the same design space
with the same budget; the LLM only counts as having shown something if it beats them.

Questions:

1. How much task competence can a smart designer move into passive mechanics for a brain that never changes?
2. Does an LLM designer beat random search and CMA-ES at equal budget?
3. Given different worlds (sand, ice, stairs, maze, steep hill, narrow corridor), do separately evolved bodies diverge?
   ("Darwin's finches".)

Plan, design-space schema, scoring and budget accounting: [PLAN.md](PLAN.md). Objections we have to survive:
[CRITIQUES.md](CRITIQUES.md). Third-party components: [THIRD_PARTY.md](THIRD_PARTY.md).

## Status

M0 (body builder, Brain A, course 1, validity gate, scoring, tests) and M1 machinery (course 2, random search, CMA-ES, the
LLM design loop, held-out report). Results are kept outside git; this README does not quote any.

## Layout

```
dumbrobot/    catalog, schema, bodylib, body_template, worlds, validity, brains, sim, scoring        (M0)
              runlib, baselines, designer, design_loop, safeio, render, report, m1                    (M1)
scripts/      run_llm_batch.sh (resumable `claude -p` batches)
tests/        pytest suite (the LLM loop is tested with a fake designer; no test calls `claude`)
PLAN.md  CRITIQUES.md  THIRD_PARTY.md
```

## M1 experiment

All arms get the same budget of proposals (invalid ones count), the same schema, validity gate, scorer and courses
(bumps, corridor); winners are re-evaluated on held-out start-heading seeds.

```
python -m dumbrobot.m1 run --arm random|cmaes --seed K            # baselines (CPU, via ~/projects/bin/sim-run)
python -m dumbrobot.m1 probe                                      # one isolated `claude -p` call, emails redacted
scripts/run_llm_batch.sh llm_param 0 1 2 3 4                       # LLM designer runs, resumable
python -m dumbrobot.m1 report                                      # held-out re-evaluation, tables, rank tests
```

Every run is resumable: proposals and designer replies are cached on disk before they are spent.

## Run the tests

```
python -m pytest -q
```

CPU simulation on the shared machine goes through `~/projects/bin/sim-run` (one thread, nice 10) with total load
kept under 20. Evaluate the hand-made example with:

```
python -m dumbrobot.cli score examples/hand_made_big_wheels.py --course bumps
```

## License

Apache-2.0, see [LICENSE](LICENSE).
