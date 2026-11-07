"""Baseline designers over the shared schema (PLAN section 11): random search and CMA-ES on the unit cube.

Both are deterministic functions of the seed and of the scores they have been told, and every proposal is cached by the
Run, so a killed run resumes by replaying: cached proposals are reloaded, not re-simulated, and not re-charged."""
from __future__ import annotations

import numpy as np

from . import schema
from .runlib import Run, genome_script

CMA_POPSIZE = 8
CMA_SIGMA0 = 0.3
CMA_VERSION_PIN = "4.5.0"


def run_random(run: Run, progress=None) -> None:
    rng = np.random.default_rng(10_000 + run.seed)
    for i in range(run.budget):
        g = schema.sample(rng)  # always draw, so a resumed run sees the same stream
        gen, j = divmod(i, CMA_POPSIZE)
        p = run.propose(gen, j, genome_script(g), genome=g)
        if p is None:
            break
        if progress:
            progress(run, p)
    run.finish()


def run_cmaes(run: Run, progress=None) -> None:
    import cma

    x0 = schema.to_unit(schema.default_genome())  # same starting design the LLM is shown
    es = cma.CMAEvolutionStrategy(x0, CMA_SIGMA0, {"seed": 20_000 + run.seed, "popsize": CMA_POPSIZE, "bounds": [0, 1],
                                                    "verbose": -9, "tolx": 0, "tolfun": 0, "tolfunhist": 0,
                                                    "tolstagnation": 10 ** 9})
    valid_scores: list[float] = []
    gen = 0
    while True:
        X = es.ask()
        outs = []
        for j, u in enumerate(X):
            g = schema.from_unit(u)
            p = run.propose(gen, j, genome_script(g), genome=g)
            if p is None:
                outs = None
                break
            outs.append(p)
            if progress:
                progress(run, p)
        if outs is None:
            break  # budget exhausted mid-generation: do not tell a partial population
        valid_scores += [p.rec["score"] for p in outs if p.rec["valid"]]
        bad = min(0.0, min(valid_scores)) if valid_scores else 0.0  # invalid proposals get the worst valid score or 0
        f = [-(p.rec["score"] if p.rec["valid"] else bad) for p in outs]
        es.tell(X, f)
        gen += 1
        if run.remaining <= 0:
            break
    run.finish()
