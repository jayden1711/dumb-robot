# Critiques this project has to survive

Each objection is paired with the planned test. "Answered" means the test exists and its result is reported honestly,
including when the result is bad for the idea. Everything is in simulation.

## 1. "This is just RoboMorph / Debate2Create / Text2Robot"

Those works use LLMs for robot morphology or design generation [verify exact scope of each; not checked offline].
The claim here is narrower and different: with a **frozen, nearly empty brain**, how much task competence can be moved into
passive body mechanics, and which mechanical tricks does the designer invent? It is a study of the body-as-controller
trade, not a new morphology generator.

Test: write a related-work table before M1 results are reported, with each paper's brain (learned or fixed), design
space and evaluation. If any of them already ran a frozen-brain, LLM-versus-CMA-ES comparison, say so and reduce the
claim to a replication on these courses.

Status (M1): not yet written. The related-work table needs papers that cannot be checked offline; it stays an open item and
no M1 result is to be called new before it exists.

## 2. "The LLM is no better than CMA-ES"

This is the default expectation and the main thing under test.

Test: random search, CMA-ES and the LLM get the same proposal budget (PLAN section 7), same schema, same scorer, several
independent seeds, held-out re-evaluation of the winners. The LLM is reported as better only if it beats both baselines
on held-out score at equal budget with the seed spread shown. Extra ablations: LLM with scalar-only feedback (does the
failure log matter?), and LLM with the schema shuffled or the stated reasons removed. A null result is published as a null
result.

Status (M1): built and tested (`dumbrobot/report.py`): equal proposal budget with invalid proposals charged, 10 seeds per
baseline, held-out winners, medians with IQR and range, an exact permutation rank test, and a pre-declared claim rule (higher
median than both baselines and one-sided p below 0.05 against each). The scalar-only ablation (`llm_param_scalar`) is
implemented. Not built: the shuffled-schema and reasons-removed ablations. Known confounds, stated in the report: the LLM
also receives a plain-words task description, and its seed design costs one evaluation.

## 3. "It exploits sim physics"

MuJoCo contacts are soft constraints; a design can win by jittering, tunnelling, or abusing solver artefacts rather than
mechanics that would exist physically.

Test: (a) validity gate (mass, envelope, intersection, settle, drive); (b) every best design is re-run with a halved timestep, a
different solver (Newton vs PGS), perturbed friction (+/-20 percent) and held-out start yaw; a score that collapses is
tagged `sim-artefact` and excluded from claims; (c) a sanity check on energy: no design may extract net energy (motor
energy accounting has no regeneration credit, and total mechanical energy gain is checked against input); (d) a human reads
every finding in the findings list against the clip before it is counted.

Status (M1): (a) the validity gate and the static script guard exist; (b) built as `report.py` perturbation re-runs (halved
timestep, PGS solver, friction plus/minus 20 percent, held-out start headings) with the `sim-artefact` tag at a perturbed score
under half of nominal; (c) the no-regeneration energy model is in `sim.py`, a separate net-energy audit is not built; (d)
`findings_candidates.md` is generated for hand review, and nothing enters `findings.md` without a human reading it against the
clip. Observed in M0/M1 development, no numbers: contact-driven designs are sensitive to tiny parameter changes (rounding a
genome changed which start headings finish), so single-seed scores are not trusted.

## 4. "The tasks are trivial"

If Brain A with a plain wheeled box already completes the courses, nothing is learned.

Test: M0 reports (as a number in untracked results, not tracked docs) how a default and a deliberately mediocre design
fare on course 1 so that the course has headroom. A course enters the experiment set only if a default design does
not complete it and some hand-built design does; otherwise it is tightened. Course difficulty is a gradient (bumps of
increasing height) so partial progress is informative.

Status (M1): `tests/test_m1.py` asserts the headroom rule for course 2 (default fails, `examples/hand_made_wedge.py` finishes held-out
seeds) and the report prints reference designs on every course. Course 2 was first built with a centred wall and no valid
design could pass it, so it was loosened to a one-sided wall; the reverse of the usual "tighten".

## 5. "The brain isn't really dumb"

Brain B's bumper reflex is a sensor-conditional rule, and a clever body can turn "a bumper" into a rich sensor.

Test: Brain A has no inputs and is a constant; the primary results use Brain A only. Brain B is a stated one-bit reflex with a
fixed turn direction (no left/right logic) and is reported as a separate condition. A unit test asserts both brains
are parameter-free, unchanged across the repository history of the experiments (hash of `brains.py` is logged with every
run), and that Brain A's output is independent of time and state. A body that encodes computation in passive mechanics
is the intended outcome, not a violation, and the findings list says explicitly which tricks are "sensing" and which are
"mechanics".

Status (M1): all M1 runs use Brain A only. `brains.py` is hashed into each run's `meta.json`; a test asserts the M1 optimizer
and designer modules never reference the brains. The Brain B condition is not run before M2. The LLM sees "frozen brain, full
forward, no sensors" in its prompt, so a design that adds a sprung switch bumper is a body tricks candidate, not sensing,
because Brain A ignores sensors; the findings review has to say so for each such entry.
