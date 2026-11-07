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

## 2. "The LLM is no better than CMA-ES"

This is the default expectation and the main thing under test.

Test: random search, CMA-ES and the LLM get the same proposal budget (PLAN section 7), same schema, same scorer, several
independent seeds, held-out re-evaluation of the winners. The LLM is reported as better only if it beats both baselines
on held-out score at equal budget with the seed spread shown. Extra ablations: LLM with scalar-only feedback (does the
failure log matter?), and LLM with the schema shuffled or the stated reasons removed. A null result is published as a null
result.

## 3. "It exploits sim physics"

MuJoCo contacts are soft constraints; a design can win by jittering, tunnelling, or abusing solver artefacts rather than
mechanics that would exist physically.

Test: (a) validity gate (mass, envelope, intersection, settle, drive); (b) every best design is re-run with a halved timestep, a
different solver (Newton vs PGS), perturbed friction (+/-20 percent) and held-out start yaw; a score that collapses is
tagged `sim-artefact` and excluded from claims; (c) a sanity check on energy: no design may extract net energy (motor
energy accounting has no regeneration credit, and total mechanical energy gain is checked against input); (d) a human reads
every finding in the findings list against the clip before it is counted.

## 4. "The tasks are trivial"

If Brain A with a plain wheeled box already completes the courses, nothing is learned.

Test: M0 reports (as a number in untracked results, not tracked docs) how a default and a deliberately mediocre design
fare on course 1 so that the course has headroom. A course enters the experiment set only if a default design does
not complete it and some hand-built design does; otherwise it is tightened. Course difficulty is a gradient (bumps of
increasing height) so partial progress is informative.

## 5. "The brain isn't really dumb"

Brain B's bumper reflex is a sensor-conditional rule, and a clever body can turn "a bumper" into a rich sensor.

Test: Brain A has no inputs and is a constant; the primary results use Brain A only. Brain B is a stated one-bit reflex with a
fixed turn direction (no left/right logic) and is reported as a separate condition. A unit test asserts both brains
are parameter-free, unchanged across the repository history of the experiments (hash of `brains.py` is logged with every
run), and that Brain A's output is independent of time and state. A body that encodes computation in passive mechanics
is the intended outcome, not a violation, and the findings list says explicitly which tricks are "sensing" and which are
"mechanics".
