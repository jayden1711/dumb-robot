# Third-party software and references

Nothing third-party is vendored in this repository. Versions are those pinned in the project virtualenv.
Licenses are recalled from upstream and could not be checked offline here; every license line is [verify] until confirmed
against the installed package metadata.

| Component | Use | Version | License |
|---|---|---|---|
| MuJoCo (`mujoco` Python bindings, mjSpec API) | physics, programmatic MJCF | 3.17.0 | Apache-2.0 [verify] |
| `cma` (CMA-ES, Nikolaus Hansen et al.) | baseline optimizer | 4.5.0 | BSD-3-Clause [verify] |
| NumPy | arrays | 2.5.4 | BSD-3-Clause [verify] |
| pytest | tests | 9.2.0 | MIT [verify] |
| Claude via the `claude` CLI | designer LLM (M1+), subscription CLI only | n/a | Anthropic terms of service, not a software dependency [verify] |

Planned, not yet used:

| Component | Use | Status |
|---|---|---|
| Blender (with the arena repo's tool scripts as reference) | evolution timelapse and final-bodies grid (M3) | the arena repo `robot-arena-sim-blender-render` has no LICENSE file; no code copied, tools only referenced. Re-check before M3 [verify] |

Related work cited in CRITIQUES.md (RoboMorph, Debate2Create, Text2Robot) is referenced by name only. Titles, authors,
venues and claims are not verified [verify]. Material densities and motor figures in `dumbrobot/catalog.py` are
approximate engineering values, not taken from any source [verify].
