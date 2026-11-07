"""M1 tests: corridor course, redaction, reply parsing, budget accounting, resumability, baselines, the LLM loop with a fake
designer (no `claude` call anywhere), report statistics."""
import json
from pathlib import Path

import numpy as np
import pytest

from dumbrobot import baselines, design_loop, report, runlib, safeio, schema
from dumbrobot.assembly import load_script
from dumbrobot.designer import ClaudeDesigner, DesignerError, FakeDesigner, claude_command, parse_reply
from dumbrobot.evaluate import evaluate
from dumbrobot.runlib import Run, check_source, genome_script
from dumbrobot.worlds import get_course

ROOT = Path(__file__).resolve().parents[1]
FAST = dict(courses=("flat",), train_seeds=(0,))  # keep loop tests quick


def rows(run):
    return [json.loads(x) for x in (run.dir / "ledger.jsonl").read_text().splitlines()]


# ---- course 2 -----------------------------------------------------------------------------------------------------------
def test_corridor_segments_are_contiguous_and_path_progress_is_monotone_in_window():
    c = get_course("corridor")
    xs = [(c.segments[i].x1, c.segments[i + 1].x0) for i in range(len(c.segments) - 1)]
    assert all(a == pytest.approx(b) for a, b in xs)
    assert c.segments[-1].x1 == pytest.approx(c.finish_x) and c.length > 5.0
    assert c.s_of(*c.path[0]) == pytest.approx(c.start_x)
    assert c.s_of(*c.path[-1], hint=c.finish_x - 0.1) == pytest.approx(c.finish_x, abs=0.02)
    # a robot on the far side of a wall cannot be credited with progress it has not reached: the window around the hint
    s_far = c.s_of(*c.path[-1], hint=c.start_x)
    assert s_far < c.start_x + 1.0


def test_corridor_headroom_default_fails_hand_design_finishes():
    g, b = load_script(ROOT / "dumbrobot" / "body_template.py")
    out = evaluate(b, g, courses=("corridor",), seeds=(0, 1))
    assert not any(e["score"]["done"] for e in out["per_course"]["corridor"]["episodes"])
    g, b = load_script(ROOT / "examples" / "hand_made_wedge.py")
    out = evaluate(b, g, courses=("corridor",), seeds=(1000, 1001))  # contact-driven: it finishes some seeds, not all
    assert out["valid"] and all(e["score"]["done"] for e in out["per_course"]["corridor"]["episodes"])
    assert all(s["status"] == "crossed" for s in out["per_course"]["corridor"]["episodes"][0]["result"]["segments"])


def test_corridor_failure_log_names_the_segment_and_contacts():
    g, b = load_script(ROOT / "dumbrobot" / "body_template.py")
    out = evaluate(b, g, courses=("corridor",), seeds=(0,))
    s = out["per_course"]["corridor"]["episodes"][0]["summary"]
    assert "wall: failed_here" in s and "touching" in s and "pillar" in s


# ---- redaction ----------------------------------------------------------------------------------------------------------
def test_redact_and_write(tmp_path):
    assert safeio.redact("mail a.b+c@example.co.uk now") == "mail [redacted-email] now"
    p = safeio.write_text(tmp_path / "x" / "r.txt", "hi person@example.org there")
    assert p.read_text() == "hi [redacted-email] there"
    assert safeio.scan_tree(tmp_path) == []


def test_run_refuses_if_a_written_file_still_contains_an_email(tmp_path, monkeypatch):
    monkeypatch.setattr(safeio, "redact", lambda t: t)  # simulate a redaction failure
    with pytest.raises(safeio.RedactionError):
        safeio.write_text(tmp_path / "bad.txt", "person@example.org")
    (tmp_path / "leak.txt").write_text("person@example.org")
    with pytest.raises(safeio.RedactionError):
        safeio.assert_clean(tmp_path)


# ---- claude command line ------------------------------------------------------------------------------------------------
def test_claude_command_is_isolated():
    cmd = claude_command()
    assert cmd[:2] == ["claude", "-p"]
    assert cmd[cmd.index("--setting-sources") + 1] == "project,local"
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert not any("api" in a.lower() and "key" in a.lower() for a in cmd)


def test_claude_designer_uses_fresh_empty_dir_under_var_tmp(monkeypatch):
    seen = {}

    def fake_run(cmd, input, cwd, capture_output, text, timeout):
        seen.update(cwd=Path(cwd), listing=list(Path(cwd).iterdir()), input=input)

        class R:
            returncode, stdout, stderr = 0, "reply", ""

        return R()

    monkeypatch.setattr("dumbrobot.designer.subprocess.run", fake_run)
    assert ClaudeDesigner().reply("prompt text") == "reply"
    assert seen["cwd"].parent == Path("/var/tmp") and seen["listing"] == [] and seen["input"] == "prompt text"
    assert not seen["cwd"].exists()  # emptied and removed


def test_claude_designer_failure_raises_designer_error(monkeypatch):
    class R:
        returncode, stdout, stderr = 1, "", "rate limit for person@example.org"

    monkeypatch.setattr("dumbrobot.designer.subprocess.run", lambda *a, **k: R())
    with pytest.raises(DesignerError) as e:
        ClaudeDesigner().reply("p")
    assert "example.org" not in str(e.value)


# ---- reply parsing ------------------------------------------------------------------------------------------------------
def test_parse_param_reply():
    txt = ("=== CHILD 1 ===\nPARENT: g000p0\nREASON: bigger wheels\nto climb.\n```json\n{\"wheel_r\": 0.08}\n```\n"
           "=== CHILD 2 ===\nPARENT: g000p0\nREASON: x\n```json\n[1,2]\n```\n=== CHILD 3 ===\nPARENT: g000p0\nREASON: none\n")
    ch = parse_reply(txt, "param")
    assert ch[0].changes == {"wheel_r": 0.08} and ch[0].reason == "bigger wheels\nto climb." and ch[0].parent == "g000p0"
    assert ch[1].error and ch[2].error and len(ch) == 3
    assert parse_reply("no structure here", "param") == []


def test_parse_free_reply():
    txt = "=== CHILD 1 ===\nPARENT: g001p2\nREASON: r\n```python\nGENOME = {}\n```\n"
    ch = parse_reply(txt, "free")
    assert ch[0].source == "GENOME = {}\n" and ch[0].parent == "g001p2"


def test_check_source_guards():
    assert check_source(genome_script(schema.default_genome())) is None
    assert check_source((ROOT / "dumbrobot" / "body_template.py").read_text()) is None
    for bad in ("import os\n", "from subprocess import run\n", "open('x')\n", "x = ().__class__\n", "eval('1')\n",
                "def f(:\n"):
        assert check_source(bad), bad


# ---- budget accounting, resume ------------------------------------------------------------------------------------------
def test_random_search_charges_every_proposal_and_resumes(tmp_path):
    r1 = Run(tmp_path, "random", 3, 6, **FAST)
    baselines.run_random(r1)
    assert r1.used == 6 and (r1.dir / "done.json").exists()
    assert any(not x["valid"] for x in rows(r1))  # invalid proposals are charged
    stamp = {p: p.stat().st_mtime_ns for p in r1.dir.glob("gen_*/proposal_*/score.json")}
    # a longer budget on the same directory: first 6 are reloaded, not re-simulated or re-charged
    r2 = Run(tmp_path, "random", 3, 9, **FAST)
    baselines.run_random(r2)
    assert r2.used == 9 and len(rows(r2)) == 9
    assert all(p.stat().st_mtime_ns == t for p, t in stamp.items())
    # same seed, fresh directory: identical proposal stream
    r3 = Run(tmp_path / "other", "random", 3, 9, **FAST)
    baselines.run_random(r3)
    assert [x["src"] for x in rows(r3)] == [x["src"] for x in rows(r2)]


def test_random_search_stops_at_budget(tmp_path):
    r = Run(tmp_path, "random", 0, 4, **FAST)
    baselines.run_random(r)
    assert len(rows(r)) == 4 and r.remaining == 0


def test_cmaes_budget_resume_and_invalid_fitness(tmp_path):
    r = Run(tmp_path, "cmaes", 1, 16, **FAST)
    baselines.run_cmaes(r)
    assert r.used == 16 and len(rows(r)) == 16
    assert {x["gen"] for x in rows(r)} == {0, 1}
    first = [x["src"] for x in rows(r)]
    r2 = Run(tmp_path, "cmaes", 1, 16, **FAST)  # resume a finished run: nothing re-charged
    baselines.run_cmaes(r2)
    assert len(rows(r2)) == 16 and [x["src"] for x in rows(r2)] == first
    r3 = Run(tmp_path, "cmaes", 1, 20, **FAST)  # extend into a partial generation: gen 2 is cut at 4 proposals
    baselines.run_cmaes(r3)
    assert len(rows(r3)) == 20 and [x["src"] for x in rows(r3)][:16] == first


def test_cmaes_same_seed_same_proposals(tmp_path):
    a, b = Run(tmp_path / "a", "cmaes", 2, 16, **FAST), Run(tmp_path / "b", "cmaes", 2, 16, **FAST)
    baselines.run_cmaes(a)
    baselines.run_cmaes(b)
    assert [x["src"] for x in rows(a)] == [x["src"] for x in rows(b)]
    assert baselines.CMA_VERSION_PIN == __import__("cma").__version__


# ---- the LLM loop with a fake designer -----------------------------------------------------------------------------------
class Boom:
    def reply(self, prompt, tag=""):
        raise AssertionError("designer must not be called on a resume")


class Flaky:
    def __init__(self, inner, fail_on):
        self.inner, self.fail_on, self.n = inner, fail_on, 0

    def reply(self, prompt, tag=""):
        self.n += 1
        if self.n == self.fail_on:
            raise DesignerError("rate limited")
        return self.inner.reply(prompt, tag)


def test_llm_loop_budget_counts_children_and_invalid_slots(tmp_path):
    calls = []
    run = Run(tmp_path, "llm_param", 0, 10, **FAST)
    design_loop.run_llm(run, FakeDesigner(0, calls), gen_size=3)
    rs = rows(run)
    assert len(rs) == 10 and run.remaining == 0 and len(calls) == 3  # 1 seed design + 3 generations of 3
    assert rs[0]["pid"] == "g000p0" and rs[0]["valid"]
    bad = {r["pid"] for r in rs if not r["valid"]}
    assert bad  # out-of-range and malformed fake children were charged, not free
    why = [json.loads((run.pdir(r["gen"], r["j"]) / "validity.json").read_text()) for r in rs if not r["valid"]]
    assert any("no_proposal" in str(w) for w in why) and any("schema" in str(w) for w in why)
    for g in (1, 2, 3):
        d = run.dir / f"gen_{g:03d}"
        assert (d / "prompt.txt").exists() and (d / "designer_reply.txt").exists() and (d / "summary.json").exists()
    assert json.loads((run.dir / "population.json").read_text())
    for p in run.dir.glob("gen_*/proposal_*"):
        assert (p / "body.py").exists() and (p / "score.json").exists() and (p / "validity.json").exists()
        assert (p / "failure_log.json").exists()


def test_llm_loop_resume_needs_no_new_calls_and_survives_designer_failure(tmp_path):
    run = Run(tmp_path, "llm_param", 1, 10, **FAST)
    inner = FakeDesigner(1)
    with pytest.raises(DesignerError):
        design_loop.run_llm(run, Flaky(inner, 2), gen_size=3)
    assert run.used == 4  # seed + one generation; the failed call charged nothing
    run2 = Run(tmp_path, "llm_param", 1, 10, **FAST)
    design_loop.run_llm(run2, Flaky(inner, 0), gen_size=3)
    assert run2.used == 10
    run3 = Run(tmp_path, "llm_param", 1, 10, **FAST)
    design_loop.run_llm(run3, Boom(), gen_size=3)  # everything cached
    assert len(rows(run3)) == 10


def test_short_reply_leaves_missing_slots_charged(tmp_path):
    class OneChild:
        def reply(self, prompt, tag=""):
            return "=== CHILD 1 ===\nPARENT: g000p0\nREASON: r\n```json\n{\"wheel_r\": 0.05}\n```\n"

    run = Run(tmp_path, "llm_param", 0, 4, **FAST)
    design_loop.run_llm(run, OneChild(), gen_size=3)
    assert len(rows(run)) == 4
    assert sum(1 for r in rows(run) if r["why_invalid"] == ["no_proposal"]) == 2


def test_replies_are_redacted_and_run_stays_clean(tmp_path):
    class Leaky:
        def reply(self, prompt, tag=""):
            return ("=== CHILD 1 ===\nPARENT: g000p0\nREASON: ask person@example.org about it\n"
                    "```json\n{\"wheel_r\": 0.05}\n```\n")

    run = Run(tmp_path, "llm_param", 0, 4, **FAST)
    design_loop.run_llm(run, Leaky(), gen_size=3)
    assert safeio.scan_tree(tmp_path) == []
    assert "[redacted-email]" in (run.dir / "gen_001" / "designer_reply.txt").read_text()
    assert "[redacted-email]" in (run.pdir(1, 0) / "reasoning.md").read_text()


def test_prompt_has_failure_logs_and_scalar_ablation_does_not(tmp_path):
    runs = {}
    for fb, arm in (("logs", "llm_param"), ("scalar", "llm_param_scalar")):
        run = Run(tmp_path, arm, 0, 4, courses=("bumps", "corridor"), train_seeds=(0,))
        design_loop.run_llm(run, FakeDesigner(0), feedback=fb, gen_size=3)
        runs[fb] = (run.dir / "gen_001" / "prompt.txt").read_text()
    assert "Failure log" in runs["logs"] and "bump_1" in runs["logs"] and "wall:" in runs["logs"]
    assert "touching:" in runs["logs"] and "ALLOWED_PARENTS: g000p0" in runs["logs"]
    assert "Failure log" not in runs["scalar"] and "touching:" not in runs["scalar"]
    assert "validity gate" in runs["logs"].lower() and "frozen" in runs["logs"]


def test_free_form_arm_runs_scripts_in_a_subprocess(tmp_path):
    run = Run(tmp_path, "llm_free", 0, 4, courses=("flat",), train_seeds=(0,), subprocess_eval=True)
    design_loop.run_llm(run, FakeDesigner(0), mode="free", gen_size=3)
    assert len(rows(run)) == 4
    assert rows(run)[0]["valid"] and "def build" in (run.pdir(0, 0) / "body.py").read_text()
    assert "BODY API" in (run.dir / "gen_001" / "prompt.txt").read_text()


def test_forbidden_script_is_rejected_without_running(tmp_path):
    p = tmp_path / "evil.py"
    p.write_text("import os\nGENOME = {}\ndef build(w, g):\n    os.system('true')\n")
    rec = runlib.evaluate_to_dir_subprocess(p, tmp_path / "o1", courses=("flat",), seeds=(0,), timeout_s=60)
    assert not rec["valid"] and rec["why_invalid"] == ["build"] and "not allowed" in rec["validity_detail"]


def test_runaway_script_hits_the_wall_clock_limit(tmp_path):
    p = tmp_path / "loop.py"
    p.write_text("GENOME = {}\nwhile True:\n    pass\n")
    rec = runlib.evaluate_to_dir_subprocess(p, tmp_path / "o2", courses=("flat",), seeds=(0,), timeout_s=3)
    assert not rec["valid"] and "wall-clock" in rec["validity_detail"]


# ---- brain stays out of the optimizer code -------------------------------------------------------------------------------
def test_m1_code_does_not_touch_brains():
    for name in ("runlib.py", "baselines.py", "designer.py", "design_loop.py", "report.py", "safeio.py"):
        assert "brains" not in (ROOT / "dumbrobot" / name).read_text(), name


def test_brains_hash_is_logged(tmp_path):
    from dumbrobot import m1

    assert len(m1.brains_hash()) == 16


# ---- statistics ---------------------------------------------------------------------------------------------------------
def test_mann_whitney_exact():
    r = report.mann_whitney([5, 6, 7, 8, 9], [1, 2, 3, 4, 5.5])
    assert r["U"] == 24.0 and r["p_greater"] < 0.01 and r["p_greater"] == pytest.approx(r["p_two_sided"] / 2)
    same = report.mann_whitney([1, 2, 3], [1, 2, 3])
    assert same["p_greater"] > 0.4 and same["p_two_sided"] == pytest.approx(1.0)
    assert report.mann_whitney([1, 2], [])["U"] is None


def test_spread_and_curve():
    s = report.spread([1, 2, 3, 4])
    assert s["median"] == 2.5 and s["min"] == 1 and s["max"] == 4
    assert report.curve([dict(score=0.1), dict(score=0.0), dict(score=0.3)], 5) == [0.1, 0.1, 0.3, 0.3, 0.3]


def test_report_end_to_end_on_tiny_runs(tmp_path):
    for arm, fn in (("random", baselines.run_random), ("cmaes", baselines.run_cmaes)):
        for sd in (0, 1):
            run = Run(tmp_path, arm, sd, 8, **FAST)
            (run.dir / "meta.json").write_text(json.dumps(dict(budget=8)))
            fn(run)
    text = report.build_report(tmp_path, with_perturb=False, budget=8)
    assert "Every run (nothing dropped)" in text and "| random | 0 |" in text and "| cmaes | 1 |" in text
    assert "Claim check" in text and (tmp_path / "findings_candidates.md").exists()
    assert safeio.scan_tree(tmp_path) == []
