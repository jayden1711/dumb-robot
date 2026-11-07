"""Designers: the thing that turns a prompt into proposals. `ClaudeDesigner` shells out to the subscription CLI
(`claude -p`), never an API. `FakeDesigner` is a deterministic stand-in for tests and dry runs."""
from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import schema
from .safeio import redact

SYSTEM_PROMPT = ("You are a mechanical design assistant working on a simulated robot (simulation only). "
                 "Answer only in the exact format requested by the user message. You have no tools; everything you need is in "
                 "the message.")
SCRATCH_PARENT = Path("/var/tmp")


class DesignerError(RuntimeError):
    """The designer could not produce a reply (CLI failure, rate limit, timeout). The run stops cleanly and can resume."""


def claude_command(model: str | None = None, effort: str | None = None) -> list[str]:
    cmd = ["claude", "-p", "--setting-sources", "project,local", "--tools", "", "--strict-mcp-config", "--mcp-config",
           '{"mcpServers":{}}', "--no-session-persistence", "--disable-slash-commands", "--output-format", "text",
           "--system-prompt", SYSTEM_PROMPT]
    if model:
        cmd += ["--model", model]
    if effort:
        cmd += ["--effort", effort]
    return cmd


class ClaudeDesigner:
    """One `claude -p` call per generation, in a fresh empty directory under /var/tmp, prompt on stdin."""

    def __init__(self, model: str | None = None, effort: str | None = None, timeout_s: float = 900.0):
        self.model, self.effort, self.timeout_s = model, effort, timeout_s

    def reply(self, prompt: str, tag: str = "") -> str:
        cwd = SCRATCH_PARENT / f"dumbrobot_llm_{uuid.uuid4().hex[:12]}"
        cwd.mkdir(mode=0o700)
        try:
            cp = subprocess.run(claude_command(self.model, self.effort), input=prompt, cwd=cwd, capture_output=True,
                                text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise DesignerError(f"claude -p timed out after {self.timeout_s:.0f} s") from e
        except FileNotFoundError as e:
            raise DesignerError("claude CLI not found on PATH") from e
        finally:
            try:
                os.rmdir(cwd)  # empty by construction; left in place (and ignored) if the CLI wrote something
            except OSError:
                pass
        if cp.returncode != 0 or not cp.stdout.strip():
            raise DesignerError(f"claude -p failed (exit {cp.returncode}): " + redact((cp.stderr or cp.stdout).strip()[-500:]))
        return cp.stdout


# ---- reply format -------------------------------------------------------------------------------------------------------
@dataclass
class Child:
    parent: str | None
    reason: str
    changes: dict | None = None
    source: str | None = None
    raw: str = ""
    error: str | None = None


_HEAD = re.compile(r"^\s*={2,}\s*CHILD\s*(\d+)\s*={2,}\s*$", re.I | re.M)
_FENCE = re.compile(r"```[ \t]*([A-Za-z]*)[ \t]*\n(.*?)\n[ \t]*```", re.S)


def parse_reply(text: str, mode: str) -> list[Child]:
    """mode 'param': each child has PARENT, REASON and a ```json block of parameter changes over the parent.
    mode 'free': each child has PARENT, REASON and a ```python block with the whole body script."""
    heads = list(_HEAD.finditer(text))
    out = []
    for i, h in enumerate(heads):
        block = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        pm = re.search(r"^\s*PARENT:\s*([A-Za-z0-9_\-]+)", block, re.M)
        rm = re.search(r"^\s*REASON:\s*(.*?)(?=^\s*```|\Z)", block, re.M | re.S)
        reason = rm.group(1).strip() if rm else ""
        fm = _FENCE.search(block)
        c = Child(pm.group(1) if pm else None, reason, raw=block.strip())
        if not fm:
            c.error = "no fenced code block in this child"
        elif mode == "param":
            try:
                ch = json.loads(fm.group(2))
                if not isinstance(ch, dict):
                    raise ValueError("not a JSON object")
                c.changes = ch
            except ValueError as e:
                c.error = f"changes are not a valid JSON object: {e}"
        else:
            c.source = fm.group(2) + "\n"
        out.append(c)
    return out


# ---- fake designer (tests, dry runs) ----------------------------------------------------------------------------------
class FakeDesigner:
    """Deterministic: reads CHILDREN_REQUESTED / ALLOWED_PARENTS / MODE from the prompt and answers in the reply format.
    Child k of a reply: 0 = mutate 2-3 random parameters in range, 1 = deliberately out-of-range value (schema reject),
    2 = no code block (malformed), then repeat."""

    def __init__(self, seed: int = 0, calls: list | None = None):
        self.rng = np.random.default_rng(seed)
        self.calls = [] if calls is None else calls

    def reply(self, prompt: str, tag: str = "") -> str:
        self.calls.append(tag)
        n = int(re.search(r"CHILDREN_REQUESTED:\s*(\d+)", prompt).group(1))
        parents = re.search(r"ALLOWED_PARENTS:\s*(.*)", prompt).group(1).split()
        mode = re.search(r"MODE:\s*(\w+)", prompt).group(1)
        out = []
        for k in range(n):
            parent = parents[0]
            if k % 3 == 2:
                out.append(f"=== CHILD {k + 1} ===\nPARENT: {parent}\nREASON: forgot the code block\n")
                continue
            changes = {}
            for p in self.rng.choice(schema.PARAMS, size=3, replace=False):
                if p.kind == "enum":
                    changes[p.name] = str(self.rng.choice(p.choices))
                elif p.kind == "int":
                    changes[p.name] = int(self.rng.integers(p.lo, p.hi + 1))
                else:
                    changes[p.name] = round(float(self.rng.uniform(p.lo, p.hi)), 4)
            if k % 3 == 1:
                changes["wheel_r"] = 5.0
            if mode == "param":
                body = "```json\n" + json.dumps(changes) + "\n```"
            else:
                g = {**schema.default_genome(), **{k2: v for k2, v in changes.items() if k2 != "wheel_r"}}
                body = ("```python\nfrom dumbrobot.body_template import build  # noqa: F401\n\nGENOME = " + json.dumps(g)
                        + "\n```")
            out.append(f"=== CHILD {k + 1} ===\nPARENT: {parent}\nREASON: fake designer, random change of "
                       f"{', '.join(changes)}\n{body}\n")
        return "\n".join(out)
