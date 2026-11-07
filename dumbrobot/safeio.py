"""Every file the designer loop writes goes through here: email addresses are redacted, then the written file is read back
and the run refuses to continue if one is still present (PLAN section 10)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

REDACTED = "[redacted-email]"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-']+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+")
EMAIL_RE_B = re.compile(EMAIL_RE.pattern.encode())


BINARY_SUFFIXES = {".npz", ".gif", ".pyc"}


class RedactionError(RuntimeError):
    pass


def redact(text: str) -> str:
    return EMAIL_RE.sub(REDACTED, text)


def write_text(path, text: str) -> Path:
    """Redact, write atomically, read back and verify."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(redact(text))
    os.replace(tmp, p)
    if EMAIL_RE.search(p.read_text()):
        raise RedactionError(f"email address still present in {p}")
    return p


def write_json(path, obj) -> Path:
    return write_text(path, json.dumps(obj, indent=1, default=str))


def append_line(path, line: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = redact(line)
    with p.open("a") as f:
        f.write(line.rstrip("\n") + "\n")
    if EMAIL_RE.search(line):
        raise RedactionError(f"email address still present in a line appended to {p}")


def scan_tree(root) -> list[str]:
    """Files under `root` that contain an email address. Used as a belt-and-braces check. Binary numeric outputs (.npz
    arrays, .gif pixels) are skipped: compressed bytes match the pattern by chance and they hold no text."""
    bad = []
    for p in sorted(Path(root).rglob("*")):
        if p.is_file() and not p.is_symlink() and p.suffix not in BINARY_SUFFIXES:
            if EMAIL_RE_B.search(p.read_bytes()):
                bad.append(str(p))
    return bad


def assert_clean(root) -> None:
    bad = scan_tree(root)
    if bad:
        raise RedactionError("email address found in: " + ", ".join(bad[:5]))
