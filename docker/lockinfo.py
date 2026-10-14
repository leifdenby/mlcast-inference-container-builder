"""Read what the base image build needs out of pyproject.toml and uv.lock.

Used by docker/Dockerfile.base's build stage, so the Dockerfile never repeats a URL,
commit or version that the lock already pins. Standard library only (it runs on the
build stage's plain Python).

    python docker/lockinfo.py mlcast             # "<git url> <commit> <version>"
    python docker/lockinfo.py torch-index cpu    # the PyTorch index URL for a torch extra
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def mlcast_source(lock_path: Path) -> tuple[str, str, str]:
    """Return mlcast's git URL, commit and version as uv.lock pins them."""
    lock = tomllib.loads(lock_path.read_text())
    package = next(p for p in lock["package"] if p["name"] == "mlcast")
    git = package["source"]["git"]  # https://github.com/org/repo?rev=<rev>#<commit>
    url, _, commit = git.partition("#")
    return url.split("?")[0], commit, package["version"]


def torch_index(pyproject_path: Path, extra: str) -> str:
    """Return the URL of the PyTorch index that ``[tool.uv.sources]`` maps a torch extra to."""
    uv = tomllib.loads(pyproject_path.read_text())["tool"]["uv"]
    name = next(s["index"] for s in uv["sources"]["torch"] if s.get("extra") == extra)
    return next(i["url"] for i in uv["index"] if i["name"] == name)


def main(argv: list[str]) -> int:
    if argv[:1] == ["mlcast"]:
        print(*mlcast_source(Path("uv.lock")))
        return 0
    if argv[:1] == ["torch-index"] and len(argv) == 2:
        print(torch_index(Path("pyproject.toml"), argv[1]))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
