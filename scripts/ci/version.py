"""Reads, stamps and bumps the version in pyproject.toml and kohakutt.__version__.
Usage: version.py read | nightly | bump patch|minor|major | write <version>
(nightly prints X.Y.(Z+1).devYYYYmmddHHMMSS).
"""

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
# The [project] table's own key: a line-anchored `version = "..."`.
PATTERN = re.compile(r'(?m)^(version\s*=\s*")([^"]+)(")')
# The package's own copy, kept equal by write().
PACKAGE = PYPROJECT.parent / "src/kohakutt/__init__.py"
ATTRIBUTE = re.compile(r'(?m)^(__version__\s*=\s*")([^"]+)(")')


def read():
    match = PATTERN.search(PYPROJECT.read_text(encoding="utf-8"))
    if not match:
        raise SystemExit("no version found in pyproject.toml")
    return match.group(2)


def write(version):
    text = PYPROJECT.read_text(encoding="utf-8")
    new, count = PATTERN.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit("no version found in pyproject.toml")
    PYPROJECT.write_text(new, encoding="utf-8")
    text = PACKAGE.read_text(encoding="utf-8")
    new, count = ATTRIBUTE.subn(rf"\g<1>{version}\g<3>", text, count=1)
    if count != 1:
        raise SystemExit("no __version__ found in the package")
    PACKAGE.write_text(new, encoding="utf-8")


def base(version):
    core = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not core:
        raise SystemExit(f"cannot parse version {version!r}")
    return tuple(int(g) for g in core.groups())


def bump(version, part):
    major, minor, patch = base(version)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def nightly(version):
    """The PEP 440 dev release of the next patch, stamped with the UTC time."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    major, minor, patch = base(version)
    return f"{major}.{minor}.{patch + 1}.dev{stamp}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("read", "nightly", "bump", "write"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()
    current = read()
    if args.action == "read":
        print(current)
    elif args.action == "nightly":
        print(nightly(current))
    elif args.action == "bump":
        print(bump(current, args.value or "patch"))
    else:
        if not args.value:
            raise SystemExit("write needs a version")
        write(args.value)
        print(args.value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
