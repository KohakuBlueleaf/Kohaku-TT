"""Release notes for a version: the commits since the previous release tag.
Usage: release_notes.py --version 1.0.0
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False, cwd=ROOT
    ).stdout.strip()


def previous_tag(version):
    """The newest v* tag that is not this version's own tag."""
    for tag in git("tag", "--list", "v*", "--sort=-creatordate").split():
        if tag != f"v{version}":
            return tag
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    prev = previous_tag(args.version)
    span = f"{prev}..HEAD" if prev else "HEAD"
    log = git("log", "--no-merges", "--pretty=* %s (%h)", span)
    pin = (ROOT / "packaging" / "tt-metal.pin").read_text().strip()
    since = f" since {prev}" if prev else ""
    print(
        f"## Kohaku-TT {args.version}\n\nBuilt against tt-metal {pin}.\n\n"
        f"### Changes{since}\n\n{log or '* none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
