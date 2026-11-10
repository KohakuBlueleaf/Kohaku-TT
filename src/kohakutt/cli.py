"""The `ktt` command line: diagnostics and setup."""

import argparse
import json
import sys

from kohakutt.core.doctor import format_report, run_doctor


def main(argv=None):
    """Parses `argv` and runs the chosen subcommand; returns the exit code."""
    parser = argparse.ArgumentParser(prog="ktt", description="Kohaku-TT tools")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="read-only report of every card")
    doctor.add_argument("--json", action="store_true", help="print JSON")

    sub.add_parser("sfpi", help="fetch the RISC-V kernel compiler into the cache")

    args = parser.parse_args(argv)
    if args.command == "sfpi":
        from kohakutt.torch_backend import registration

        print(registration.fetch_sfpi())
        return 0
    if args.command == "doctor":
        report = run_doctor()
        print(json.dumps(report, indent=2) if args.json else format_report(report))
        return 0 if report["cards"] else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
