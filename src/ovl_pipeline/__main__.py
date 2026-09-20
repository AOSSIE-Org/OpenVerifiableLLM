"""Explicit fixture-only command line; expensive production actions are not implicit."""
import argparse
from pathlib import Path
import sys

from .canonical import EvidenceError, canonical, parse_json, read_json
from .fixture import run_fixture, verify_fixture
from .training import TrustPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("fixture")
    run.add_argument("--source", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--trust-policy", type=Path, required=True)
    verify = sub.add_parser("verify-fixture")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--trust-policy", type=Path, required=True)
    acquisition = sub.add_parser("acquire-wikipedia", help="Download and fully verify a completed dated monolithic article dump")
    acquisition.add_argument("--status", type=Path, required=True, help="Retained official dumpstatus.json")
    acquisition.add_argument("--date", required=True)
    acquisition.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            result = run_fixture(args.source, args.output, args.trust_policy)
        elif args.command == "acquire-wikipedia":
            from .acquisition import acquire, wikipedia_source
            spec = wikipedia_source(read_json(args.status, canonical_required=False), args.date)
            result = acquire(spec, args.output)
        else:
            policy = TrustPolicy(**parse_json(args.trust_policy.read_bytes(), canonical_required=True))
            result = verify_fixture(args.bundle, policy)
        print(canonical(result).decode())
        return 0
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as e:
        print(canonical({"result": "FAIL", "reason": str(e)}).decode())
        return 1


if __name__ == "__main__":
    sys.exit(main())
