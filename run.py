"""
run.py
------
Interface to run the velocity picking algorithms on the test case bank.

Interactive (menus):
    python run.py

Direct:
    python run.py --list
    python run.py --case marmousi2_cdp06800 --algorithm HillClimbing --seed 42
    python run.py --case layers3 --algorithm RandomSearch --max-evals 5000

Check a saved solution (recomputes its objective value):
    python run.py --verify results/reports/<file>.json

The report (best solution and its value) is printed on screen and saved
in results/reports/ as .txt (readable) and .json (complete).
"""

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.cli import (ALGORITHMS, available_cases, format_report,   # noqa: E402
                     load_defaults, run_once, save_report, verify_solution)

CASES_DIR = os.path.join(ROOT, 'cases')
DEFAULTS  = os.path.join(ROOT, 'configs', 'interface_defaults.json')


def choose(prompt, options):
    """Numbered menu; returns the index chosen by the user."""
    for i, text in enumerate(options, 1):
        print(f"  [{i}] {text}")
    while True:
        answer = input(f"{prompt} [1-{len(options)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print("  Invalid option, try again.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true', help='list the test cases and exit')
    parser.add_argument('--case', help='test case name (see --list)')
    parser.add_argument('--algorithm', choices=sorted(ALGORITHMS))
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-evals', type=int, default=None,
                        help='override the evaluation budget')
    parser.add_argument('--verify', metavar='JSON', help='re-evaluate a saved solution')
    args = parser.parse_args()

    if args.verify:
        stored, recomputed = verify_solution(args.verify, ROOT)
        print(f"Stored objective value     : {stored:.10f}")
        print(f"Recomputed objective value : {recomputed:.10f}")
        print("OK: values match." if abs(stored - recomputed) < 1e-9
              else "WARNING: values differ.")
        return

    cases = available_cases(CASES_DIR)
    if not cases:
        sys.exit("No test cases found in cases/. Run: python scripts/generate_cases.py")

    if args.list:
        for _, name, desc in cases:
            print(f"  {name:30s} {desc}")
        return

    interactive = args.case is None or args.algorithm is None
    if interactive:
        print("\nTest cases:")
    if args.case is None:
        idx = choose("Choose a test case", [f"{n:30s} {d}" for _, n, d in cases])
    else:
        names = [n for _, n, _ in cases]
        if args.case not in names:
            sys.exit(f"Unknown case '{args.case}'. Use --list to see the cases.")
        idx = names.index(args.case)
    case_path = cases[idx][0]

    algorithms = sorted(ALGORITHMS)
    if args.algorithm is None:
        print("\nAlgorithms:")
        algorithm = algorithms[choose("Choose an algorithm", algorithms)]
    else:
        algorithm = args.algorithm

    seed = args.seed
    if seed is None:
        seed = 42
        if interactive:
            answer = input("Seed [42]: ").strip()
            seed = int(answer) if answer.lstrip('-').isdigit() else 42

    overrides = {'max_evals': args.max_evals} if args.max_evals else None
    defaults = load_defaults(DEFAULTS)

    print(f"\nRunning {algorithm} on {cases[idx][1]} (seed {seed})...")
    result = run_once(case_path, algorithm, seed, defaults, overrides, ROOT)

    print("\n" + format_report(result))
    txt, js = save_report(result, os.path.join(ROOT, defaults['reports_dir']))
    print(f"\nReport saved   : {os.path.relpath(txt, ROOT)}")
    print(f"Solution saved : {os.path.relpath(js, ROOT)}")
    print(f"Check it with  : python run.py --verify {os.path.relpath(js, ROOT)}")


if __name__ == '__main__':
    main()
