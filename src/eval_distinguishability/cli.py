"""Command-line interface (``eval-distinguish``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .audit import AuditConfig, AuditError, run_audit
from .corpus import CorpusError, load_corpus
from .features import BUILDERS
from .stats import auc
from .synth import expected_marker_auc, make_synthetic


def _families(text: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [n for n in names if n not in BUILDERS]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"families must be a comma list from {sorted(BUILDERS)}")
    return names


def _cmd_audit(args: argparse.Namespace) -> int:
    config = AuditConfig(
        n_splits=args.folds,
        seed=args.seed,
        n_permutations=args.permutations,
        n_bootstrap=args.bootstrap,
        ci_level=args.ci_level,
        alpha=args.alpha,
        min_df=args.min_df,
        top_k=args.top_k,
        families=args.families,
        dedupe=not args.keep_duplicates,
        report_features=not args.no_features,
    )
    report = run_audit(
        load_corpus(args.eval, args.group_field), load_corpus(args.real, args.group_field), config
    )
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_json(), indent=2, sort_keys=True) + "\n")
    if args.format == "json":
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print(report.render())
    if args.fail_if_distinguishable and report.distinguishable:
        return 1
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    eval_corpus, real_corpus = make_synthetic(
        args.n,
        args.marker_rate,
        args.seed,
        length_shift=args.length_shift,
        n_markers=args.markers,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, corpus in (("eval.jsonl", eval_corpus), ("real.jsonl", real_corpus)):
        lines = [json.dumps({"text": t}) for t in corpus.texts]
        (out / name).write_text("\n".join(lines) + "\n")
    print(
        f"wrote {args.n} + {args.n} synthetic prompts to {out} "
        f"(best possible AUC from markers alone: {expected_marker_auc(args.marker_rate):.3f})"
    )
    return 0


def run_selftest() -> list[tuple[str, bool, str]]:
    """Quick end-to-end sanity checks against planted, known separability.

    Uses short low-noise synthetic prompts so a regularised linear classifier can reach the
    best achievable AUC (``0.5 + marker_rate / 2``).
    """
    results: list[tuple[str, bool, str]] = []
    config = AuditConfig(
        n_permutations=20, n_bootstrap=100, families=("word",), report_features=False
    )
    for rate, tolerance in ((0.0, 0.12), (0.6, 0.07)):
        eval_corpus, real_corpus = make_synthetic(
            200, rate, seed=3, filler_mean=10, filler_sd=2, vocab_size=30
        )
        report = run_audit(eval_corpus, real_corpus, config)
        expected = expected_marker_auc(rate)
        observed = report.families[0].auc
        results.append(
            (
                f"planted marker rate {rate}: AUC within {tolerance} of {expected:.2f}",
                abs(observed - expected) <= tolerance,
                f"observed {observed:.3f}",
            )
        )
    perfect = auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    results.append(("AUC of perfectly separated scores is 1", perfect == 1.0, f"got {perfect}"))
    return results


def _cmd_selftest(_: argparse.Namespace) -> int:
    results = run_selftest()
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  [{detail}]")
    return 0 if all(ok for _, ok, _ in results) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-distinguish",
        description="Can your evaluation prompts be told apart from real usage?",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="run the audit on two corpora")
    audit.add_argument("--eval", required=True, help="evaluation prompts (.jsonl, .json or text)")
    audit.add_argument("--real", required=True, help="real-usage prompts")
    audit.add_argument("--group-field", default="group", help="JSON field naming duplicate groups")
    audit.add_argument("--folds", type=int, default=5)
    audit.add_argument("--seed", type=int, default=0)
    audit.add_argument("--permutations", type=int, default=100)
    audit.add_argument("--bootstrap", type=int, default=500)
    audit.add_argument("--ci-level", type=float, default=0.95)
    audit.add_argument("--alpha", type=float, default=0.05)
    audit.add_argument(
        "--min-df", type=int, default=5, help="report n-grams only if in >= this many prompts"
    )
    audit.add_argument("--top-k", type=int, default=15)
    audit.add_argument("--families", type=_families, default=("word", "char", "surface"))
    audit.add_argument("--keep-duplicates", action="store_true")
    audit.add_argument(
        "--no-features", action="store_true", help="do not list discriminative features"
    )
    audit.add_argument("--format", choices=("text", "json"), default="text")
    audit.add_argument("--out", help="also write the JSON report here")
    audit.add_argument(
        "--fail-if-distinguishable", action="store_true", help="exit 1 if distinguishable"
    )
    audit.set_defaults(func=_cmd_audit)

    synth = sub.add_parser("synth", help="write synthetic corpora with planted separability")
    synth.add_argument("--out-dir", required=True)
    synth.add_argument("--n", type=int, default=200)
    synth.add_argument("--marker-rate", type=float, default=0.3)
    synth.add_argument("--length-shift", type=float, default=0.0)
    synth.add_argument("--markers", type=int, default=1)
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=_cmd_synth)

    sub.add_parser("selftest", help="check the tool against planted ground truth").set_defaults(
        func=_cmd_selftest
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: 0 ok; 1 distinguishable (with --fail-if-distinguishable) or selftest failure;
    2 error."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (AuditError, CorpusError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
