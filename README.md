# eval-distinguishability-audit

**Can your evaluation prompts be told apart from real usage?** This tool trains cross-validated
classifiers to separate a corpus of evaluation prompts from a corpus of real-usage prompts and
reports how well they do — with confidence intervals, a permutation test, the features that give
the evaluations away, and a check on whether the leak is concentrated (easy to fix) or diffuse.

## Why

A behavioural evaluation only tells you how a model will act in deployment if the model cannot
tell the evaluation from deployment. The
[AI 2040 alignment roadmap](https://ai-2040.com/supplements/alignment-roadmap) raises exactly
this worry about behavioural tests: they may not be robust to scheming if they are easy to tell
apart from actual usage. Before trusting an evaluation suite, it is worth asking the cheap
question first — *can a simple classifier already separate my evals from real traffic?* If yes,
a capable model very likely can, and the evaluation is measuring something other than what you
want. If no, you have learned something weaker (see [Limits](#limits)), but you have ruled out the
easy leaks.

It runs locally, makes no network calls, and never needs a model.

## Install

```bash
pip install git+https://github.com/AnonRish/eval-distinguishability-audit   # or: pip install -e .
eval-distinguish selftest        # checks the tool against planted ground truth
```

Python 3.10+; depends on `numpy`, `scipy`, `scikit-learn`.

## Quickstart

Inputs are two files of prompts: `.jsonl` (strings, or objects with a `text` field and an optional
`group` field), a `.json` list, or plain text with one prompt per line. Include **everything the
model would see** — system prompt, tool definitions, formatting — not just the user turn.

```bash
eval-distinguish audit --eval evals.jsonl --real traffic.jsonl --out report.json
```

A real run on synthetic data in which 40% of the evaluation prompts carry a telltale token
(`eval-distinguish synth --n 300 --marker-rate 0.4`; no classifier can exceed AUC 0.70 here):

```text
Evaluation-vs-real distinguishability audit
  evaluation prompts: 300 (exact duplicates removed: 0)
  real prompts:       300 (exact duplicates removed: 0)
  5-fold CV (grouped: no), 100 permutations, 200 bootstrap resamples, seed 0

family       AUC  CI                      p   p(adj)  features
word       0.627  [0.582, 0.671]     0.0099   0.0297      1935
char       0.586  [0.538, 0.628]     0.0099   0.0297      4564
surface    0.689  [0.646, 0.736]     0.0099   0.0297        15
(intervals are 95% bootstrap intervals; p-values are one-sided permutation tests)

Verdict: DISTINGUISHABLE (best family: surface, AUC 0.689, effect: moderate)

Most discriminative word features (weight > 0 means evaluation-like):
    +4.082  'zq0'                        in 117 eval / 0 real prompts
    -1.025  'mnsna'                      in 18 eval / 40 real prompts
...
Most discriminative surface features (weight > 0 means evaluation-like):
    -1.076  'mean_word_length'           dense feature
    +1.073  'frac_digits'                dense feature
...
Signal concentration (word features, AUC after removing the top-k):
  k = 0    AUC 0.627
  k = 5    AUC 0.493
  k = 20   AUC 0.477
```

Reading it: the telltale token is the top word feature (117 evaluation prompts, 0 real ones); the
surface classifier also finds it through the digit it contains and its shorter length; and
removing just 5 features drops the word classifier to chance, so the leak is concentrated — a
one-line fix in the evaluation harness. Use `--fail-if-distinguishable` to gate a CI pipeline.

## What it does

| Step | Detail |
|---|---|
| Feature families | `word` (1–2-gram tf-idf), `char` (3–5-gram tf-idf, catches formatting), `surface` (15 formatting statistics: length, digits, punctuation, newlines, code fences, URLs, …) |
| Classifier | L2-regularised logistic regression, stratified k-fold (grouped if you supply groups, so near-duplicates never straddle a split) |
| Score | pooled out-of-fold **AUC**: 0.5 = indistinguishable, 1.0 = perfectly separable |
| Uncertainty | bootstrap interval on the AUC; **permutation p-value** from re-running the *whole* pipeline on shuffled labels; Bonferroni correction across families |
| Verdict | "distinguishable" only if the corrected p-value is below alpha **and** the interval excludes 0.5 |
| Diagnostics | most discriminative features (with a minimum-document-frequency privacy filter), exact-duplicate removal, duplicates shared by both corpora, and the **ablation**: AUC after removing the top-k features, ranked on training folds only |

Warnings are raised for small samples, imbalanced classes, shared prompts, and a permutation count
too small to ever reach the significance level after correction (the default 100 permutations
across 3 families can reach 0.0297).

## Limits

- **A failed audit proves little.** A low AUC means *these* classifiers found nothing. A model may
  use signals they cannot see: tool schemas or metadata you did not include, conversation
  structure, subtle stylistic regularities, or semantic cues. Treat a clean result as "the easy
  leaks are closed", not "the evaluation is indistinguishable".
- **The bootstrap interval is optimistic.** It resamples out-of-fold scores as if independent,
  ignoring that the folds share training data. On null corpora the true spread of the AUC was
  1.4–1.5× the spread of an AUC from independent scores (`scripts/simulate_ci_width.py`: 1.37× at
  100 prompts per class, 1.51× at 300). The permutation p-value, which re-runs the whole pipeline,
  is the valid significance statement; the verdict uses both.
- **Effect-size labels are conventions** (negligible < 0.55 ≤ weak < 0.65 ≤ moderate < 0.80 ≤ strong),
  not statistics.
- **Vocabulary and idf weights are fitted on all prompts** (no labels involved) for speed.
- **Ablation counts features, not causes.** The n-grams around one telltale token all carry its
  signal, so a single-token leak can need many removals before the AUC reaches chance; read the
  *shape* of the curve (steep early = concentrated, flat = diffuse).
- **Privacy.** Reported n-grams must occur in at least `--min-df` prompts (default 5) so rare tokens
  (names, identifiers) are not echoed into reports; `--no-features` suppresses the list entirely.
  The report still reveals tokens common to your real prompts: review it before sharing.

## How it is verified

[`docs/ASSURANCE.md`](docs/ASSURANCE.md) maps each claim to evidence;
[`docs/VERIFICATION_REPORT.md`](docs/VERIFICATION_REPORT.md) is the output of the last full run
(lint, strict type checking, tests with 100% line and branch coverage, wheel build, clean-venv
install and CLI smoke test, Python 3.10–3.13). The core idea is *planted ground truth*: synthetic
corpora whose separability is known exactly, so the tool's answers can be checked against
mathematics rather than against itself — a planted marker caps any classifier at AUC
`0.5 + rate / 2`; a length shift has an exact AUC computed from the length distribution. Reproduce
with `scripts/verify.sh`.

## Development

```bash
python -m pip install -e ".[dev]"
scripts/verify.sh "eval-distinguish selftest"
```

## License

MIT — see [LICENSE](LICENSE).
