# Assurance case

The central device is **planted ground truth** (`src/eval_distinguishability/synth.py`): synthetic
corpora that are identical in distribution except for a difference whose separability is known
in closed form, so tests compare the tool with mathematics instead of with itself.

| # | Claim | Evidence |
|---|---|---|
| 1 | `auc` equals scikit-learn's `roc_auc_score` (including heavy ties), is symmetric under score negation and label inversion, and invariant to monotone rescaling. | `tests/test_stats.py` (Hypothesis, independent). |
| 2 | The bootstrap interval covers the true AUC of a fixed scoring rule at about the nominal rate (150 simulated experiments with a known Gaussian AUC), is deterministic for a seed, narrows with data and with level. | `tests/test_stats.py` (`slow`). |
| 3 | The permutation p-value has the stated formula and extremes, keeps class counts, and is valid under the null (300 simulated null experiments: rejection rate at 5% within [1%, 10%], mean p near 0.5). | `tests/test_stats.py`. |
| 4 | **Planted markers are recovered near the best achievable AUC.** For marker rates 0.3, 0.6, 0.9 the mean AUC over three seeds is within 0.06 of the Bayes-optimal `0.5 + rate/2`, and no run exceeds it by more than 0.07. | `tests/test_audit.py` (derivation in `synth.py`). |
| 5 | A strong planted signal is declared distinguishable, with the permutation p-value at its exact resolution `1/(B+1)`. | `tests/test_audit.py`. |
| 6 | **Calibration under the null.** Over 24 null corpora the false-positive rate at alpha 0.1 is at most 0.30 and the mean AUC is within 0.035 of 0.5; a separate 60-corpus simulation shows no systematic AUC bias (means 0.505 and 0.493). | `test_null_corpora_rarely_look_distinguishable` (`slow`); `scripts/simulate_ci_width.py`. |
| 7 | **Length differences** are caught by the surface family, whose AUC matches the *exact* AUC of document length computed from the length distribution (within 0.06; the exact function itself is checked against empirical lengths for four shifts). | `tests/test_audit.py`, `tests/test_synth.py`. |
| 8 | **Duplicate leakage** is prevented by grouping: triplicated prompts without groups inflate the AUC above 0.8; with groups it falls below 0.65. | `test_grouped_cross_validation_prevents_duplicate_leakage`. |
| 9 | Exact duplicates are removed and counted, shared prompts are warned about, small/imbalanced samples are flagged, and a permutation count that cannot reach alpha after correction is flagged. | `tests/test_audit.py`. |
| 10 | The telltale token is the top feature and points at evaluation prompts; rare tokens are withheld by `min_df`; feature reporting can be switched off; dense features carry no document counts. | `tests/test_audit.py`. |
| 11 | The ablation separates concentrated from diffuse signals, leaves chance (not "anti-signal") once the telltale n-grams are gone, and skips counts larger than the feature set. Two design choices exist because tests exposed the failure otherwise: ranking features on *training folds only* (ranking on all data pushed the leftover AUC far below 0.5) and re-normalising rows after dropping columns (the vector norm leaked the dropped weight). | `tests/test_audit.py`. |
| 12 | Runs are deterministic for a seed and differ across seeds; reports round-trip through JSON; every input error is a clear message. | `tests/test_audit.py`, `tests/test_cli.py`, `tests/test_corpus_features.py`. |
| 13 | The CLI works end to end, gates on `--fail-if-distinguishable`, and the package installs from a wheel into a clean virtualenv and runs. | `tests/test_cli.py`; `scripts/verify.sh`. |

## What this does *not* establish

- That a corpus the tool finds indistinguishable is indistinguishable to a language model.
- That the interval is a nominal-coverage interval for the pipeline (it is not; see the README).
- Anything about real traffic: all verification uses synthetic text.

## Reproduce

```bash
python -m pip install -e ".[dev]"
scripts/verify.sh "eval-distinguish selftest"
python scripts/simulate_ci_width.py
```
