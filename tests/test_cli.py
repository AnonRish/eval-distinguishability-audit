from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from eval_distinguishability import __version__
from eval_distinguishability import cli as cli_module
from eval_distinguishability.cli import main, run_selftest


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def synth_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    code, out, _ = run(
        capsys, "synth", "--out-dir", str(tmp_path), "--n", "200", "--marker-rate", "0.9"
    )
    assert code == 0 and "best possible AUC from markers alone: 0.950" in out
    return tmp_path


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and __version__ in capsys.readouterr().out


def test_module_entry_point() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "eval_distinguishability.cli", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in result.stdout


def test_synth_writes_jsonl_corpora(synth_dir: Path) -> None:
    lines = (synth_dir / "eval.jsonl").read_text().splitlines()
    assert len(lines) == 200 and "text" in json.loads(lines[0])
    assert len((synth_dir / "real.jsonl").read_text().splitlines()) == 200
    marked = sum("zq0" in json.loads(line)["text"].split() for line in lines)
    assert marked > 150  # about 90%


def test_audit_text_json_and_exit_codes(
    synth_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = [
        "audit",
        "--eval",
        str(synth_dir / "eval.jsonl"),
        "--real",
        str(synth_dir / "real.jsonl"),
        "--families",
        "word",
        "--permutations",
        "30",
        "--bootstrap",
        "50",
    ]
    code, out, _ = run(capsys, *base)
    assert code == 0
    assert "Verdict: DISTINGUISHABLE" in out and "zq0" in out and "Signal concentration" in out
    report_path = tmp_path / "report.json"
    code, out, _ = run(capsys, *base, "--format", "json", "--out", str(report_path))
    payload = json.loads(out)
    assert code == 0 and payload["verdict"] == "distinguishable"
    assert json.loads(report_path.read_text()) == payload
    code, _, _ = run(capsys, *base, "--fail-if-distinguishable")
    assert code == 1
    code, out, _ = run(
        capsys, *base, "--no-features", "--keep-duplicates", "--min-df", "3", "--top-k", "2"
    )
    assert code == 0 and "Most discriminative" not in out


def test_audit_on_indistinguishable_corpora_passes_the_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run(
        capsys,
        "synth",
        "--out-dir",
        str(tmp_path),
        "--n",
        "100",
        "--marker-rate",
        "0.0",
        "--seed",
        "5",
    )
    code, out, _ = run(
        capsys,
        "audit",
        "--eval",
        str(tmp_path / "eval.jsonl"),
        "--real",
        str(tmp_path / "real.jsonl"),
        "--families",
        "word,surface",
        "--permutations",
        "20",
        "--bootstrap",
        "50",
        "--fail-if-distinguishable",
    )
    assert code == 0 and "NO EVIDENCE OF DISTINGUISHABILITY" in out


def test_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(
        capsys, "audit", "--eval", str(tmp_path / "no.txt"), "--real", str(tmp_path / "no2.txt")
    )
    assert code == 2 and "cannot read" in err
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("only one prompt\n")
    code, _, err = run(capsys, "audit", "--eval", str(tiny), "--real", str(tiny))
    assert code == 2 and "at least" in err
    with pytest.raises(SystemExit) as exc:
        main(["audit", "--eval", "a", "--real", "b", "--families", "word,bogus"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        main(["audit", "--eval", "a", "--real", "b", "--families", ","])
    capsys.readouterr()
    code, _, err = run(capsys, "synth", "--out-dir", str(tmp_path / "s"), "--marker-rate", "2")
    assert code == 2 and "marker_rate" in err


def test_selftest(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    code, out, _ = run(capsys, "selftest")
    assert code == 0 and out.count("PASS") == 3 and "FAIL" not in out
    assert all(ok for _, ok, _ in run_selftest())
    monkeypatch.setattr(cli_module, "expected_marker_auc", lambda rate: 0.0)
    code, out, _ = run(capsys, "selftest")
    assert code == 1 and "FAIL  planted marker rate" in out
