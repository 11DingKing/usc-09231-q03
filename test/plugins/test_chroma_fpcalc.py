"""Resource-boundary tests for the external fpcalc calculator.

The chroma plugin starts the ``fpcalc`` fingerprint calculator as a
short-lived subprocess once per track. During a long multi-disc analysis
(thousands of tracks) any process or pipe handle that is not released on
a single invocation accumulates until the process exhausts its file
descriptor table and later, unrelated tasks start failing randomly.

These tests use tiny shell scripts as fake calculators -- including ones
that return empty output, exit non-zero, print garbage, keep the stdout
pipe open from a descendant, or are interrupted while their output is
read -- and assert that:

* every invocation releases every process and pipe descriptor,
* every failure mode surfaces as ``FingerprintGenerationError`` (or
  ``NoBackendError`` when the calculator is missing), and
* descriptors do not accumulate across a multi-track loop driving the
  actual ``acoustid_match`` / ``fingerprint_item`` batch entry points.
"""

from __future__ import annotations

import gc
import os
from unittest.mock import MagicMock

import acoustid
import pytest

from beets import util
from beets.library import Item

chroma = pytest.importorskip("beetsplug.chroma", exc_type=ImportError)


def open_fds() -> set[int]:
    """Return the set of file descriptors currently open in this process."""
    return {int(fd) for fd in os.listdir(f"/proc/{os.getpid()}/fd")}


def assert_no_fd_accumulation(fn, iterations: int = 25) -> None:
    """Run ``fn`` repeatedly; no descriptor may linger between runs.

    The check is taken at *every* iteration boundary (not just before and
    after) so a descriptor released on a later iteration cannot hide one
    that an earlier iteration leaked.
    """
    gc.collect()
    baseline = open_fds()
    leaked: set[int] = set()
    for _ in range(iterations):
        fn()
        gc.collect()
        leaked |= open_fds() - baseline
    assert not leaked, f"lingering descriptors across iterations: {sorted(leaked)}"


@pytest.fixture
def calculator(tmp_path, monkeypatch):
    """Factory installing an executable fake fpcalc via ``$FPCALC``."""

    def install(body: str, name: str = "fake-fpcalc") -> str:
        path = tmp_path / name
        path.write_text("#!/bin/sh\n" + body)
        os.chmod(path, 0o755)
        monkeypatch.setenv(chroma.FPCALC_ENVVAR, str(path))
        return str(path)

    return install


@pytest.fixture
def fast_timeout(monkeypatch):
    """Keep the hung-calculator test fast without affecting other tests."""
    monkeypatch.setattr(chroma, "FPCALC_TIMEOUT", 1.0)


# --- Single-invocation behaviour ----------------------------------------


def test_run_fpcalc_parses_output(calculator):
    calculator('echo "DURATION=1.5"\necho "FINGERPRINT=ABC123"\n')
    duration, fp = chroma._run_fpcalc("/some/track.flac")
    assert duration == 1.5
    assert fp == b"ABC123"


def test_run_fpcalc_tolerates_stray_lines(calculator):
    calculator(
        'echo "some unrelated warning"\n'
        'echo "DURATION=2"\n'
        'echo "LINE_WITHOUT_EQUALS"\n'
        'echo "FINGERPRINT=XYZ"\n'
    )
    duration, fp = chroma._run_fpcalc("/track")
    assert (duration, fp) == (2.0, b"XYZ")


def test_run_fpcalc_missing_calculator_raises(monkeypatch):
    monkeypatch.setenv(chroma.FPCALC_ENVVAR, "/does/not/exist/fpcalc")
    with pytest.raises(acoustid.NoBackendError):
        chroma._run_fpcalc("/track")


def test_run_fpcalc_empty_output_raises(calculator):
    calculator("exit 0\n")
    with pytest.raises(acoustid.FingerprintGenerationError, match="missing"):
        chroma._run_fpcalc("/track")


def test_run_fpcalc_only_duration_raises(calculator):
    calculator('echo "DURATION=3"\n')
    with pytest.raises(acoustid.FingerprintGenerationError):
        chroma._run_fpcalc("/track")


def test_run_fpcalc_nonzero_exit_raises(calculator):
    calculator('echo "boom" >&2\nexit 7\n')
    with pytest.raises(acoustid.FingerprintGenerationError, match="status 7"):
        chroma._run_fpcalc("/track")


def test_run_fpcalc_bad_duration_raises(calculator):
    calculator('echo "DURATION=notanumber"\necho "FINGERPRINT=X"\n')
    with pytest.raises(acoustid.FingerprintGenerationError):
        chroma._run_fpcalc("/track")


# --- Resource lifecycle ---------------------------------------------------


def test_repeated_success_does_not_leak(calculator):
    calculator('echo "DURATION=1"\necho "FINGERPRINT=FP"\n')
    assert_no_fd_accumulation(lambda: chroma._run_fpcalc("/track"))


def test_repeated_empty_output_does_not_leak(calculator):
    calculator("exit 0\n")

    def run() -> None:
        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._run_fpcalc("/track")

    assert_no_fd_accumulation(run)


def test_repeated_nonzero_exit_does_not_leak(calculator):
    calculator("exit 3\n")

    def run() -> None:
        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._run_fpcalc("/track")

    assert_no_fd_accumulation(run)


def test_repeated_missing_calculator_does_not_leak(monkeypatch):
    monkeypatch.setenv(chroma.FPCALC_ENVVAR, "/does/not/exist/fpcalc")

    def run() -> None:
        with pytest.raises(acoustid.NoBackendError):
            chroma._run_fpcalc("/track")

    assert_no_fd_accumulation(run)


def test_hung_descendant_holding_pipe_is_reaped(calculator, fast_timeout):
    # The calculator prints valid-looking output but spawns a child that
    # inherits the stdout pipe and sleeps. The pipe never reaches EOF while
    # that child lives, so a plain communicate() would block forever and
    # leak the pipe; the timeout must kill the whole process group.
    calculator(
        'echo "DURATION=1"\n'
        'echo "FINGERPRINT=X"\n'
        "(sleep 30) &\n"
        "wait\n"
    )
    with pytest.raises(acoustid.FingerprintGenerationError, match="timed out"):
        chroma._run_fpcalc("/track")

    # Everything (calculator + its sleeping child) is gone.
    assert os.system("pgrep -f 'sleep 30' >/dev/null 2>&1") != 0


def test_hung_descendant_repeated_does_not_leak(calculator, fast_timeout):
    calculator(
        'echo "DURATION=1"\n'
        'echo "FINGERPRINT=X"\n'
        "(sleep 30) &\n"
        "wait\n"
    )

    def run() -> None:
        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._run_fpcalc("/track")

    assert_no_fd_accumulation(run, iterations=5)
    assert os.system("pgrep -f 'sleep 30' >/dev/null 2>&1") != 0


def test_keyboard_interrupt_during_read_does_not_leak(calculator, monkeypatch):
    import subprocess

    # Simulate an exception arriving while the calculator's output is
    # being read: the running process must still be killed and reaped.
    calculator("sleep 30\n")

    def interrupted_communicate(self, *args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupted_communicate)
    with pytest.raises(KeyboardInterrupt):
        chroma._run_fpcalc("/track")
    gc.collect()
    # No sleep 30 process outlives the interrupted read.
    assert os.system("pgrep -f 'sleep 30' >/dev/null 2>&1") != 0


# --- Multi-track batch entry points --------------------------------------


def test_multi_track_acoustid_match_does_not_leak(calculator, monkeypatch):
    calculator('echo "DURATION=42"\necho "FINGERPRINT=BATCHFP"\n')
    # Avoid network access: the lookup simply finds nothing.
    monkeypatch.setattr(
        acoustid, "lookup", lambda *a, **k: {"status": "ok", "results": []}
    )
    log = MagicMock()

    def run() -> None:
        path = f"/album/track/{id(object())}.flac".encode()
        chroma.acoustid_match(log, path)

    assert_no_fd_accumulation(run, iterations=20)


def test_multi_track_fingerprint_item_does_not_leak(calculator):
    calculator('echo "DURATION=12"\necho "FINGERPRINT=ITEMFP"\n')
    import logging

    log = logging.getLogger("test.chroma.fpcalc")

    def run() -> None:
        item = Item(path=f"/lib/track/{os.getpid()}-{id(object())}.mp3".encode())
        item.length = 10
        fp = chroma.fingerprint_item(log, item)
        assert fp == "ITEMFP"

    assert_no_fd_accumulation(run, iterations=20)


def test_multi_track_mixed_success_and_failure_does_not_leak(
    calculator, monkeypatch
):
    # Alternate a working and a failing calculator across the batch by
    # rewriting the script on disk between tracks.
    good = "#!/bin/sh\necho DURATION=1\necho FINGERPRINT=OK\n"
    bad = "#!/bin/sh\nexit 5\n"
    path = calculator(good)

    def run() -> None:
        with open(path, "w") as fh:
            fh.write(bad if run.toggle else good)
        os.chmod(path, 0o755)
        run.toggle = not run.toggle
        try:
            duration, fp = chroma._run_fpcalc("/track")
        except acoustid.FingerprintGenerationError:
            return
        assert (duration, fp) == (1.0, b"OK")

    run.toggle = True
    assert_no_fd_accumulation(run, iterations=20)


# --- Backend selection ----------------------------------------------------


def test_fingerprint_file_uses_fpcalc_without_chromaprint(
    calculator, monkeypatch
):
    calculator('echo "DURATION=9"\necho "FINGERPRINT=SEL"\n')
    monkeypatch.setattr(acoustid, "have_audioread", False)
    monkeypatch.setattr(acoustid, "have_chromaprint", False)
    assert chroma._fingerprint_file(util.syspath(b"/t")) == (9.0, b"SEL")


def test_fingerprint_file_prefers_in_process_backend(monkeypatch):
    # When the Chromaprint library is importable, no subprocess (and so no
    # external-calculator resource boundary) is involved at all.
    monkeypatch.setattr(acoustid, "have_audioread", True)
    monkeypatch.setattr(acoustid, "have_chromaprint", True)
    sentinel = (3.0, b"INPROC")
    monkeypatch.setattr(acoustid, "fingerprint_file", lambda p: sentinel)
    assert chroma._fingerprint_file("/t") == sentinel
