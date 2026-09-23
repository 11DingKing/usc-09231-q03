import itertools
import os
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import acoustid
import pytest

from beets import metadata_plugins
from beets import logging as beets_logging
from beets.autotag import AlbumInfo, TrackInfo
from beets.library import Item
from beets.test.helper import ImportHelper, IOMixin, PluginMixin

chroma = pytest.importorskip("beetsplug.chroma", exc_type=ImportError)

TEST_TITLE_1 = "TEST_TITLE_1"
TEST_TITLE_2 = "TEST_TITLE_2"
FINGERPRINT_1 = "FP_1"
FINGERPRINT_1_CLOSE = "FP_1_CLOSE"
FINGERPRINT_2 = "FP_2"


@patch("acoustid.compare_fingerprints")
class TestChroma(IOMixin, PluginMixin, ImportHelper):
    plugin = "chroma"

    def setup_lib(self):
        item1 = Item(path="/file")
        item1.length = 30
        item1.title = TEST_TITLE_1
        item1.acoustid_fingerprint = FINGERPRINT_1
        item1.add(self.lib)

        item2 = Item(path="/file")
        item2.length = 30
        item2.title = TEST_TITLE_2
        item2.acoustid_fingerprint = FINGERPRINT_2
        item2.add(self.lib)

    def run_search(self, fp):
        return self.run_with_output("chromasearch", "-s", fp, "-f", "$title")

    def line_count(self, str_):
        return len(
            [line for line in str_.split("\n") if line.strip(" \n") != ""]
        )

    def compare_fingerprints(self, *args, **kwargs):
        if args[0][1] == args[1][1]:
            return 1

        if args[0][1] == FINGERPRINT_1_CLOSE and args[1][1] == FINGERPRINT_1:
            return 0.9

        return 0.1

    def test_chroma_search_exact(self, compare_fingerprints):
        self.setup_lib()
        compare_fingerprints.side_effect = self.compare_fingerprints

        output = self.run_search(FINGERPRINT_2)
        assert self.line_count(output) == 1
        assert TEST_TITLE_2 in output

        output = self.run_search(FINGERPRINT_1)
        assert self.line_count(output) == 1
        assert TEST_TITLE_1 in output

    def test_chroma_search_close(self, compare_fingerprints):
        self.setup_lib()
        compare_fingerprints.side_effect = self.compare_fingerprints

        output = self.run_search(FINGERPRINT_1_CLOSE)
        assert self.line_count(output) == 2
        assert TEST_TITLE_1 in output.split("\n")[0]


def _seed_acoustid_match(item_path: bytes = b"/fake/path.mp3") -> Item:
    """Seed the chroma module-level match cache as if acoustid had run."""
    chroma._matches[item_path] = (
        ["rec-id-1"],
        ["rel-id-1", "rel-id-1", "rel-id-1"],
    )
    return Item(path=item_path)


class TestChromaCandidates(PluginMixin):
    """Regression tests for issue #6212: chroma must respect which metadata
    source plugins are enabled.

    When the musicbrainz plugin is not loaded, chroma must not produce any
    MusicBrainz-sourced candidates (via either ``candidates`` or
    ``item_candidates``). When it IS loaded, chroma resolves acoustid
    matches through the registered plugin instance.

    ``plugin`` is intentionally not set on the class so that
    :py:meth:`PluginMixin.load_plugins` honours explicit plugin-name
    arguments and each test can choose its own combination. The autouse
    fixture clears the ``@cache``-decorated metadata-source registry and
    the chroma match state between tests.
    """

    preload_plugin = False

    @pytest.fixture(autouse=True)
    def _setup_chroma(self):
        metadata_plugins.find_metadata_source_plugins.cache_clear()
        metadata_plugins.get_metadata_source.cache_clear()
        chroma._matches.clear()
        yield
        chroma._matches.clear()
        self.unload_plugins()
        metadata_plugins.find_metadata_source_plugins.cache_clear()
        metadata_plugins.get_metadata_source.cache_clear()

    def test_candidates_returns_empty_without_musicbrainz(self):
        self.load_plugins("chroma")
        plugin = chroma.AcoustidPlugin()
        item = _seed_acoustid_match()

        result = plugin.candidates(
            [item], artist="A", album="B", va_likely=False
        )

        assert list(result) == []

    def test_item_candidates_returns_empty_without_musicbrainz(self):
        self.load_plugins("chroma")
        plugin = chroma.AcoustidPlugin()
        item = _seed_acoustid_match()

        result = plugin.item_candidates(item, artist="A", title="B")

        assert list(result) == []

    def test_candidates_returns_mb_albums_with_musicbrainz(self, monkeypatch):
        self.load_plugins("chroma", "musicbrainz")

        fake_album = AlbumInfo(
            tracks=[], album_id="rel-id-1", album="Fake Album"
        )
        mb_plugin = metadata_plugins.get_metadata_source("musicbrainz")
        assert mb_plugin is not None
        monkeypatch.setattr(
            mb_plugin, "album_for_id", MagicMock(return_value=fake_album)
        )

        plugin = chroma.AcoustidPlugin()
        item = _seed_acoustid_match()

        result = list(
            plugin.candidates([item], artist="A", album="B", va_likely=False)
        )

        assert result == [fake_album]
        mb_plugin.album_for_id.assert_called_with("rel-id-1")

    def test_item_candidates_returns_mb_tracks_with_musicbrainz(
        self, monkeypatch
    ):
        self.load_plugins("chroma", "musicbrainz")

        fake_track = TrackInfo(title="Fake Track", track_id="rec-id-1")
        mb_plugin = metadata_plugins.get_metadata_source("musicbrainz")
        assert mb_plugin is not None
        monkeypatch.setattr(
            mb_plugin, "track_for_id", MagicMock(return_value=fake_track)
        )

        plugin = chroma.AcoustidPlugin()
        item = _seed_acoustid_match()

        result = list(plugin.item_candidates(item, artist="A", title="B"))

        assert result == [fake_track]
        mb_plugin.track_for_id.assert_called_with("rec-id-1")


# Resource-boundary tests for the external fingerprint calculator.
#
# A long-running Chroma analysis spawns one ``fpcalc`` process per
# track. These tests exercise that boundary in multi-track loops while
# injecting calculator failures -- empty output, non-zero exits, a
# missing binary, hung processes, and mid-spawn errors -- and assert
# that file descriptors never accumulate.

FPCALC_OK_BODY = "echo DURATION=30.5; echo FINGERPRINT=AQADtEmSHJmWJI8"
FPCALC_FINGERPRINT = b"AQADtEmSHJmWJI8"


def fd_count() -> int:
    """Return the number of open file descriptors of this process."""
    return len(os.listdir("/proc/self/fd"))


requires_proc_fd = pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"),
    reason="needs /proc/self/fd to count file descriptors",
)


@pytest.fixture
def make_fpcalc(tmp_path, monkeypatch):
    """Return a factory creating fake ``fpcalc`` executables.

    Each script is written to a temporary directory, made executable,
    and selected as the fingerprint calculator through the ``FPCALC``
    environment variable that ``beetsplug.chroma`` honors.
    """
    counter = itertools.count()

    def make(body: str) -> str:
        script = tmp_path / f"fpcalc-{next(counter)}"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        monkeypatch.setenv(chroma.FPCALC_ENVVAR, str(script))
        return str(script)

    return make


@pytest.fixture
def log():
    return beets_logging.getLogger("beets.test")


class TestFingerprintFile:
    """Behavior of the calculator boundary for a single computation."""

    @pytest.fixture(autouse=True)
    def _clear_chroma_state(self):
        chroma._matches.clear()
        chroma._fingerprints.clear()
        chroma._acoustids.clear()
        yield
        chroma._matches.clear()
        chroma._fingerprints.clear()
        chroma._acoustids.clear()

    def test_success_returns_duration_and_fingerprint(self, make_fpcalc):
        make_fpcalc(FPCALC_OK_BODY)

        duration, fp = chroma._fingerprint_file("/fake/audio.mp3")

        assert duration == 30.5
        assert fp == FPCALC_FINGERPRINT

    @pytest.mark.parametrize(
        "body, match",
        [
            # The calculator returns empty output.
            ("exit 0", "missing fpcalc output"),
            # Stray output that is not KEY=VALUE is ignored, but then the
            # required fields are missing.
            ("echo stray-line-without-separator", "missing fpcalc output"),
            ("echo DURATION=30.5", "missing fpcalc output"),
            ("echo FINGERPRINT=AQADtEmSHJmWJI8", "missing fpcalc output"),
            ("echo DURATION=30.5; echo FINGERPRINT=", "missing fpcalc output"),
            ("echo DURATION=oops; echo FINGERPRINT=AQAD", "not numeric"),
        ],
    )
    def test_invalid_output_rejected(self, make_fpcalc, body, match):
        make_fpcalc(body)

        with pytest.raises(acoustid.FingerprintGenerationError, match=match):
            chroma._fingerprint_file("/fake/audio.mp3")

    def test_nonzero_exit_rejected(self, make_fpcalc):
        make_fpcalc("echo some fatal error >&2; exit 3")

        with pytest.raises(
            acoustid.FingerprintGenerationError, match="status 3"
        ):
            chroma._fingerprint_file("/fake/audio.mp3")

    def test_missing_calculator_rejected(self, monkeypatch):
        monkeypatch.setenv(chroma.FPCALC_ENVVAR, "/nonexistent/fpcalc-bin")

        with pytest.raises(
            acoustid.FingerprintGenerationError, match="execution failed"
        ):
            chroma._fingerprint_file("/fake/audio.mp3")

    def test_hung_calculator_is_killed(self, make_fpcalc):
        make_fpcalc("exec sleep 12")

        start = time.monotonic()
        with pytest.raises(
            acoustid.FingerprintGenerationError, match="timed out"
        ):
            chroma._fingerprint_file("/fake/audio.mp3", timeout=0.2)

        # The computation must give up promptly instead of waiting for
        # the sleeper to finish on its own.
        assert time.monotonic() - start < 10

    def test_large_stderr_does_not_block(self, make_fpcalc):
        # stderr is discarded, not piped: a chatty calculator cannot fill
        # a pipe buffer and deadlock the analysis.
        make_fpcalc("head -c 200000 /dev/zero >&2; exit 1")

        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._fingerprint_file("/fake/audio.mp3", timeout=10)

    def test_acoustid_match_populates_caches(
        self, make_fpcalc, monkeypatch, log
    ):
        make_fpcalc(FPCALC_OK_BODY)
        monkeypatch.setattr(
            chroma.acoustid,
            "lookup",
            lambda *args, **kwargs: {
                "status": "ok",
                "results": [
                    {
                        "score": 0.95,
                        "id": "acoustid-1",
                        "recordings": [
                            {
                                "id": "rec-1",
                                "releases": [{"id": "rel-1"}, {"id": "rel-2"}],
                            }
                        ],
                    }
                ],
            },
        )

        chroma.acoustid_match(log, b"/fake/track.mp3")

        assert (
            chroma._fingerprints[b"/fake/track.mp3"]
            == FPCALC_FINGERPRINT.decode()
        )
        assert chroma._acoustids[b"/fake/track.mp3"] == "acoustid-1"
        assert chroma._matches[b"/fake/track.mp3"] == (
            ["rec-1"],
            ["rel-1", "rel-2"],
        )

    def test_acoustid_match_survives_calculator_failure(
        self, make_fpcalc, log
    ):
        make_fpcalc("exit 1")

        chroma.acoustid_match(log, b"/fake/track.mp3")

        assert b"/fake/track.mp3" not in chroma._fingerprints
        assert b"/fake/track.mp3" not in chroma._matches


@requires_proc_fd
class TestFingerprintFileDescriptors:
    """Multi-track loops must not accumulate file descriptors."""

    ITERATIONS = 25

    @pytest.fixture(autouse=True)
    def _clear_chroma_state(self):
        chroma._matches.clear()
        chroma._fingerprints.clear()
        chroma._acoustids.clear()
        yield
        chroma._matches.clear()
        chroma._fingerprints.clear()
        chroma._acoustids.clear()

    def test_repeated_success_releases_fds(self, make_fpcalc):
        make_fpcalc(FPCALC_OK_BODY)

        # Warm up one-shot costs, then take the baseline.
        chroma._fingerprint_file("/fake/warmup.mp3")
        baseline = fd_count()

        for i in range(self.ITERATIONS):
            duration, fp = chroma._fingerprint_file(f"/fake/track-{i}.mp3")
            assert duration == 30.5
            assert fp == FPCALC_FINGERPRINT

        assert fd_count() == baseline

    def test_repeated_failures_release_fds(self, make_fpcalc, monkeypatch):
        failing_scripts = [
            make_fpcalc("exit 0"),  # empty output
            make_fpcalc("echo error >&2; exit 1"),  # non-zero exit
            make_fpcalc("echo garbage-without-separator"),  # malformed
        ]
        missing = "/nonexistent/fpcalc-bin"
        calculators = [*failing_scripts, missing]

        # Exercise each failure mode once before taking the baseline.
        for calculator in calculators:
            monkeypatch.setenv(chroma.FPCALC_ENVVAR, calculator)
            with pytest.raises(acoustid.FingerprintGenerationError):
                chroma._fingerprint_file("/fake/warmup.mp3")
        baseline = fd_count()

        for i in range(self.ITERATIONS):
            monkeypatch.setenv(
                chroma.FPCALC_ENVVAR, calculators[i % len(calculators)]
            )
            with pytest.raises(acoustid.FingerprintGenerationError):
                chroma._fingerprint_file(f"/fake/track-{i}.mp3")

        assert fd_count() == baseline

    def test_spawn_failure_releases_fds(self, monkeypatch):
        """An error while starting the process must not leak handles."""

        def exploding_popen(*args, **kwargs):
            raise OSError("injected spawn failure")

        monkeypatch.setattr(subprocess, "Popen", exploding_popen)

        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._fingerprint_file("/fake/warmup.mp3")
        baseline = fd_count()

        for _ in range(self.ITERATIONS):
            with pytest.raises(acoustid.FingerprintGenerationError):
                chroma._fingerprint_file("/fake/track.mp3")

        assert fd_count() == baseline

    def test_pipe_read_failure_releases_fds(self, make_fpcalc, monkeypatch):
        """If draining the output pipe fails, the pipe is still closed."""

        class BrokenCommunicate(subprocess.Popen):
            def communicate(self, *args, **kwargs):
                raise RuntimeError("injected pipe failure")

        make_fpcalc(FPCALC_OK_BODY)
        monkeypatch.setattr(subprocess, "Popen", BrokenCommunicate)

        with pytest.raises(RuntimeError, match="injected pipe failure"):
            chroma._fingerprint_file("/fake/warmup.mp3")
        baseline = fd_count()

        for _ in range(self.ITERATIONS):
            with pytest.raises(RuntimeError):
                chroma._fingerprint_file("/fake/track.mp3")

        assert fd_count() == baseline

    def test_hung_calculators_release_fds(self, make_fpcalc):
        make_fpcalc("exec sleep 12")

        with pytest.raises(acoustid.FingerprintGenerationError):
            chroma._fingerprint_file("/fake/warmup.mp3", timeout=0.2)
        baseline = fd_count()

        for _ in range(3):
            with pytest.raises(acoustid.FingerprintGenerationError):
                chroma._fingerprint_file("/fake/track.mp3", timeout=0.2)

        assert fd_count() == baseline

    def test_fingerprint_task_releases_fds(
        self, make_fpcalc, monkeypatch, log
    ):
        """The import-task batch loop releases handles for every track."""
        make_fpcalc(FPCALC_OK_BODY)
        monkeypatch.setattr(
            chroma.acoustid,
            "lookup",
            lambda *args, **kwargs: {"status": "ok", "results": []},
        )

        def run_batch(round_name):
            items = [
                Item(path=f"/fake/{round_name}-{i}.mp3".encode())
                for i in range(self.ITERATIONS)
            ]
            task = SimpleNamespace(items=items)
            chroma.fingerprint_task(log, task, session=None)

        run_batch("warmup")
        assert len(chroma._fingerprints) == self.ITERATIONS
        baseline = fd_count()

        chroma._fingerprints.clear()
        run_batch("batch")
        assert len(chroma._fingerprints) == self.ITERATIONS

        assert fd_count() == baseline

    def test_fingerprint_task_tolerates_mixed_failures(
        self, make_fpcalc, monkeypatch, log
    ):
        """Failures injected mid-batch neither abort the batch nor leak."""
        make_fpcalc(
            'case "$3" in\n'
            "  *fail*) exit 1 ;;\n"
            f"  *) {FPCALC_OK_BODY} ;;\n"
            "esac"
        )
        monkeypatch.setattr(
            chroma.acoustid,
            "lookup",
            lambda *args, **kwargs: {"status": "ok", "results": []},
        )
        items = [
            Item(
                path=f"/fake/{'fail' if i % 2 else 'ok'}-{i}.mp3".encode()
            )
            for i in range(self.ITERATIONS)
        ]
        task = SimpleNamespace(items=items)

        chroma.fingerprint_task(log, task, session=None)
        baseline = fd_count()

        chroma._fingerprints.clear()
        chroma.fingerprint_task(log, task, session=None)

        # Every odd (failing) track was skipped; every even track has a
        # fingerprint. The batch processed all tracks either way.
        assert len(chroma._fingerprints) == (self.ITERATIONS + 1) // 2
        assert fd_count() == baseline

    def test_fingerprint_item_releases_fds(self, make_fpcalc, log):
        make_fpcalc(FPCALC_OK_BODY)

        def fingerprint_one(i):
            item = Item(path=f"/fake/item-{i}.mp3".encode())
            item.length = 30
            return chroma.fingerprint_item(log, item, quiet=True)

        assert fingerprint_one(0) == FPCALC_FINGERPRINT.decode()
        baseline = fd_count()

        for i in range(1, self.ITERATIONS + 1):
            assert fingerprint_one(i) == FPCALC_FINGERPRINT.decode()

        assert fd_count() == baseline

    def test_fingerprint_item_failure_returns_none_and_releases_fds(
        self, make_fpcalc, log
    ):
        make_fpcalc("exit 1")

        def fingerprint_one(i):
            item = Item(path=f"/fake/item-{i}.mp3".encode())
            item.length = 30
            return chroma.fingerprint_item(log, item, quiet=True)

        assert fingerprint_one(0) is None
        baseline = fd_count()

        for i in range(1, self.ITERATIONS + 1):
            assert fingerprint_one(i) is None

        assert fd_count() == baseline
