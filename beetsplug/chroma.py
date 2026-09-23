"""Adds Chromaprint/Acoustid acoustic fingerprinting support to the
autotagger. Requires the pyacoustid library.
"""

from __future__ import annotations

import errno
import heapq
import os
import re
import signal
import subprocess
from collections import defaultdict
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Protocol

import acoustid
import confuse

from beets import config, ui, util
from beets.autotag import Distance
from beets.exceptions import UserError
from beets.metadata_plugins import MetadataSourcePlugin, get_metadata_source
from beets.util.color import colorize

if TYPE_CHECKING:
    import optparse
    from collections.abc import Iterable, Iterator, Sequence

    from beets.autotag import AlbumInfo, TrackInfo
    from beets.importer import ImportSession, ImportTask
    from beets.library import Item, Library
    from beets.logging import BeetsLogger as Logger
    from beetsplug.musicbrainz import MusicBrainzPlugin

    from ._typing import JSONDict


class ChromaSearchCLIOpts(Protocol):
    count: int
    full: bool | None
    search: str | None
    write: bool | None


API_KEY = "1vOwZtEn"
SCORE_THRESH = 0.5
TRACK_ID_WEIGHT = 10.0
COMMON_REL_THRESH = 0.6  # How many tracks must have an album in common?
MAX_RECORDINGS = 5
MAX_RELEASES = 5

# External fingerprint calculator ("fpcalc") configuration. The
# calculator is a short-lived subprocess: one invocation per track, so
# every pipe and process handle it creates must be released again even
# when the calculator fails -- long multi-disc imports otherwise leak
# file descriptors until unrelated later tasks start failing.
FPCALC_COMMAND = "fpcalc"
FPCALC_ENVVAR = "FPCALC"
MAX_AUDIO_LENGTH = acoustid.MAX_AUDIO_LENGTH
# Bound the time spent waiting on the calculator. If the calculator
# exits but a descendant keeps the stdout pipe open, communicate() would
# block forever and leak the pipe; the timeout lets us kill the whole
# process group and release everything.
FPCALC_TIMEOUT: float = 30.0

# Stores the Acoustid match information for each track. This is
# populated when an import task begins and then used when searching for
# candidates. It maps audio file paths to (recording_ids, release_ids)
# pairs. If a given path is not present in the mapping, then no match
# was found.
_matches: dict[bytes, tuple[list[str], list[str]]] = {}

# Stores the fingerprint and Acoustid ID for each track. This is stored
# as metadata for each track for later use but is not relevant for
# autotagging.
_fingerprints: dict[bytes, str] = {}
_acoustids: dict[bytes, str] = {}


def prefix(it: Iterable[Any], count: int) -> Iterator[Any]:
    """Truncate an iterable to at most `count` items."""
    for i, v in enumerate(it):
        if i >= count:
            break
        yield v


def releases_key(
    release: JSONDict, countries: Sequence[re.Pattern[str]], original_year: bool
) -> tuple[int, int, int, int]:
    """Used as a key to sort releases by date then preferred country"""
    date = release.get("date")
    if date and original_year:
        year = date.get("year", 9999)
        month = date.get("month", 99)
        day = date.get("day", 99)
    else:
        year = 9999
        month = 99
        day = 99

    # Uses index of preferred countries to sort
    country_key = 99
    if release.get("country"):
        for i, country in enumerate(countries):
            if country.match(release["country"]):
                country_key = i
                break

    return (year, month, day, country_key)


def _run_fpcalc(path: str) -> tuple[float, bytes]:
    """Run the external ``fpcalc`` calculator and parse its output.

    The calculator is invoked once per track, so a batch import starts
    hundreds of short-lived subprocesses in a row. Every process and pipe
    handle created here is released before the function returns or raises,
    no matter whether the calculator is missing, is killed, exits
    non-zero, or prints no usable output -- otherwise the leaked
    descriptors accumulate over a long-running analysis session and
    eventually make unrelated later tasks fail.

    Raises :class:`acoustid.NoBackendError` when the calculator is not
    installed and :class:`acoustid.FingerprintGenerationError` for every
    other failure (empty output, non-zero exit status, unparseable or
    partial output).
    """
    fpcalc = os.environ.get(FPCALC_ENVVAR, FPCALC_COMMAND)
    command = [fpcalc, "-length", str(MAX_AUDIO_LENGTH), path]

    proc: subprocess.Popen[bytes] | None = None

    def reap() -> bytes:
        """Read the calculator output, reaping it even on failure.

        Returns the stdout bytes. On timeout or interruption the whole
        process group is killed and reaped so no descendant can keep
        the inherited stdout pipe (and thus this process's descriptor)
        open.
        """
        assert proc is not None
        try:
            output, _ = proc.communicate(timeout=FPCALC_TIMEOUT)
        except subprocess.TimeoutExpired:
            _terminate_proc_group(proc)
            output, _ = proc.communicate()
            raise acoustid.FingerprintGenerationError(
                f"fpcalc timed out after {FPCALC_TIMEOUT:g}s"
            ) from None
        except BaseException:
            # KeyboardInterrupt / SystemExit while reading: kill the
            # calculator and its descendants and reap before unwinding.
            _terminate_proc_group(proc)
            proc.wait()
            raise
        return output

    try:
        # ``devnull`` is held open only while spawning and reading; the
        # child inherits its descriptor, but the parent's copy is closed
        # by the ``with`` block. ``start_new_session`` puts the child in
        # its own process group so a stuck descendant can be reaped too.
        with open(os.devnull, "wb") as devnull:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=devnull,
                close_fds=True,
                start_new_session=True,
            )
            output = reap()
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            raise acoustid.NoBackendError("fpcalc not found") from exc
        raise acoustid.FingerprintGenerationError(
            f"fpcalc invocation failed: {exc}"
        ) from exc

    # communicate() has reaped the process; returncode is always set.
    retcode = proc.returncode
    if retcode:
        raise acoustid.FingerprintGenerationError(
            f"fpcalc exited with status {retcode}"
        )

    duration: float | None = None
    fp: bytes | None = None
    for line in output.splitlines():
        parts = line.split(b"=", 1)
        if len(parts) != 2:
            # Tolerate stray or malformed lines rather than failing the
            # whole calculation on them.
            continue
        key, value = parts
        if key == b"DURATION":
            try:
                duration = float(value)
            except ValueError:
                raise acoustid.FingerprintGenerationError(
                    "fpcalc duration not numeric"
                ) from None
        elif key == b"FINGERPRINT":
            fp = value

    # Empty output, missing DURATION/FINGERPRINT lines, or an empty
    # fingerprint value are all treated as generation failures.
    if duration is None or not fp:
        raise acoustid.FingerprintGenerationError("missing fpcalc output")
    return duration, fp


def _terminate_proc_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and every descendant it created, then reap it."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        # Fall back to terminating just the direct child.
        proc.kill()
    try:
        proc.wait(timeout=FPCALC_TIMEOUT)
    except subprocess.TimeoutExpired:
        pass


def _fingerprint_file(path: str) -> tuple[float, bytes]:
    """Fingerprint a file, owning the resource lifecycle of whichever
    backend is used.

    When the Chromaprint library is available, fingerprinting happens
    in-process and no operating-system handles are involved. Otherwise
    the external ``fpcalc`` calculator is launched through
    :func:`_run_fpcalc`, which guarantees that every process and pipe
    handle is released even on failure.
    """
    if acoustid.have_audioread and acoustid.have_chromaprint:
        return acoustid.fingerprint_file(path)
    return _run_fpcalc(path)


def acoustid_match(log: Logger, path: bytes) -> None:
    """Gets metadata for a file from Acoustid and populates the
    _matches, _fingerprints, and _acoustids dictionaries accordingly.
    """
    try:
        duration, fp = _fingerprint_file(util.syspath(path))
    except acoustid.FingerprintGenerationError as exc:
        log.error(
            "fingerprinting of {} failed: {}",
            util.displayable_path(repr(path)),
            exc,
        )
        return
    fp = fp.decode()
    _fingerprints[path] = fp
    try:
        res = acoustid.lookup(
            API_KEY, fp, duration, meta="recordings releases", timeout=10
        )
    except acoustid.AcoustidError as exc:
        log.debug(
            "fingerprint matching {} failed: {}",
            util.displayable_path(repr(path)),
            exc,
        )
        return
    log.debug("chroma: fingerprinted {}", util.displayable_path(repr(path)))

    # Ensure the response is usable and parse it.
    if res["status"] != "ok" or not res.get("results"):
        log.debug("no match found")
        return
    result = res["results"][0]  # Best match.
    if result["score"] < SCORE_THRESH:
        log.debug("no results above threshold")
        return
    _acoustids[path] = result["id"]

    # Get recording and releases from the result
    if not result.get("recordings"):
        log.debug("no recordings found")
        return
    recording_ids = []
    releases = []
    for recording in result["recordings"]:
        recording_ids.append(recording["id"])
        if "releases" in recording:
            releases.extend(recording["releases"])

    # The releases list is essentially in random order from the Acoustid lookup
    # so we optionally sort it using the match.preferred configuration options.
    # 'original_year' to sort the earliest first and
    # 'countries' to then sort preferred countries first.
    country_patterns = config["match"]["preferred"]["countries"].as_str_seq()
    countries = [re.compile(pat, re.I) for pat in country_patterns]
    original_year = config["match"]["preferred"]["original_year"].get(bool)
    releases.sort(
        key=partial(
            releases_key, countries=countries, original_year=original_year
        )
    )
    release_ids = [rel["id"] for rel in releases]

    log.debug(
        "matched recordings {} on releases {}", recording_ids, release_ids
    )
    _matches[path] = recording_ids, release_ids


# Plugin structure and autotagging logic.


def _all_releases(items: Sequence[Item]) -> Iterator[str]:
    """Given an iterable of Items, determines (according to Acoustid)
    which releases the items have in common. Generates release IDs.
    """
    # Count the number of "hits" for each release.
    relcounts = defaultdict[str, int](int)
    for item in items:
        if item.path not in _matches:
            continue

        _, release_ids = _matches[item.path]
        for release_id in release_ids:
            relcounts[release_id] += 1

    for release_id, count in relcounts.items():
        if float(count) / len(items) > COMMON_REL_THRESH:
            yield release_id


class AcoustidPlugin(MetadataSourcePlugin):
    def __init__(self) -> None:
        super().__init__()
        self.config.add({"auto": True})
        config["acoustid"]["apikey"].redact = True

        if self.config["auto"]:
            self.register_listener("import_task_start", self.fingerprint_task)
        self.register_listener("import_task_apply", apply_acoustid_metadata)

    @cached_property
    def mb(self) -> MusicBrainzPlugin | None:
        """The loaded MusicBrainz plugin, or ``None``.

        Acoustid lookups return MusicBrainz IDs, so chroma needs the
        ``musicbrainz`` plugin to resolve them into album/track
        candidates. When the user has not enabled ``musicbrainz``,
        chroma must not produce any candidates.

        Uses the plugin registry so that any plugin that swaps the
        musicbrainz instance at runtime (e.g. :doc:`plugins/mbpseudo`)
        is respected.
        """
        plugin = get_metadata_source("musicbrainz")
        if plugin is None:
            self._log.debug(
                "musicbrainz plugin not enabled; "
                "acoustid matches will not produce candidates"
            )
        return plugin  # type: ignore[return-value]

    def fingerprint_task(
        self, task: ImportTask, session: ImportSession
    ) -> None:
        return fingerprint_task(self._log, task, session)

    def track_distance(self, item: Item, info: TrackInfo) -> Distance:
        dist = Distance()
        if item.path not in _matches or not info.track_id:
            # Match failed or no track ID.
            return dist

        recording_ids, _ = _matches[item.path]
        dist.add_expr("track_id", info.track_id not in recording_ids)
        return dist

    def candidates(
        self, items: Sequence[Item], artist: str, album: str, va_likely: bool
    ) -> list[AlbumInfo]:
        if self.mb is None:
            return []

        albums = [
            a
            for relid in prefix(_all_releases(items), MAX_RELEASES)
            if (a := self.mb.album_for_id(relid))
        ]

        self._log.debug("acoustid album candidates: {}", len(albums))
        return albums

    def item_candidates(
        self, item: Item, artist: str, title: str
    ) -> Iterable[TrackInfo]:
        if item.path not in _matches:
            return []

        if self.mb is None:
            return []

        recording_ids, _ = _matches[item.path]
        tracks = []
        for recording_id in prefix(recording_ids, MAX_RECORDINGS):
            track = self.mb.track_for_id(recording_id)
            if track:
                tracks.append(track)
        self._log.debug("acoustid item candidates: {}", len(tracks))
        return tracks

    def album_for_id(self, *args, **kwargs) -> None:
        # Lookup by fingerprint ID does not make too much sense.
        return None

    def track_for_id(self, *args, **kwargs) -> None:
        # Lookup by fingerprint ID does not make too much sense.
        return None

    def commands(self) -> list[ui.Subcommand]:
        submit_cmd = ui.Subcommand(
            "submit", help="submit Acoustid fingerprints"
        )

        def submit_cmd_func(
            lib: Library, opts: optparse.Values, args: list[str]
        ) -> None:
            try:
                apikey = config["acoustid"]["apikey"].as_str()
            except confuse.NotFoundError:
                raise UserError("no Acoustid user API key provided")
            submit_items(self._log, apikey, lib.items(args))

        submit_cmd.func = submit_cmd_func

        fingerprint_cmd = ui.Subcommand(
            "fingerprint", help="generate fingerprints for items without them"
        )

        def fingerprint_cmd_func(
            lib: Library, opts: optparse.Values, args: list[str]
        ) -> None:
            for item in lib.items(args):
                fingerprint_item(self._log, item, write=ui.should_write())

        fingerprint_cmd.func = fingerprint_cmd_func

        return [submit_cmd, fingerprint_cmd, self.chromasearch_cmd()]

    def chromasearch_cmd(self) -> ui.Subcommand:
        cmd = ui.Subcommand(
            "chromasearch", help="search local database by chroma fingerprint"
        )
        cmd.parser.add_path_option()
        cmd.parser.add_format_option()
        cmd.parser.add_option(
            "-s",
            "--search",
            dest="search",
            action="store",
            help="Fingerprint to search for (from the output of fpcalc -plain)",
        )
        cmd.parser.add_option(
            "-c",
            "--count",
            dest="count",
            action="store",
            default=5,
            type=int,
            help="Number of items in result",
        )
        cmd.parser.add_option(
            "--full",
            dest="full",
            action="store_true",
            help="Don't stop searching once we found an exact match",
        )
        cmd.parser.add_option(
            "-w",
            "--write",
            dest="write",
            action="store_true",
            help="Write computed fingerprints to files",
        )

        def search_cmd_func(
            lib: Library, opts: ChromaSearchCLIOpts, args: list[str]
        ) -> None:
            if not opts.search:
                raise UserError("no --search provided")
            if opts.count <= 0:
                raise UserError("--count must be > 0")

            target = (0, opts.search.encode("utf-8"))
            top = TopN(opts.count)

            for item in lib.items(args):
                fp = fingerprint_item(
                    self._log,
                    item,
                    write=ui.should_write(opts.write),
                    quiet=True,
                )
                if fp is None:
                    self._log.warning(f"{item}: could not compute fingerprint")
                    continue

                score = acoustid.compare_fingerprints(
                    target, (0, fp.encode("utf-8"))
                )

                if score == 1 and not opts.full:
                    ui.print_(
                        f"{colorize('text_success', 'Found exact match')}: {item}"
                    )
                    return

                if score > 0:
                    top.add(ScoredItem(item, score))

            for scored_item in top:
                ui.print_(str(scored_item))

        cmd.func = search_cmd_func

        return cmd


# Hooks into import process.


def fingerprint_task(
    log: Logger, task: ImportTask, session: ImportSession
) -> None:
    """Fingerprint each item in the task for later use during the
    autotagging candidate search.
    """
    for item in task.items:
        acoustid_match(log, item.path)


def apply_acoustid_metadata(task: ImportTask, session: ImportSession) -> None:
    """Apply Acoustid metadata (fingerprint and ID) to the task's items."""
    for item in task.imported_items():
        if item.path in _fingerprints:
            item.acoustid_fingerprint = _fingerprints[item.path]
        if item.path in _acoustids:
            item.acoustid_id = _acoustids[item.path]


# UI commands.


def submit_items(
    log: Logger, userkey: str, items: Sequence[Item], chunksize: int = 64
) -> None:
    """Submit fingerprints for the items to the Acoustid server."""
    # The running list of dictionaries to submit.
    data: list[JSONDict] = []

    def submit_chunk() -> None:
        """Submit the current accumulated fingerprint data."""
        log.info("submitting {} fingerprints", len(data))
        try:
            acoustid.submit(API_KEY, userkey, data, timeout=10)
        except acoustid.AcoustidError as exc:
            log.warning("acoustid submission error: {}", exc)
        del data[:]

    for item in items:
        fp = fingerprint_item(log, item, write=ui.should_write())

        # Construct a submission dictionary for this item.
        item_data = {"duration": int(item.length), "fingerprint": fp}
        if item.mb_trackid:
            item_data["mbid"] = item.mb_trackid
            log.debug("submitting MBID")
        else:
            item_data.update(
                {
                    "track": item.title,
                    "artist": item.artist,
                    "album": item.album,
                    "albumartist": item.albumartist,
                    "year": item.year,
                    "trackno": item.track,
                    "discno": item.disc,
                }
            )
            log.debug("submitting textual metadata")
        data.append(item_data)

        # If we have enough data, submit a chunk.
        if len(data) >= chunksize:
            submit_chunk()

    # Submit remaining data in a final chunk.
    if data:
        submit_chunk()


def fingerprint_item(
    log: Logger, item: Item, write: bool = False, quiet: bool = False
) -> str | None:
    """Get the fingerprint for an Item. If the item already has a
    fingerprint, it is not regenerated. If fingerprint generation fails,
    return None. If the items are associated with a library, they are
    saved to the database. If `write` is set, then the new fingerprints
    are also written to files' metadata.
    """
    # Get a fingerprint and length for this track.
    if not item.length:
        log.info("{.filepath}: no duration available", item)
    elif item.acoustid_fingerprint:
        if not quiet:
            if write:
                log.info("{.filepath}: fingerprint exists, skipping", item)
            else:
                log.info("{.filepath}: using existing fingerprint", item)
        return item.acoustid_fingerprint
    else:
        log.info("{.filepath}: fingerprinting", item)
        try:
            _, fp = _fingerprint_file(util.syspath(item.path))
            item.acoustid_fingerprint = fp.decode()
            if write:
                log.info("{.filepath}: writing fingerprint", item)
                item.try_write()
            if item._db:
                item.store()
            return item.acoustid_fingerprint
        except acoustid.FingerprintGenerationError as exc:
            log.info("fingerprint generation failed: {}", exc)
    return None


# Classes for search.


class ScoredItem:
    def __init__(self, item: Item, score: float) -> None:
        self.item = item
        self.score = score

    def __lt__(self, other: object) -> bool:
        return type(self) is type(other) and self.score < other.score

    def __gt__(self, other: object) -> bool:
        return type(self) is type(other) and self.score > other.score

    def __str__(self) -> str:
        percent = f"{round(self.score * 100, 2)}%".rjust(6)
        if self.score >= 0.95:
            percent = colorize("text_success", percent)
        elif self.score >= 0.85:
            percent = colorize("text_warning", percent)
        else:
            percent = colorize("text_error", percent)

        return f"[{percent}] {self.item}"


class TopN:
    def __init__(self, n: int) -> None:
        self.n = n
        self.heap: list[ScoredItem] = []

    def add(self, value: ScoredItem) -> None:
        if len(self.heap) < self.n:
            heapq.heappush(self.heap, value)
        else:
            if value > self.heap[0]:
                heapq.heapreplace(self.heap, value)

    def __iter__(self) -> Iterator[ScoredItem]:
        return iter(sorted(self.heap, reverse=True))
